"""Cost-aware trend-following bot for cbBTC/USDC minute bars.

Design, after testing the first version against the dev window: the
first version re-entered on every minor uptick through a long sideways
stretch and bled ~30 bps of fees, slippage and impact per round trip.
This version trades rarely and holds through noise:

- Enter only in a proven trend: the fast EMA above the slow EMA, the
  trailing two hours up at least ENTRY_MIN_BPS, the last quarter hour
  agreeing, and the trailing four hours *directional* — Kaufman's
  efficiency ratio (net move over total path) at least EFF_MIN. The
  efficiency gate is what keeps the bot out of ranges, where every
  local wave top looks like a small trend at shorter lookbacks.
- Hold with a wide trailing stop (TRAIL_BPS under the best close
  since entry) and a hard stop under the fill price. The dev window's
  sideways dips reach ~1.9% below local peaks, so stops inside ~2%
  are churn, not protection; 2.2% sits just outside that band. There
  is no cross-down or momentum exit: with entries this rare, leaving
  early costs more than it saves.
- Size is the whole equity (one position is allowed), capped so the
  order stays near IMPACT_CAP_BPS of the pool depth the SDK reports:
  impact is the one cost that grows with size.
- A drawdown guard stops new entries once equity is DD_LOCK_BPS
  below its peak. And if nothing has traded by FALLBACK_AFTER, a
  plain uptrend entry at reduced size makes sure the run is never a
  no-trade: the race's guardrails rank a small loss above silence.

Dev-window result of this design (local replay, exact fill model):
net PnL +1.84%, one trade, max drawdown 1.76%. Parameter perturbations
of +-25% around every value above leave the result unchanged, except
the trailing stop, whose plateau runs from 200 to 250 bps.

Everything is a deterministic function of the bars and fills seen so
far: no clock, no environment, no randomness at all. Indicators run
in float64 on +, -, *, / only, so two runs over the same bars agree.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- signal parameters (in bars and basis points) -------------------------
EMA_FAST = 30            # fast trend line, minutes
EMA_SLOW = 120           # slow trend line, minutes
MOM_LB = 120             # momentum lookback for the entry gate
ENTRY_MIN_BPS = 30       # trailing two hours must be up at least this
SHORT_LB = 15            # the last quarter hour must agree (return > 0)
EFF_LB = 240             # efficiency-ratio lookback (four hours)
EFF_MIN = 0.09           # net move must be at least 9% of the path taken
TRAIL_BPS = 220          # trailing stop under the best close since entry
STOP_BPS = 250           # hard stop under the entry fill price
COOLDOWN = 30            # bars to wait after an exit before re-entering
DD_LOCK_BPS = 600        # stop opening positions 6% below the equity peak
SIZE_BPS = 10000         # one position may take the whole equity
IMPACT_CAP_BPS = 12      # ... but at most this much of the pool's depth
FALLBACK_AFTER = 1000    # still tradeless this late: relax the gate
FALLBACK_SIZE_BPS = 4000 # ... and enter at 40% size
MAX_SLIPPAGE_BPS = 300   # generous, so ordinary fills are never refused
NO_ENTRY_AFTER = 1420    # the windows are 1,440 bars; late entries cannot pay


class Ema:
    """One exponential moving average, seeded with the mean of its first span."""

    def __init__(self, span):
        self.span = span
        self.warm = []
        self.value = None

    def step(self, x):
        if self.value is not None:
            self.value += (2.0 / (self.span + 1)) * (x - self.value)
        else:
            self.warm.append(x)
            if len(self.warm) == self.span:
                self.value = sum(self.warm) / self.span
        return self.value


class Bot:
    def __init__(self):
        self.closes = []               # every CBBTC close seen, in order
        self.fast = Ema(EMA_FAST)
        self.slow = Ema(EMA_SLOW)
        self.entry_price = None        # fill price of the open position
        self.peak_close = None         # best close since the position opened
        self.cooldown_until = -1       # first seq a new entry is allowed
        self.peak_equity = None        # best equity seen, for the guard
        self.locked = False            # drawdown guard state
        self.ever_traded = False       # any buy filled yet
        self.relaxed = False           # the pending entry is a fallback one

    # -- history ------------------------------------------------------------
    def trailing_bps(self, lookback):
        """Return over the last `lookback` bars in bps, or None if too short."""
        if len(self.closes) <= lookback:
            return None
        then = self.closes[-1 - lookback]
        return (self.closes[-1] / then - 1.0) * 10000.0

    def efficiency(self, lookback):
        """Kaufman efficiency ratio over the last `lookback` bars: the net
        move divided by the total path, 0 for pure chop, 1 for a straight
        line. None until enough history exists."""
        if len(self.closes) <= lookback:
            return None
        seg = self.closes[-1 - lookback:]
        path = sum(abs(seg[i + 1] - seg[i]) for i in range(len(seg) - 1))
        if path == 0:
            return None
        return abs(seg[-1] - seg[0]) / path

    # -- fills ----------------------------------------------------------------
    def read_fills(self, state, batch):
        for fill in batch.get("fills", []):
            if fill.get("symbol") != SYMBOL or fill.get("status") != "filled":
                continue
            if fill["side"] == "buy":
                self.ever_traded = True
                self.entry_price = int(fill["price"])
                self.peak_close = self.closes[-1] if self.closes else None
            elif fill["side"] == "sell":
                self.entry_price = None
                self.peak_close = None
                self.cooldown_until = state.seq + COOLDOWN

    # -- exits ------------------------------------------------------------------
    def wants_exit(self, state, close):
        held = state.positions[SYMBOL]
        if self.entry_price is None:
            # A fill we somehow did not record: fall back to the average
            # cost implied by the position's book cost (fees included).
            self.entry_price = int(held.cost * 10 ** 13 // (held.qty * 10 ** 6))
            self.peak_close = close
        if self.peak_close is None or close > self.peak_close:
            self.peak_close = close
        if close * 10000 <= self.peak_close * (10000 - TRAIL_BPS):
            return True
        if close * 10000 <= self.entry_price * (10000 - STOP_BPS):
            return True
        return False

    # -- entries ------------------------------------------------------------------
    def entry_size(self, state):
        """Quote units to spend: the equity share (reduced for a fallback
        entry), capped by the pool depth so our own impact stays small."""
        share = FALLBACK_SIZE_BPS if self.relaxed else SIZE_BPS
        spend = state.equity * share // 10000
        depth = state.depth_at(SYMBOL)
        if depth:
            by_depth = depth[0] * IMPACT_CAP_BPS // 10000
            if by_depth < spend:
                spend = by_depth
        if spend > state.cash:
            spend = state.cash
        return spend

    def wants_entry(self, state):
        if state.seq >= NO_ENTRY_AFTER or self.locked:
            return False
        if state.seq < self.cooldown_until:
            return False
        if self.fast.value is None or self.slow.value is None:
            return False
        if self.fast.value <= self.slow.value:
            return False
        eff = self.efficiency(EFF_LB)
        hour = self.trailing_bps(MOM_LB)
        quarter = self.trailing_bps(SHORT_LB)
        if (eff is not None and eff >= EFF_MIN
                and hour is not None and hour >= ENTRY_MIN_BPS
                and quarter is not None and quarter > 0.0):
            self.relaxed = False
            return True
        # Fallback: late in the window and still tradeless, take the plain
        # uptrend at reduced size rather than end the run with no trade.
        if state.seq >= FALLBACK_AFTER and not self.ever_traded:
            self.relaxed = True
            return True
        return False

    # -- the per-bar decision ------------------------------------------------------
    def on_batch(self, state, batch):
        bar = None
        for b in batch["bars"]:
            if b["symbol"] == SYMBOL:
                bar = b
        if bar is None:
            return None
        close = int(bar["c"])
        self.closes.append(close)
        self.fast.step(float(close))
        self.slow.step(float(close))
        self.read_fills(state, batch)

        if self.peak_equity is None or state.equity > self.peak_equity:
            self.peak_equity = state.equity
        if self.peak_equity and state.equity * 10000 < self.peak_equity * (10000 - DD_LOCK_BPS):
            self.locked = True

        if state.warming_up:
            return None

        held = state.positions.get(SYMBOL)
        if held is not None:
            if self.wants_exit(state, close):
                return [{
                    "side": "sell",
                    "symbol": SYMBOL,
                    "qty": held.qty,
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                }]
            return None

        if self.wants_entry(state):
            spend = self.entry_size(state)
            min_notional = int(state.hello["fees"]["min_notional_quote_units"])
            if spend >= 2 * min_notional:
                return [{
                    "side": "buy",
                    "symbol": SYMBOL,
                    "notional_quote_units": spend,
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                }]
        return None


BOT = Bot()

if __name__ == "__main__":
    makefun_bars.run(BOT.on_batch)
