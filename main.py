"""Cost-aware trend-following bot for cbBTC/USDC minute bars.

Strategy in one paragraph: stay in cash unless the market is in an
established up-move, then ride it with most of the equity and get out
when the move breaks. "Established" means the fast EMA sits above the
slow EMA and the trailing hour returned at least ENTRY_MIN_BPS: a full
round trip costs roughly 25-35 bps in fees, slippage and price impact,
so anything weaker is not worth paying for. Exits are a trailing stop
under the best close since entry, a hard stop under the fill price, a
trend-failure check, and a maximum hold. A drawdown guard locks out
new entries once equity falls DD_LOCK_BPS below its peak, so a bad
patch cannot compound. Position size is capped both as a share of
equity and by the pool depth the SDK reports, so our own price impact
stays a small, known part of the cost.

Everything is a deterministic function of the bars and fills seen so
far: no clock, no environment, no randomness at all. Indicators run in
float64 on +, -, *, / only, so two runs over the same bars agree.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- signal parameters (in bars and basis points) -------------------------
EMA_FAST = 20            # fast trend line, minutes
EMA_SLOW = 60            # slow trend line, minutes
MOM_LB = 60              # momentum lookback for the entry gate
ENTRY_MIN_BPS = 20       # trailing-hour return must clear this to enter
SHORT_LB = 15            # short lookback: the last quarter hour must agree
TREND_FAIL_LB = 30       # lookback for the trend-failure exit
TREND_FAIL_BPS = -25     # exit when the trailing half hour falls past this
TRAIL_BPS = 55           # trailing stop under the best close since entry
STOP_BPS = 100           # hard stop under the entry fill price
MAX_HOLD = 480           # never hold longer than 8 hours
COOLDOWN = 10            # bars to wait after an exit before re-entering
DD_LOCK_BPS = 600        # stop opening positions 6% below the equity peak
SIZE_BPS = 8500          # spend at most 85% of equity on one position
IMPACT_CAP_BPS = 9       # ... and at most this much of the pool's depth
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
        self.entry_seq = None          # batch seq at which the buy filled
        self.peak_close = None         # best close since the position opened
        self.cooldown_until = -1       # first seq a new entry is allowed
        self.peak_equity = None        # best equity seen, for the guard
        self.locked = False            # drawdown guard state

    # -- history ------------------------------------------------------------
    def trailing_bps(self, lookback):
        """Return over the last `lookback` bars in bps, or None if too short."""
        if len(self.closes) <= lookback:
            return None
        then = self.closes[-1 - lookback]
        return (self.closes[-1] / then - 1.0) * 10000.0

    # -- fills ----------------------------------------------------------------
    def read_fills(self, state, batch):
        for fill in batch.get("fills", []):
            if fill.get("symbol") != SYMBOL or fill.get("status") != "filled":
                continue
            if fill["side"] == "buy":
                self.entry_price = int(fill["price"])
                self.entry_seq = state.seq
                self.peak_close = self.closes[-1] if self.closes else None
            elif fill["side"] == "sell":
                self.entry_price = None
                self.entry_seq = None
                self.peak_close = None
                self.cooldown_until = state.seq + COOLDOWN

    # -- exits ------------------------------------------------------------------
    def exit_reason(self, state, close):
        held = state.positions[SYMBOL]
        if self.entry_price is None:
            # A fill we somehow did not record: fall back to the average cost
            # implied by the position's book cost (fees included).
            self.entry_price = int(
                held.cost * 10 ** 13 // (held.qty * 10 ** 6)
            )
            self.entry_seq = state.seq
            self.peak_close = close
        if self.peak_close is None or close > self.peak_close:
            self.peak_close = close
        if close * 10000 <= self.peak_close * (10000 - TRAIL_BPS):
            return "trailing stop"
        if close * 10000 <= self.entry_price * (10000 - STOP_BPS):
            return "hard stop"
        if self.fast.value is not None and self.slow.value is not None:
            if self.fast.value < self.slow.value:
                return "trend cross down"
        recent = self.trailing_bps(TREND_FAIL_LB)
        if recent is not None and recent <= TREND_FAIL_BPS:
            return "trend failure"
        if self.entry_seq is not None and state.seq - self.entry_seq >= MAX_HOLD:
            return "max hold"
        return None

    # -- entries ------------------------------------------------------------------
    def entry_size(self, state):
        """Quote units to spend: the smaller of the equity share and the
        depth cap that keeps our own impact near IMPACT_CAP_BPS."""
        by_equity = state.equity * SIZE_BPS // 10000
        depth = state.depth_at(SYMBOL)
        by_depth = None
        if depth:
            by_depth = depth[0] * IMPACT_CAP_BPS // 10000
        spend = by_equity
        if by_depth is not None and by_depth < spend:
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
        hour = self.trailing_bps(MOM_LB)
        if hour is None or hour < ENTRY_MIN_BPS:
            return False
        quarter = self.trailing_bps(SHORT_LB)
        if quarter is None or quarter <= 0.0:
            return False
        return True

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
            if self.exit_reason(state, close) is not None:
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
