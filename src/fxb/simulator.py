"""Single-position CFD simulator: decisions at close, fills at next open.

Bid OHLC cannot identify intrabar order. If both barriers hit, assume stop first.
Contract conversions/margin are snapshots; this is not a tick-accurate broker.
"""
from dataclasses import dataclass, asdict
from datetime import date
import numpy as np
from .data import day_key, iso
from .risk import Guard, Risk, levels, size_lots

WAIT, LONG, SHORT, CLOSE = range(4)
ACTION_NAMES = ["WAIT", "LONG", "SHORT", "CLOSE"]
PORTFOLIO_DIM = 5


def portfolio_state(side, age, unrealized, equity, guard, risk):
    return np.asarray([side, min(age / risk.max_hold_bars, 2),
        np.clip(unrealized / max(equity * risk.fraction, 1), -5, 5),
        max(0, 1 - equity / max(guard.day_start, 1)) / risk.daily_loss,
        max(0, 1 - equity / max(guard.peak, 1)) / risk.max_drawdown])


@dataclass
class Position:
    side: int
    entry: float
    stop: float
    target: float
    lots: float
    opened: int
    risk_cash: float
    fees: float


class Simulator:
    def __init__(self, bars, encoded, atr, start, end, risk=None, initial=10_000):
        if start < 64 or end <= start + 1 or end > len(bars.time):
            raise ValueError("Invalid simulation range")
        self.bars, self.x, self.atr = bars, encoded, atr
        self.start, self.end = start, end
        self.risk = risk or Risk()
        if not np.isfinite(initial) or initial <= 0:
            raise ValueError("Initial balance must be positive")
        self.initial = initial
        self.reset()

    def reset(self):
        self.i = self.start
        self.balance = self.equity = self.initial
        self.guard = Guard()
        self.guard.update(self.equity, day_key(self.bars.time[self.i] + 300), self.risk)
        self.position = None
        self.trades, self.curve = [], [self.initial]
        self.done = False
        return self.state()

    def state(self):
        p = self.position
        extra = portfolio_state(p.side if p else 0, self.i - p.opened + 1 if p else 0,
                                self.equity - self.balance, self.equity, self.guard, self.risk)
        return np.r_[self.x[self.i], extra]

    def mask(self):
        if self.position:
            return np.array([True, False, False, True])
        allowed = not self.guard.reason(self.risk)
        return np.array([True, allowed, allowed, False])

    def exit(self, price, reason, j):
        p, spec = self.position, self.bars.spec
        fill = price - p.side * spec.slippage_points * spec.point
        pnl = spec.pnl(p.side, p.entry, fill, p.lots)
        exit_fee = p.lots * spec.commission_per_lot_side
        self.balance += pnl - exit_fee
        net = pnl - exit_fee - p.fees
        self.trades.append({**asdict(p), "entry_time": iso(self.bars.time[p.opened]),
                            "exit_time": iso(self.bars.time[j] + 300), "exit": fill,
                            "pnl": net, "r_multiple": net / p.risk_cash, "reason": reason})
        self.position = None

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has finished")
        if type(action) is not int or action not in range(4) or not self.mask()[action]:
            raise ValueError("Invalid or masked action")
        b, risk, spec = self.bars, self.risk, self.bars.spec
        j = self.i + 1
        before = self.equity
        prev_dd = max(0, 1 - before / self.guard.peak)
        self.guard.update(before, day_key(b.time[j]), risk)
        opened = False
        if self.position:
            p = self.position
            nights = (date.fromisoformat(day_key(b.time[j])) - date.fromisoformat(day_key(b.time[self.i]))).days
            if nights > 0:
                fee = p.lots * spec.overnight_cost_per_lot * nights
                self.balance -= fee
                p.fees += fee
            # All close decisions were known before this opening price.
            if action == CLOSE or j - p.opened >= risk.max_hold_bars or self.guard.daily_halt or self.guard.total_halt:
                quote = b.open[j] + (b.spread[j] if p.side < 0 else 0)
                self.exit(quote, "policy" if action == CLOSE else "risk_or_time", j)
        elif action in (LONG, SHORT) and not self.guard.reason(risk):
            if b.spread[j] <= risk.max_spread_atr * self.atr[self.i]:
                side = 1 if action == LONG else -1
                entry = b.open[j] + (b.spread[j] if side > 0 else 0) + side * spec.slippage_points * spec.point
                stop, target = levels(side, entry, self.atr[self.i], spec, risk)
                loss = abs(spec.pnl(side, entry, stop - side * spec.slippage_points * spec.point, 1)) + 2 * spec.commission_per_lot_side
                lots = size_lots(before, loss, before, spec.margin_per_lot, spec, risk)
                if lots:
                    fees = lots * spec.commission_per_lot_side
                    self.balance -= fees
                    self.position = Position(side, entry, stop, target, lots, j, loss * lots, fees)
                    self.guard.entries += 1
                    opened = True
        p = self.position
        if p:
            spread = b.spread[j] if p.side < 0 else 0
            lo, hi, op = b.low[j] + spread, b.high[j] + spread, b.open[j] + spread
            stop_hit = lo <= p.stop if p.side > 0 else hi >= p.stop
            target_hit = hi >= p.target if p.side > 0 else lo <= p.target
            if stop_hit:
                self.exit(min(op, p.stop) if p.side > 0 else max(op, p.stop), "stop", j)
            elif target_hit:
                self.exit(p.target, "target", j)
        self.i = j
        self.done = j >= self.end - 1
        if self.done and self.position:
            quote = b.close[j] + (b.spread[j] if self.position.side < 0 else 0)
            self.exit(quote, "end_of_partition", j)
        self.equity = self.balance
        if self.position:
            p = self.position
            quote = b.close[j] + (b.spread[j] if p.side < 0 else 0)
            self.equity += spec.pnl(p.side, p.entry, quote, p.lots)
        self.guard.update(self.equity, day_key(b.time[j] + 300), risk)
        self.done |= self.guard.total_halt
        if self.done and self.position:
            p = self.position
            quote = b.close[j] + (b.spread[j] if p.side < 0 else 0)
            self.exit(quote, "drawdown_stop", j)
            self.equity = self.balance
        dd = max(0, 1 - self.equity / self.guard.peak)
        reward = (np.log(max(self.equity, 0.01) / before) / risk.fraction
                  - 0.5 * max(0, dd - prev_dd) / risk.fraction - 0.005 * opened)
        self.curve.append(self.equity)
        return self.state(), float(reward), bool(self.done), {"equity": self.equity, "opened": opened}

    def report(self):
        pnl = np.asarray([t["pnl"] for t in self.trades])
        wins = float(pnl[pnl > 0].sum())
        loss = float(-pnl[pnl < 0].sum())
        curve = np.asarray(self.curve)
        dd = 1 - curve / np.maximum.accumulate(curve)
        return {"return_pct": 100 * (self.equity / self.initial - 1),
                "max_drawdown_pct": 100 * float(dd.max()), "trades": len(pnl),
                "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
                "profit_factor": wins / loss if loss else None,
                "net_pnl": self.equity - self.initial, "final_equity": self.equity,
                "halted": bool(self.guard.total_halt), "bars_processed": len(curve) - 1,
                "requested_bars": self.end - self.start - 1,
                "synthetic": self.bars.synthetic}
