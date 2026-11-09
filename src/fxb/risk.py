"""Independent risk constraints; model outputs cannot change these limits."""
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import math


@dataclass(frozen=True)
class Risk:
    fraction: float = 0.0025
    daily_loss: float = 0.02
    max_drawdown: float = 0.06
    max_trades: int = 6
    stop_atr: float = 2.0
    reward_risk: float = 2.0
    max_hold_bars: int = 12
    max_spread_atr: float = 0.20
    margin_fraction: float = 0.30

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError("Risk values must be finite and positive")
        if not 0 < self.fraction <= 0.01:
            raise ValueError("Per-trade risk must be at most 1%")
        if not self.fraction <= self.daily_loss <= self.max_drawdown < 0.5:
            raise ValueError("Risk limits must satisfy trade <= daily <= drawdown < 50%")
        if self.margin_fraction > 1 or self.max_spread_atr > 1:
            raise ValueError("Invalid margin or spread fraction")
        if type(self.max_trades) is not int or type(self.max_hold_bars) is not int:
            raise ValueError("Trade and holding limits must be integers")


@dataclass(frozen=True)
class Spec:
    symbol: str
    currency: str
    timeframe_seconds: int
    point: float
    tick_size: float
    tick_value: float
    lot_min: float
    lot_max: float
    lot_step: float
    margin_per_lot: float
    commission_per_lot_side: float
    slippage_points: float
    overnight_cost_per_lot: float
    stops_level: float = 0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in ("symbol", "currency"):
                if not value:
                    raise ValueError(f"Missing {name}")
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid contract field: {name}")
        for name in ("point", "tick_size", "tick_value", "lot_min", "lot_max", "lot_step", "margin_per_lot"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Contract field must be positive: {name}")
        if self.lot_max < self.lot_min or self.timeframe_seconds != 300:
            raise ValueError("This release requires M5 bid bars and valid lot bounds")

    def pnl(self, side, entry, exit_price, lots):
        return side * (exit_price - entry) / self.tick_size * self.tick_value * lots


def floor_lots(raw, minimum, maximum, step):
    """Never round risk up to the broker minimum."""
    if not all(math.isfinite(x) for x in (raw, minimum, maximum, step)):
        return 0.0
    if raw <= 0 or minimum <= 0 or maximum < minimum or step <= 0:
        return 0.0
    value = (Decimal(str(min(raw, maximum))) / Decimal(str(step))).to_integral_value(rounding=ROUND_FLOOR) * Decimal(str(step))
    return float(value) if value >= Decimal(str(minimum)) else 0.0


def round_price(value, tick_size, up=False):
    return float((Decimal(str(value)) / Decimal(str(tick_size))).to_integral_value(
        rounding=ROUND_CEILING if up else ROUND_FLOOR) * Decimal(str(tick_size)))


def levels(side, entry, atr, spec, risk):
    distance = max(atr * risk.stop_atr, spec.stops_level + spec.tick_size)
    stop = round_price(entry - side * distance, spec.tick_size, up=side < 0)
    target = round_price(entry + side * abs(entry - stop) * risk.reward_risk,
                         spec.tick_size, up=side > 0)
    if min(stop, target) <= 0:
        raise ValueError("Nonpositive stop or target")
    return stop, target


def size_lots(equity, stop_loss_per_lot, margin_free, margin_per_lot, spec, risk):
    if not all(math.isfinite(x) and x > 0 for x in
               (equity, stop_loss_per_lot, margin_free, margin_per_lot)):
        return 0.0
    budget_lots = equity * risk.fraction / stop_loss_per_lot
    margin_lots = margin_free * risk.margin_fraction / margin_per_lot
    return floor_lots(min(budget_lots, margin_lots), spec.lot_min, spec.lot_max, spec.lot_step)


@dataclass
class Guard:
    day: str = ""
    day_start: float = 0
    peak: float = 0
    entries: int = 0
    daily_halt: bool = False
    total_halt: bool = False

    def update(self, equity, day, risk):
        if not math.isfinite(equity) or equity <= 0:
            self.total_halt = True
            return
        if day != self.day:
            self.day, self.day_start, self.entries = day, equity, 0
            self.daily_halt = False
        self.peak = max(self.peak, equity)
        self.daily_halt |= equity <= self.day_start * (1 - risk.daily_loss)
        self.total_halt |= equity <= self.peak * (1 - risk.max_drawdown)

    def reason(self, risk):
        if self.total_halt:
            return "maximum_drawdown"
        if self.daily_halt:
            return "daily_loss"
        if self.entries >= risk.max_trades:
            return "daily_trade_limit"
        return ""
