"""Explicit simulation funding schedule, independent of the Kabul risk day.

Amounts remain a declared account-currency cost scenario, not a reconstructed
broker swap tariff. Weekday weights are Monday through Sunday.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, time, timedelta
import math
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Rollover:
    zone: str = "UTC"
    hour: int = 0
    minute: int = 0
    weekday_multipliers: tuple = (1, 1, 1, 1, 1, 1, 1)
    weekday_offset: int = 0

    def __post_init__(self):
        ZoneInfo(self.zone)
        if (type(self.hour) is not int or not 0 <= self.hour < 24
                or type(self.minute) is not int or not 0 <= self.minute < 60
                or self.minute % 5 or type(self.weekday_offset) is not int
                or self.weekday_offset not in (-1, 0)):
            raise ValueError("Invalid rollover time or weekday offset")
        weights = tuple(self.weekday_multipliers)
        if len(weights) != 7 or any(not math.isfinite(v) or v < 0 for v in weights):
            raise ValueError("Rollover requires seven nonnegative weekday multipliers")
        object.__setattr__(self, "weekday_multipliers", weights)

    @classmethod
    def from_config(cls, value=None):
        if value is not None and (not isinstance(value, dict)
                or set(value) - {"zone", "hour", "minute", "weekday_multipliers", "weekday_offset"}):
            raise ValueError("Invalid rollover configuration")
        return cls(**(value or {}))

    def config(self):
        return asdict(self) | {"weekday_multipliers": list(self.weekday_multipliers)}

    def units(self, previous, current):
        """Count cutoff events in (previous, current], including history gaps."""
        if not math.isfinite(previous) or not math.isfinite(current) or current < previous:
            raise ValueError("Invalid rollover interval")
        tz = ZoneInfo(self.zone)
        first = datetime.fromtimestamp(previous, tz).date()
        last = datetime.fromtimestamp(current, tz).date()
        total = 0.0
        for days in range((last - first).days + 1):
            day = first + timedelta(days=days)
            wall = datetime.combine(day, time(self.hour, self.minute))
            candidates = set()
            for fold in (0, 1):
                candidate = wall.replace(tzinfo=tz, fold=fold).timestamp()
                if datetime.fromtimestamp(candidate, tz).replace(tzinfo=None) == wall:
                    candidates.add(candidate)
            if len(candidates) != 1:
                raise ValueError("Ambiguous or nonexistent rollover time at DST transition")
            event = candidates.pop()
            if previous < event <= current:
                total += self.weekday_multipliers[(day.weekday() + self.weekday_offset) % 7]
        return total
