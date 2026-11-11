"""Explicit broker wall-clock conversion. Never infer an offset from an old quote.

MT5 transports may expose either UTC epochs or server wall-clock digits encoded
as epochs. UTC remains the default. Non-UTC sources require an operator-supplied
policy; fixed offsets also require a bounded validity interval. IANA zones and
the NY+7 convention handle historical DST only when that rule is confirmed for
the particular server. Ambiguous/nonexistent local times are rejected.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = timezone.utc


def instant(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Clock validity dates must be nonempty ISO timestamps")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Clock validity dates require an explicit UTC offset")
    return dt.timestamp()


@dataclass(frozen=True)
class BrokerClock:
    zone: str = "UTC"
    valid_from: str | None = None
    valid_until: str | None = None

    def __post_init__(self):
        if not isinstance(self.zone, str):
            raise ValueError("Invalid broker clock zone")
        fixed = re.fullmatch(r"([+-])(\d{2}):(\d{2})", self.zone)
        shift = 0
        if self.zone == "UTC":
            tz = UTC
        elif fixed:
            hours, minutes = int(fixed[2]), int(fixed[3])
            if minutes >= 60 or hours > 14 or hours == 14 and minutes or minutes % 5:
                raise ValueError("Broker UTC offset must be within 14 hours and aligned to M5")
            offset = (hours * 60 + minutes) * (1 if fixed[1] == "+" else -1)
            tz = timezone(timedelta(minutes=offset))
            if not self.valid_from or not self.valid_until:
                raise ValueError("Fixed broker offsets require valid_from and valid_until; do not apply today's offset to all history")
        else:
            name = "America/New_York" if self.zone == "NY+7" else self.zone
            shift = 7 * 3600 if self.zone == "NY+7" else 0
            try:
                tz = ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError) as e:
                raise ValueError("Unknown broker time zone; install tzdata on Windows or use a bounded offset") from e
        lower = instant(self.valid_from) if self.valid_from is not None else -math.inf
        upper = instant(self.valid_until) if self.valid_until is not None else math.inf
        if lower >= upper:
            raise ValueError("Clock valid_from must precede valid_until")
        object.__setattr__(self, "_tz", tz)
        object.__setattr__(self, "_shift", shift)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def from_config(cls, value=None):
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - {"zone", "valid_from", "valid_until"}:
            raise ValueError("Invalid broker clock configuration")
        return cls(**value)

    def config(self):
        return {"zone": self.zone, "valid_from": self.valid_from, "valid_until": self.valid_until}

    def compatible(self, previous):
        prior = BrokerClock.from_config(previous)
        return self.zone == prior.zone and self.lower <= prior.lower and self.upper >= prior.upper

    def check(self, utc):
        if not math.isfinite(utc) or not self.lower <= utc < self.upper:
            raise ValueError("Broker clock policy does not cover this date; confirm the historical DST rule or use data inside its validity interval")
        return utc

    def to_utc(self, raw):
        raw = float(raw)
        if not math.isfinite(raw):
            raise ValueError("Non-finite broker timestamp")
        if self.zone == "UTC":
            return self.check(raw)
        wall = datetime.fromtimestamp(raw - self._shift, UTC).replace(tzinfo=None)
        candidates = set()
        for fold in (0, 1):
            candidate = wall.replace(tzinfo=self._tz, fold=fold).timestamp()
            if datetime.fromtimestamp(candidate, self._tz).replace(tzinfo=None) == wall:
                candidates.add(candidate)
        if len(candidates) != 1:
            raise ValueError("Ambiguous or nonexistent broker timestamp at a DST transition")
        return self.check(candidates.pop())

    def from_utc(self, utc):
        self.check(float(utc))
        wall = datetime.fromtimestamp(utc, self._tz).replace(tzinfo=UTC)
        return wall.timestamp() + self._shift

    def query_bounds(self, start, end):
        self.check(float(start))
        self.check(float(end))
        if start > end:
            raise ValueError("Invalid history interval")
        # Query a superset and then filter normalized deal times. This also
        # covers a repeated hour when an interval crosses a DST transition.
        padding = 0 if self.zone == "UTC" else 14 * 3600
        return datetime.fromtimestamp(start - padding, UTC), datetime.fromtimestamp(end + padding, UTC)
