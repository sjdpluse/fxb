"""Strict UTC bid-bar ingestion and causal, finite-window observations."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from .risk import Spec

UTC = timezone.utc
KABUL = timezone(timedelta(hours=4, minutes=30), "Asia/Kabul")
FEATURE_VERSION = "m5-bid-causal-v1"
FEATURE_NAMES = ["ret_1", "ret_3", "ret_12", "ret_48", "trend_12", "trend_48",
                 "atr_fraction", "volatility", "body_atr", "upper_wick_atr",
                 "lower_wick_atr", "breakout_high", "breakout_low", "volume_ratio",
                 "spread_atr", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
WARMUP = 64


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Timestamps must include UTC or an explicit offset")
    return dt.timestamp()


def iso(value):
    return datetime.fromtimestamp(float(value), UTC).isoformat()


def day_key(value):
    return datetime.fromtimestamp(float(value), KABUL).date().isoformat()


@dataclass
class Bars:
    time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    spread: np.ndarray  # price distance, not MT5 points
    spec: Spec
    source: str
    synthetic: bool

    def validate(self):
        arrays = [getattr(self, k) for k in ("time", "open", "high", "low", "close", "volume", "spread")]
        n = len(self.time)
        if n < WARMUP + 2 or any(a.shape != (n,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Bars must be finite, equally sized and contain at least 66 rows")
        if np.any(np.diff(self.time) <= 0):
            raise ValueError("Duplicate or unordered timestamps; input is never silently sorted")
        if np.any(self.time % self.spec.timeframe_seconds != 0):
            raise ValueError("Bar opening times must be aligned to M5 UTC boundaries")
        if (np.any(self.low <= 0) or np.any(self.high < np.maximum(self.open, self.close))
                or np.any(self.low > np.minimum(self.open, self.close))
                or np.any(self.spread < 0) or np.any(self.volume < 0)):
            raise ValueError("Invalid OHLC, volume or spread")
        if not self.source or type(self.synthetic) is not bool:
            raise ValueError("Explicit source and synthetic flag required")
        return self

    def digest(self):
        h = hashlib.sha256(json.dumps(asdict(self.spec), sort_keys=True).encode())
        for key in ("time", "open", "high", "low", "close", "volume", "spread"):
            h.update(np.asarray(getattr(self, key), dtype="<f8").tobytes())
        h.update(json.dumps([self.source, self.synthetic]).encode())
        return h.hexdigest()


def load_bars(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if meta.get("price_basis") != "bid" or meta.get("spread_unit") != "price":
        raise ValueError("Metadata must specify bid bars and spread_unit=price")
    fields = ["time", "open", "high", "low", "close", "volume", "spread"]
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.asarray([timestamp(r[k]) if k == "time" else float(r[k]) for r in rows]) for k in fields}
    return Bars(**cols, spec=Spec(**meta["spec"]), source=meta["source"], synthetic=meta["synthetic"]).validate()


def save_bars(bars, path):
    bars.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["time", "open", "high", "low", "close", "volume", "spread"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(len(bars.time)):
            w.writerow([iso(bars.time[i])] + [getattr(bars, k)[i] for k in fields[1:]])
    path.with_suffix(".meta.json").write_text(json.dumps({
        "spec": asdict(bars.spec), "source": bars.source, "synthetic": bars.synthetic,
        "price_basis": "bid", "spread_unit": "price"}, indent=2), encoding="utf-8")


def synthetic_bars(count=2400, seed=7, symbol="XAUUSD"):
    """Manufactured test fixture; its prices and returns are NOT market evidence."""
    if symbol not in ("XAUUSD", "US100") or count < 800:
        raise ValueError("Synthetic fixtures support XAUUSD/US100 and at least 800 bars")
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=UTC).timestamp()
    times = start + np.arange(count) * 300
    base = 2000 if symbol == "XAUUSD" else 15000
    phase = np.arange(count) // 250 % 3
    drift = np.where(phase == 0, 0.00012, np.where(phase == 1, -0.00012, 0))
    close = base * np.exp(np.cumsum(rng.normal(drift, 0.0005)))
    opens = np.r_[base, close[:-1]]
    extra = base * rng.uniform(0.0001, 0.0005, count)
    point = 0.01 if symbol == "XAUUSD" else 0.1
    spec = Spec(symbol, "USD", 300, point, point, 1.0 if symbol == "XAUUSD" else 0.1,
                0.01, 20, 0.01, 1000, 3, 2, 10)
    return Bars(times, opens, np.maximum(opens, close) + extra,
                np.minimum(opens, close) - extra, close, rng.integers(50, 500, count).astype(float),
                np.full(count, point * 15), spec, "synthetic-test-fixture", True).validate()


def features(bars):
    n = len(bars.time)
    out = np.zeros((n, len(FEATURE_NAMES)))
    atr = np.zeros(n)
    c = bars.close
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(bars.high - bars.low, np.maximum(abs(bars.high - prev), abs(bars.low - prev)))
    returns = np.log(c / prev)
    for i in range(n):
        a = atr[i] = max(float(tr[max(0, i - 13):i + 1].mean()), bars.spec.tick_size)
        ret = [np.log(c[i] / c[max(0, i - k)]) for k in (1, 3, 12, 48)]
        trend = [(c[i] - c[max(0, i - k + 1):i + 1].mean()) / a for k in (12, 48)]
        old = slice(max(0, i - 20), max(1, i))
        dt = datetime.fromtimestamp(float(bars.time[i] + 300), UTC)
        hour = (dt.hour + dt.minute / 60) * 2 * np.pi / 24
        week = dt.weekday() * 2 * np.pi / 7
        out[i] = ret + trend + [a / c[i], returns[max(0, i - 23):i + 1].std(),
            (c[i] - bars.open[i]) / a,
            (bars.high[i] - max(c[i], bars.open[i])) / a,
            (min(c[i], bars.open[i]) - bars.low[i]) / a,
            (c[i] - bars.high[old].max()) / a, (c[i] - bars.low[old].min()) / a,
            bars.volume[i] / max(1, bars.volume[max(0, i - 19):i + 1].mean()),
            bars.spread[i] / a, np.sin(hour), np.cos(hour), np.sin(week), np.cos(week)]
    return out, atr


class Scaler:
    def __init__(self, mean, scale):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)

    @classmethod
    def fit(cls, train_only):
        return cls(train_only.mean(axis=0), np.maximum(train_only.std(axis=0), 1e-8))

    def transform(self, x):
        return np.clip((x - self.mean) / self.scale, -8, 8)
