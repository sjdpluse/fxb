"""Causal M1 structure and a bounded, reward-trained connectome readout.

These are explicit ICT-inspired rules, not a claim that a biological graph
understands trading. Confirmed pivots require two CLOSED bars on the right.
The rolling 500-bar context is not an out-of-sample performance estimate.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import numpy as np
from .risk import Risk

VERSION = "gold-m1-structure-v1"
FEATURES = ["return_1_atr", "return_3_atr", "body_atr", "upper_wick_atr",
            "lower_wick_atr", "range_atr", "distance_swing_high_atr",
            "distance_swing_low_atr", "swing_trend", "structure_break",
            "liquidity_sweep", "displacement", "fvg_side", "distance_fvg_atr",
            "fvg_width_atr", "spread_atr", "relative_tick_volume", "closed_m5_bias",
            "premium_discount"]
READOUT_DIM = 86  # 19 engineered + 64 neural + direction + setup type + bias


@dataclass(frozen=True)
class ScalpConfig:
    context_bars: int = 500
    pivot_width: int = 2
    gap_lifetime: int = 12
    cooldown_seconds: int = 60
    entry_window_seconds: int = 20
    loss_pause_seconds: int = 900
    max_consecutive_losses: int = 3
    min_score: float = .60
    learning_rate: float = .08
    max_adaptation: float = .20
    max_weight_norm: float = .75

    def __post_init__(self):
        if (self.context_bars != 500 or self.pivot_width != 2 or self.gap_lifetime != 12
                or not 30 <= self.cooldown_seconds <= 3600
                or not 1 <= self.entry_window_seconds <= 30
                or not 60 <= self.loss_pause_seconds <= 86400
                or not 1 <= self.max_consecutive_losses <= 10):
            raise ValueError("Invalid M1 context, structure or timing configuration")
        if any(type(getattr(self, k)) is not int for k in
               ("context_bars", "pivot_width", "gap_lifetime", "cooldown_seconds",
                "entry_window_seconds", "loss_pause_seconds", "max_consecutive_losses")):
            raise ValueError("Scalper counts and seconds must be integers")
        if (not all(math.isfinite(x) for x in asdict(self).values())
                or not .5 <= self.min_score <= .9 or not 0 < self.learning_rate <= .1
                or not 0 < self.max_adaptation <= .2 or not 0 < self.max_weight_norm <= 1):
            raise ValueError("Online adaptation must stay within fixed bounds")


def scalp_risk():
    return Risk(fraction=.001, daily_loss=.01, max_drawdown=.05, max_trades=40,
                stop_atr=1, reward_risk=1.5, max_hold_bars=5, max_spread_atr=.20,
                margin_fraction=.20)


def analyze(bars, config=None):
    """Return per-close features/signals. Appending future bars cannot edit a row."""
    cfg = config or ScalpConfig()
    bars.validate()
    if bars.spec.timeframe_seconds != 60:
        raise ValueError("The scalp adapter requires real M1 bid bars, not resampled M5")
    n = len(bars.time)
    out, atr = np.zeros((n, 19)), np.zeros(n)
    signals, structures = [None] * n, []
    highs, lows, m5_closes = [], [], []
    last_break, sweeps, gaps = {1: -100, -1: -100}, {}, []
    for i in range(n):
        t, c, o, h, l = (float(getattr(bars, k)[i]) for k in ("time", "close", "open", "high", "low"))
        first = max(0, i - 13)
        previous = np.r_[bars.close[max(0, first-1)], bars.close[first:i]]
        tr = np.maximum(bars.high[first:i+1] - bars.low[first:i+1], np.maximum(
            abs(bars.high[first:i+1] - previous), abs(bars.low[first:i+1] - previous)))
        a = atr[i] = max(float(tr.mean()), bars.spec.tick_size)
        if i and t - bars.time[i-1] != 60:
            gaps, sweeps = [], {}
            last_break = {1: -100, -1: -100}
        width, j = cfg.pivot_width, i - cfg.pivot_width
        if j >= width:
            window = slice(j-width, j+width+1)
            if np.all(np.diff(bars.time[window]) == 60):
                if bars.high[j] == bars.high[window].max() and np.sum(bars.high[window] == bars.high[j]) == 1:
                    highs.append((j, float(bars.high[j])))
                if bars.low[j] == bars.low[window].min() and np.sum(bars.low[window] == bars.low[j]) == 1:
                    lows.append((j, float(bars.low[j])))
        sh = highs[-1][1] if highs else None
        sl = lows[-1][1] if lows else None
        prev = float(bars.close[max(0, i-1)])
        bos = 1 if sh and prev <= sh < c else -1 if sl and prev >= sl > c else 0
        sweep = 1 if sl and l < sl < c else -1 if sh and h > sh > c else 0
        if sweep:
            sweeps[sweep] = (i, l if sweep == 1 else h)
        if bos:
            last_break[bos] = i
        # M5 confirmation uses only complete, contiguous five-minute blocks.
        if i >= 4 and t % 300 == 240 and np.all(np.diff(bars.time[i-4:i+1]) == 60):
            m5_closes.append(c)
        bias = int(np.sign(m5_closes[-1] - np.mean(m5_closes[-4:]))) if len(m5_closes) >= 4 else 0
        trend = 0
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]: trend = 1
            elif highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]: trend = -1
        # Evaluate existing gaps before creating this bar's new gap: a gap
        # cannot be traded on the candle that makes it observable.
        kept, entry_gap = [], None
        for gap in gaps:
            side = gap["side"]
            invalid = (c < gap["low"] if side > 0 else c > gap["high"])
            if i - gap["born"] > cfg.gap_lifetime or invalid:
                continue
            touched = l <= gap["high"] and h >= gap["low"]
            if touched:
                rejection = c >= gap["high"] and c > o if side > 0 else c <= gap["low"] and c < o
                aligned = bias in (0, side)
                if rejection and aligned and i >= 64:
                    entry_gap = gap
                    quality = .68 if gap["kind"] == "sweep_mss_fvg" else .63
                    quality += .05 if bias == side else 0
                    signals[i] = {"side": side, "kind": gap["kind"], "score": quality,
                        "stop_bid": gap["stop"], "gap_low": gap["low"], "gap_high": gap["high"],
                        "setup_id": f"{VERSION}:{int(bars.time[gap['born']])}:{side}",
                        "formed_at": float(bars.time[gap["born"]] + 60)}
                # A used/touched gap is consumed even if the first test fails.
            else:
                kept.append(gap)
        gaps = kept
        if i >= 2 and np.all(np.diff(bars.time[i-2:i+1]) == 60):
            middle_body = float(bars.close[i-1] - bars.open[i-1])
            side = 1 if l > bars.high[i-2] and middle_body > .8 * atr[i-1] else (
                -1 if h < bars.low[i-2] and middle_body < -.8 * atr[i-1] else 0)
            if side and 0 <= i-last_break[side] <= 3:
                low, high = (float(bars.high[i-2]), l) if side > 0 else (h, float(bars.low[i-2]))
                if high - low >= .08 * a:
                    swept = sweeps.get(side)
                    reversal = bool(swept and 0 <= last_break[side]-swept[0] <= 8)
                    anchor = swept[1] if reversal else (float(bars.low[i-2:i+1].min()) if side > 0 else float(bars.high[i-2:i+1].max()))
                    gaps.append({"born": i, "side": side, "low": low, "high": high,
                                 "stop": anchor - side * .15 * a,
                                 "kind": "sweep_mss_fvg" if reversal else "bos_fvg_retest"})
        gap = entry_gap or (gaps[-1] if gaps else None)
        upper, lower = float(bars.high[max(0, i-49):i+1].max()), float(bars.low[max(0, i-49):i+1].min())
        out[i] = [(c-prev)/a, (c-bars.close[max(0, i-3)])/a, (c-o)/a,
                  (h-max(c,o))/a, (min(c,o)-l)/a, (h-l)/a,
                  (c-sh)/a if sh else 0, (c-sl)/a if sl else 0, trend, bos, sweep,
                  (c-o)/a if abs(c-o) > .8*a else 0, gap["side"] if gap else 0,
                  (c-(gap["low"]+gap["high"])/2)/a if gap else 0,
                  (gap["high"]-gap["low"])/a if gap else 0, bars.spread[i]/a,
                  bars.volume[i]/max(1, float(bars.volume[max(0, i-19):i+1].mean()))-1,
                  bias, (2*c-upper-lower)/max(upper-lower, bars.spec.tick_size)]
        structures.append({"swing_high": sh, "swing_low": sl, "swing_trend": trend,
            "structure_break": bos, "liquidity_sweep": sweep, "m5_bias": bias,
            "fvg": dict(gap) if gap else None})
    return np.clip(out, -8, 8), atr, signals, structures


def readout_context(embedding, signal):
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape != (83,) or not np.isfinite(embedding).all():
        raise ValueError("Expected the complete 19+64 connectome readout")
    return np.r_[np.tanh(embedding / 4), signal["side"],
                 1 if signal["kind"] == "sweep_mss_fvg" else -1, 1.0]


def learning_state(memory, scope):
    value = memory.get_state("scalp-learning:" + scope, {})
    return {"weights": [0.] * READOUT_DIM, "samples": 0, "wins": 0, "losses": 0,
            "net_cashflow": 0., "sum_r": 0., "loss_streak": 0, "last_exit": 0., **value}


def entry_score(state, vector, base, config):
    weights = np.asarray(state["weights"], dtype=float)
    if (weights.shape != (READOUT_DIM,) or not np.isfinite(weights).all()
            or np.linalg.norm(weights) > config.max_weight_norm + 1e-8):
        raise ValueError("Invalid or out-of-bounds online readout")
    adjustment = config.max_adaptation * float(np.tanh(weights @ vector))
    return float(base + adjustment), adjustment


def learn_closed(memory, scope, position_id, symbol, vector, net_cashflow, risk_cash,
                 opened_at, closed_at, observed_at, config=None):
    """Commit event + weight update together. Replaying a deal cannot learn twice."""
    cfg = config or ScalpConfig()
    vector = np.asarray(vector, dtype=float)
    if (vector.shape != (READOUT_DIM,) or not np.isfinite(vector).all()
            or not np.isfinite([net_cashflow, risk_cash, opened_at, closed_at, observed_at]).all()
            or risk_cash <= 0 or not opened_at <= closed_at <= observed_at):
        raise ValueError("Invalid settled-trade learning sample")
    reward = float(net_cashflow / risk_cash)
    event_id = hashlib.sha256(f"{scope}:settled:{position_id}".encode()).hexdigest()
    # Do not call Memory.record()/put_state() here: their nested commits would
    # split this exactly-once update into multiple crash boundaries.
    with memory.db:
        cursor = memory.db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,0)",
            (event_id, scope, symbol, "scalp_reward", opened_at, observed_at,
             json.dumps({"position_id": position_id, "net_cashflow": net_cashflow,
                        "risk_cash": risk_cash, "reward_r": reward, "closed_at": closed_at,
                        "state": vector.tolist()}, allow_nan=False)))
        if cursor.rowcount == 0:
            return False
        state = learning_state(memory, scope)
        entry_score(state, vector, 0, cfg)  # validate persisted weights
        weights = np.asarray(state["weights"])
        error = float(np.clip(np.clip(reward, -2, 2) - weights @ vector, -1, 1))
        weights = weights * .999 + cfg.learning_rate * error * vector / (1 + vector @ vector)
        norm = np.linalg.norm(weights)
        if norm > cfg.max_weight_norm: weights *= cfg.max_weight_norm / norm
        state.update(weights=weights.tolist(), samples=state["samples"]+1,
                     wins=state["wins"]+int(net_cashflow > 0), losses=state["losses"]+int(net_cashflow < 0),
                     net_cashflow=state["net_cashflow"]+net_cashflow, sum_r=state["sum_r"]+reward,
                     loss_streak=state["loss_streak"]+1 if net_cashflow < 0 else 0,
                     last_exit=max(state["last_exit"], closed_at))
        memory.db.execute("INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                          ("scalp-learning:"+scope, json.dumps(state, allow_nan=False)))
    return True
