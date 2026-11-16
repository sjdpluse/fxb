"""Causal active entries. Thresholds are engineering defaults, not fitted edges.

Closed-candle entries and fresh-tick breakouts share the independent broker risk
path. No target trade count, synthetic reward, or random trading is implemented.
"""
import numpy as np
from .scalping import ACTIVE_VERSION


def _signal(bars, i, side, kind, score, stop, reference, formed_at):
    return {"side": side, "kind": kind, "score": score, "stop_bid": float(stop),
            "entry_reference": float(reference), "max_entry_drift_atr": .5,
            "setup_id": f"{ACTIVE_VERSION}:{kind}:{int(bars.time[i])}:{side}",
            "formed_at": float(formed_at)}


def bar_signal(bars, i, atr, bias):
    if i < 64 or np.any(np.diff(bars.time[i-19:i+1]) != 60):
        return None
    o, h, l, c = (float(getattr(bars, k)[i]) for k in ("open", "high", "low", "close"))
    body, span = c-o, h-l
    if span <= 0 or span > 2.5 * atr:
        return None
    side = 1 if body > 0 else -1
    edge = float(bars.high[i-3:i].max() if side > 0 else bars.low[i-3:i].min())
    close_quality = (c-l)/span if side > 0 else (h-c)/span
    alignment = .03 if bias == side else -.03 if bias == -side else 0
    if (abs(body) >= .45*atr and abs(body)/span >= .6 and close_quality >= .75
            and .05*atr < side*(c-edge) <= atr):
        stop = min(edge-.1*atr, c-.6*atr) if side > 0 else max(edge+.1*atr, c+.6*atr)
        return _signal(bars, i, side, "momentum_breakout", .62+alignment,
                       stop, c, bars.time[i]+60)
    fast, previous_fast = float(bars.close[i-4:i+1].mean()), float(bars.close[i-5:i].mean())
    slow = float(bars.close[i-12:i+1].mean())
    touched = (bars.low[i-1] <= previous_fast+.15*atr if side > 0
               else bars.high[i-1] >= previous_fast-.15*atr)
    previous_edge = bars.high[i-1] if side > 0 else bars.low[i-1]
    if (side*(fast-slow) > .05*atr and side*(fast-previous_fast) > 0
            and side*(bars.close[i-1]-bars.open[i-1]) < 0 and touched
            and side*(c-previous_edge) > 0 and abs(body) >= .25*atr
            and close_quality >= .65 and span <= 2*atr):
        stop = (min(float(bars.low[i-1:i+1].min())-.1*atr, c-.6*atr) if side > 0
                else max(float(bars.high[i-1:i+1].max())+.1*atr, c+.6*atr))
        return _signal(bars, i, side, "trend_pullback", .61+alignment,
                       stop, c, bars.time[i]+60)
    return None


def live_signal(bars, atr, bias, tick, history, now):
    """Require an observed crossing and two distinct confirming fresh quotes.

    `history` belongs to this worker/profile and contains at most eight seconds.
    Starting while price is already beyond the level does not invent a crossing.
    """
    at, bid, ask = float(tick.time_msc)/1000, float(tick.bid), float(tick.ask)
    if (not np.isfinite([at, bid, ask, atr, now]).all() or atr <= 0 or bid <= 0
            or ask < bid or not -2 <= now-at <= 10):
        history.clear()
        return None
    if history and at <= history[-1][0]:
        return None
    history[:] = [row for row in history if 0 <= at-row[0] <= 8][-31:]
    history.append((at, bid, ask))
    if (len(history) < 3 or at-history[0][0] < .3
            or np.any(np.diff(bars.time[-20:]) != 60)):
        return None
    movement = bid-history[0][1]
    spread = max(ask-bid, history[0][2]-history[0][1])
    if abs(movement) < max(.15*atr, 2*spread):
        return None
    side = 1 if movement > 0 else -1
    level = float(bars.high[-5:].max() if side > 0 else bars.low[-5:].min())
    threshold = level + side*max(.06*atr, 2*bars.spec.tick_size)
    if (side*(history[0][1]-threshold) > 0 or side*(history[-2][1]-threshold) <= 0
            or side*(bid-threshold) <= 0 or side*(bid-level) > .7*atr):
        return None
    stop = min(level-.35*atr, bid-.7*atr) if side > 0 else max(level+.35*atr, bid+.7*atr)
    signal = _signal(bars, len(bars.time)-1, side, "live_momentum_breakout",
                     .62 + (.03 if bias == side else -.03 if bias == -side else 0),
                     stop, bid, min(at, now))
    signal.update(max_entry_drift_atr=.25, expires_at=min(at, now)+8)
    return signal
