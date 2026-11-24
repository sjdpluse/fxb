"""Causal price zones for Micro. A setup is a hypothesis, not a win probability.

Use bid bars, with two closed bars confirming each pivot. Targets stop BEFORE
the nearest observed opposing liquidity; costs cannot push a target beyond it.
"""
import numpy as np
from .mt5 import Blocked


def pivots(bars, lookback=120):
    highs, lows = [], []
    for i in range(max(2, len(bars.time)-lookback), len(bars.time)-2):
        w = slice(i-2, i+3)
        if not np.all(np.diff(bars.time[w]) == 60):
            continue
        if bars.high[i] == max(bars.high[w]) and np.sum(bars.high[w] == bars.high[i]) == 1:
            highs.append(float(bars.high[i]))
        if bars.low[i] == min(bars.low[w]) and np.sum(bars.low[w] == bars.low[i]) == 1:
            lows.append(float(bars.low[i]))
    return highs, lows


def structure_plan(bars, structure, side, bid, atr, tick_size, signal=None, manual=False):
    """Return immutable bid zones, or a reason to wait. Never invent liquidity."""
    highs, lows = pivots(bars)
    buffer = max(.10*atr, 2*tick_size)
    c, o = float(bars.close[-1]), float(bars.open[-1])
    anchor, kind = None, None
    bias = structure.get("m5_bias", 0)
    if manual:
        anchors = [x for x in (lows if side > 0 else highs) if side*(bid-x) > 0]
        if anchors:
            anchor = max(anchors) if side > 0 else min(anchors)
            kind = "manual_structural_levels"
    elif signal and signal["side"] == side:
        # The ICT adapter already buffers its swept/retest extreme.
        anchor = signal["stop_bid"] + side*buffer
        kind = signal["kind"]
    elif structure.get("liquidity_sweep") == side and side*(c-o) > 0 and abs(bid-c) <= .5*atr:
        anchor = float(bars.low[-1] if side > 0 else bars.high[-1])
        kind = "confirmed_liquidity_sweep"
    elif bias in (0, side):
        level = float(max(bars.high[-8:-2]) if side > 0 else min(bars.low[-8:-2]))
        recent = bars.close[-2:]
        touched = float(bars.low[-1] if side > 0 else bars.high[-1])
        if (max(side*(recent-level)) > 0 and side*(c-level) > 0
                and side*(touched-level) <= .15*atr and -.10*atr <= side*(bid-level) <= .35*atr):
            anchor = float(min(bars.low[-2:]) if side > 0 else max(bars.high[-2:]))
            kind = "micro_breakout_retest"
        elif bias == side and structure.get("swing_trend", 0) in (0, side):
            ema = float(bars.close[-60])
            for close in bars.close[-59:]:
                ema += (2/21)*(float(close)-ema)
            touched = float(min(bars.low[-3:]) if side > 0 else max(bars.high[-3:]))
            if (side*(c-o) > .10*atr and side*(c-ema) >= 0
                    and side*(touched-ema) <= .15*atr and abs(bid-c) <= .35*atr):
                anchor = touched
                kind = "micro_trend_pullback"
    if anchor is None:
        raise Blocked("Tick impulse has no confirmed structure setup", code="no_structure_setup")
    stop = anchor-side*buffer
    distance = side*(bid-stop)
    if not 0 < distance <= 2.5*atr:
        raise Blocked("Structural invalidation is already crossed or too far away", code="structure_stop_distance",
            details={"stage": "structure_validation", "side": side, "bid": float(bid),
                     "invalidation_bid": float(anchor), "stop_bid": float(stop), "atr": float(atr),
                     "distance_price": float(distance), "max_distance_price": float(2.5*atr),
                     "distance_atr": float(distance/atr) if atr > 0 else None,
                     "failure": "crossed" if distance <= 0 else "too_far", "setup": kind})
    # Confirmed pivots AND the observed range bound are known before entry.
    targets = highs+[float(max(bars.high[-120:]))] if side > 0 else lows+[float(min(bars.low[-120:]))]
    targets = [x for x in targets if side*(x-bid) > 0]
    if not targets:
        raise Blocked("No observed opposing liquidity target ahead of this entry", code="no_structure_target")
    liquidity = min(targets) if side > 0 else max(targets)
    return {"kind": kind, "stop_bid": stop, "target_bid": liquidity-side*buffer,
            "invalidation_bid": anchor, "liquidity_bid": liquidity, "buffer_price": buffer,
            "bar_closed_at": float(bars.time[-1]+60), "m5_bias": bias,
            "source": "closed_M1_confirmed_pivots_and_retest"}
