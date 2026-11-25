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


def setup_assessment(bars, structure, side, bid, atr, buffer, signal=None):
    """Evaluate the existing alternatives in order and retain their measurements.

    A passing setup is not permission to send: zones, costs and account gates
    are evaluated afterwards. Later alternatives are not evaluated after a match.
    """
    kinds = ("ict_retest", "confirmed_liquidity_sweep", "micro_breakout_retest", "micro_trend_pullback")
    candidates = [{"kind": k, "status": "not_evaluated", "checks": []} for k in kinds]
    audit = {"matched": False, "selected_kind": None, "side": side, "bid": float(bid),
             "atr": float(atr), "bar_closed_at": float(bars.time[-1]+60), "candidates": candidates}

    def check(name, value, operator, threshold):
        if isinstance(value, np.generic):
            value = value.item()
        passed = {"eq": lambda: value == threshold, "gt": lambda: value > threshold,
                  "ge": lambda: value >= threshold, "le": lambda: value <= threshold,
                  "in": lambda: value in threshold,
                  "between": lambda: threshold[0] <= value <= threshold[1]}[operator]()
        return {"name": name, "value": value, "operator": operator,
                "threshold": threshold, "passed": bool(passed)}

    def candidate(index, checks, **levels):
        row = candidates[index]
        row.update(checks=checks, levels=levels,
                   status="pass" if all(c["passed"] for c in checks) else "wait")
        return row["status"] == "pass"

    def selected(anchor, kind):
        audit.update(matched=True, selected_kind=kind)
        return float(anchor), kind, audit

    c, o = float(bars.close[-1]), float(bars.open[-1])
    bias, trend = structure.get("m5_bias", 0), structure.get("swing_trend", 0)
    if candidate(0, [check("signal_present", bool(signal), "eq", True),
                     check("signal_side", signal["side"] if signal else 0, "eq", side)]):
        # The ICT adapter already buffers its swept/retest extreme.
        return selected(signal["stop_bid"] + side*buffer, signal["kind"])
    if candidate(1, [check("sweep_side", structure.get("liquidity_sweep", 0), "eq", side),
                     check("candle_body", side*(c-o), "gt", 0.),
                     check("distance_from_close", abs(bid-c), "le", .5*atr)], close_price=c):
        return selected(bars.low[-1] if side > 0 else bars.high[-1], "confirmed_liquidity_sweep")
    level = float(max(bars.high[-8:-2]) if side > 0 else min(bars.low[-8:-2]))
    recent = bars.close[-2:]
    touched = float(bars.low[-1] if side > 0 else bars.high[-1])
    if candidate(2, [check("m5_bias", bias, "in", [0, side]),
                     check("recent_break", max(side*(recent-level)), "gt", 0.),
                     check("close_beyond_level", side*(c-level), "gt", 0.),
                     check("retest_distance", side*(touched-level), "le", .15*atr),
                     check("entry_distance", side*(bid-level), "between", [-.10*atr, .35*atr])],
                 level_price=level, close_price=c, touched_price=touched):
        return selected(min(bars.low[-2:]) if side > 0 else max(bars.high[-2:]), "micro_breakout_retest")
    ema = float(bars.close[-60])
    for close in bars.close[-59:]:
        ema += (2/21)*(float(close)-ema)
    touched = float(min(bars.low[-3:]) if side > 0 else max(bars.high[-3:]))
    if candidate(3, [check("m5_bias", bias, "eq", side),
                     check("swing_trend", trend, "in", [0, side]),
                     check("candle_body", side*(c-o), "gt", .10*atr),
                     check("close_vs_ema", side*(c-ema), "ge", 0.),
                     check("pullback_distance", side*(touched-ema), "le", .15*atr),
                     check("distance_from_close", abs(bid-c), "le", .35*atr)],
                 ema_price=ema, close_price=c, touched_price=touched):
        return selected(touched, "micro_trend_pullback")
    return None, None, audit


def structure_plan(bars, structure, side, bid, atr, tick_size, signal=None, manual=False):
    """Return immutable bid zones, or a reason to wait. Never invent liquidity."""
    highs, lows = pivots(bars)
    buffer = max(.10*atr, 2*tick_size)
    anchor, kind = None, None
    audit = None
    bias = structure.get("m5_bias", 0)
    if manual:
        anchors = [x for x in (lows if side > 0 else highs) if side*(bid-x) > 0]
        if anchors:
            anchor = max(anchors) if side > 0 else min(anchors)
            kind = "manual_structural_levels"
    else:
        anchor, kind, audit = setup_assessment(bars, structure, side, bid, atr, buffer, signal)
    if anchor is None:
        raise Blocked("Tick impulse has no confirmed structure setup", code="no_structure_setup",
                      details={"stage": "structure_setup", "setup_assessment": audit})
    stop = anchor-side*buffer
    distance = side*(bid-stop)
    if not 0 < distance <= 2.5*atr:
        raise Blocked("Structural invalidation is already crossed or too far away", code="structure_stop_distance",
            details={"stage": "structure_validation", "side": side, "bid": float(bid),
                     "invalidation_bid": float(anchor), "stop_bid": float(stop), "atr": float(atr),
                     "distance_price": float(distance), "max_distance_price": float(2.5*atr),
                     "distance_atr": float(distance/atr) if atr > 0 else None,
                     "failure": "crossed" if distance <= 0 else "too_far", "setup": kind,
                     "setup_assessment": audit})
    # Confirmed pivots AND the observed range bound are known before entry.
    targets = highs+[float(max(bars.high[-120:]))] if side > 0 else lows+[float(min(bars.low[-120:]))]
    targets = [x for x in targets if side*(x-bid) > 0]
    if not targets:
        raise Blocked("No observed opposing liquidity target ahead of this entry", code="no_structure_target",
                      details={"stage": "structure_target", "setup_assessment": audit})
    liquidity = min(targets) if side > 0 else max(targets)
    return {"kind": kind, "stop_bid": stop, "target_bid": liquidity-side*buffer,
            "invalidation_bid": anchor, "liquidity_bid": liquidity, "buffer_price": buffer,
            "bar_closed_at": float(bars.time[-1]+60), "m5_bias": bias,
            "source": "closed_M1_confirmed_pivots_and_retest", "setup_assessment": audit}
