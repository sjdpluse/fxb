"""Serialized, journaled fixed-volume baskets and operator-requested orders.

Only this worker constructs basket permissions. Submission remains centralized
in MT5.send_order; every child has its own durable claim and broker receipt.
"""
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import time
import uuid
import numpy as np
from .data import KABUL, day_key
from .risk import Spec, Risk, Guard, round_price
from .mt5 import Blocked, MAGIC, NewsGate, closed_bar_status
from .scalping import (ScalpConfig, analyze, readout_context, micro_readout_context,
                      learning_state, entry_score, learn_closed)
from .scalper_runtime import policy, learning_scope, scalp_cycle
from .micro_scalping import MicroConfig, quote_pressure, assess_micro_signal, weakening


def account_key(account):
    return hashlib.sha256(f"{account.server}:{account.login}".encode()).hexdigest()[:20]


def order_cost(bridge, request, spec):
    side = 1 if request["type"] == bridge.api.ORDER_TYPE_BUY else -1
    slip = math.ceil(spec.slippage_points)*spec.point
    lot, entry = request["volume"], request["price"]
    loss = bridge.api.order_calc_profit(request["type"], spec.symbol, lot,
                                      entry+side*slip, request["sl"]-side*slip)
    gain = bridge.api.order_calc_profit(request["type"], spec.symbol, lot,
                                      entry+side*slip, request["tp"]-side*slip)
    margin = bridge.api.order_calc_margin(request["type"], spec.symbol, lot, entry+side*slip)
    if any(v is None or not math.isfinite(v) for v in (loss, gain, margin)) or loss >= 0 or margin <= 0:
        raise Blocked("Broker could not calculate basket risk and margin")
    fees = 2*spec.commission_per_lot_side*lot
    return -loss+fees, gain-fees, margin


def entry_limits(account, guard, risk):
    """Explain an entry halt without clearing it or changing the chosen risk."""
    return {"equity": account.equity, "currency": account.currency,
            "day_start_equity": guard.day_start, "peak_equity": guard.peak,
            "daily_loss_floor": guard.day_start*(1-risk.daily_loss),
            "drawdown_floor": guard.peak*(1-risk.max_drawdown),
            "daily_loss_limit_percent": 100*risk.daily_loss,
            "drawdown_limit_percent": 100*risk.max_drawdown,
            "daily_halt_latched": guard.daily_halt, "drawdown_halt_latched": guard.total_halt,
            "entries_today": guard.entries, "daily_entry_limit": risk.max_trades,
            "daily_entries_remaining": max(0, risk.max_trades-guard.entries) if risk.max_trades else None,
            "risk_day": guard.day, "timezone": "Asia/Kabul"}


def budget_check(account, guard, risk, risk_cash, margin_cash, count, fraction):
    available = min(account.equity*fraction,
                    account.equity-guard.day_start*(1-risk.daily_loss),
                    account.equity-guard.peak*(1-risk.max_drawdown))
    detail = {"requested_orders": count, "total_risk_cash": risk_cash,
              "risk_budget_cash": max(0., available), "total_margin_cash": margin_cash,
              "margin_budget_cash": account.margin_free*risk.margin_fraction, "currency": account.currency,
              "limits": entry_limits(account, guard, risk)}
    reason = guard.reason(risk)
    if reason:
        raise Blocked(reason, code=reason, details=detail)
    if risk.max_trades and guard.entries+count > risk.max_trades:
        raise Blocked("Basket exceeds remaining daily entry quota", code="basket_quota", details=detail)
    if risk_cash > available+1e-8 or margin_cash > account.margin_free*risk.margin_fraction+1e-8:
        raise Blocked("Entire requested basket exceeds risk or margin budget", code="basket_budget", details=detail)
    return detail


def volume_capacity(info, side, lot, positions=(), pending=()):
    """SYMBOL_VOLUME_LIMIT covers open + pending lots in one direction.

    ACCOUNT_LIMIT_ORDERS only describes pending orders, not hedged positions.
    An undeclared (zero) volume limit is not a promise that a broker will fill.
    """
    limit = float(getattr(info, "volume_limit", 0.))
    if not math.isfinite(limit) or limit < 0:
        raise Blocked("Invalid broker directional volume limit")
    if not limit:
        return None
    direction = 0 if side > 0 else 1
    used = sum(p.volume for p in positions if p.type == direction)
    used += sum(p.volume_current for p in pending if p.type in
                ((0, 2, 4, 6) if side > 0 else (1, 3, 5, 7)))
    return max(0, math.floor((limit-used+1e-9)/lot))


def costed_target(bridge, request, spec, risk, atr):
    """Price the planned reward AFTER costs without widening the stop or lot.

    Use the broker's account-currency calculator, including for the marginal
    price increment. Repricing is bounded by the existing 2.5 ATR structural
    stop envelope times the configured reward ratio, and verified afterwards.
    """
    original_target = request["tp"]
    side = 1 if request["type"] == bridge.api.ORDER_TYPE_BUY else -1
    loss, gain, margin = order_cost(bridge, request, spec)
    required_gain = loss*risk.reward_risk
    max_distance = 2.5*atr*risk.reward_risk
    costs = {"slippage_points_per_side": spec.slippage_points,
             "slippage_price_per_side": math.ceil(spec.slippage_points)*spec.point,
             "commission_per_lot_side": spec.commission_per_lot_side,
             "round_trip_commission_cash": 2*spec.commission_per_lot_side*request["volume"]}
    details = {"original_tp": original_target, "required_net_gain_cash": required_gain,
               "loss_per_order": loss, "net_gain_per_order": gain,
               "max_target_distance": max_distance, "currency": spec.currency, "costs": costs}
    if gain+1e-8 < required_gain:
        per_tick = bridge.api.order_calc_profit(request["type"], spec.symbol, request["volume"],
                                              original_target, original_target+side*spec.tick_size)
        if per_tick is None or not math.isfinite(per_tick) or per_tick <= 0:
            raise Blocked("Broker could not price a cost-adjusted target", code="target_calculation", details=details)
        ticks = math.ceil((required_gain-gain)/per_tick)
        target = round_price(original_target+side*ticks*spec.tick_size, spec.tick_size, up=side > 0)
        details["required_tp"] = target
        if target <= 0 or side*(target-request["price"]) > max_distance+1e-8:
            raise Blocked("Configured costs require a target beyond the scalp volatility limit",
                          code="target_cost_limit", details=details)
        request = request | {"tp": target}
        loss, gain, margin = order_cost(bridge, request, spec)
    if gain+1e-7 < required_gain:
        raise Blocked("Broker-priced target does not cover the configured net reward",
                      code="target_calculation", details=details | {"net_gain_per_order": gain})
    return request, loss, margin, details | {"net_gain_per_order": gain,
        "net_reward_risk": gain/loss, "target_adjusted_for_costs": request["tp"] != original_target}


def request_plan(bridge, account, guard, spec, risk, side, atr, lot, count, fraction, zones=None):
    # Zero is internal capacity sizing, never a request to submit zero orders.
    if type(count) is not int or count < 0 or type(side) is not int or side not in (-1, 1):
        raise ValueError("Choose BUY/SELL and a nonnegative integer order count")
    if lot is not None and (type(lot) not in (int, float) or lot not in (.01, .02)):
        raise ValueError("Choose 0.01 or 0.02 lot")
    if count != 1 and getattr(account, "margin_mode", None) != 2:
        raise Blocked("Separate basket positions require an MT5 hedging account")
    tick, info = bridge.tick(spec.symbol), bridge.symbol(spec.symbol)
    spread = tick.ask-tick.bid
    # Shorter stop than one full M1 ATR, but never narrower than spread or
    # the broker's minimum closing-quote distance. Each order retains SL/TP.
    distance = max(.45*atr, 4.1*spread, info.trade_stops_level*info.point+spread+spec.tick_size)
    entry = tick.ask if side > 0 else tick.bid
    stop = zones["stop_bid"] + (spread if side < 0 else 0) if zones else entry-side*distance
    try:
        request = bridge.request(spec.symbol, side, atr, spec, risk, account,
                                 structural_stop=stop, fixed_volume=lot)
    except Blocked as error:
        error.details = {"stage": "local_risk_check", "atr": atr, "spread": spread,
                         "requested_orders": count, "requested_lot": lot, **(error.details or {})}
        if error.code == "fixed_lot_budget":
            loss = error.details["risk_cash"]
            error.details.update(total_risk_cash=loss*max(count, 1), basket_risk_budget_cash=account.equity*fraction,
                                 per_order_risk_percent=risk.fraction*100,
                                 minimum_equity_for_selected_risk=max(loss/risk.fraction, loss*max(count, 1)/fraction),
                                 minimum_equity_assumes_no_existing_risk_halt=True,
                                 total_margin_cash=error.details["margin_cash"]*max(count, 1),
                                 margin_budget_cash=account.margin_free*risk.margin_fraction)
        raise
    if request["type_filling"] not in (bridge.api.ORDER_FILLING_FOK, bridge.api.ORDER_FILLING_IOC):
        raise Blocked("Baskets require FOK or IOC; unfilled volume must not remain pending")
    sizing = "risk" if lot is None else "fixed"
    lot = request["volume"]
    if zones:
        target = zones["target_bid"] + (spread if side < 0 else 0)
        target = round_price(target, spec.tick_size, up=side < 0)
        request = request | {"tp": target}
        loss, gain, margin = order_cost(bridge, request, spec)
        target_detail = {"zones": zones, "net_reward_risk": gain/loss,
                         "net_gain_per_order": gain, "loss_per_order": loss,
                         "required_net_reward_risk": risk.reward_risk, "currency": account.currency,
                         "target_adjusted_for_costs": False}
        closing = tick.bid if side > 0 else tick.ask
        if (target <= 0 or side*(target-closing) <= info.trade_stops_level*info.point
                or gain+1e-8 < risk.reward_risk*loss):
            raise Blocked("Nearest liquidity target cannot cover the required reward after costs",
                          code="structure_reward_risk", details=target_detail)
    else:
        request, loss, margin, target_detail = costed_target(bridge, request, spec, risk, atr)
    broker_count = volume_capacity(info, side, lot)
    capacity_detail = {}
    if count == 0:
        available = min(account.equity*fraction,
            account.equity-guard.day_start*(1-risk.daily_loss),
            account.equity-guard.peak*(1-risk.max_drawdown))
        limits = {"stop_risk": max(0, math.floor((available+1e-8)/loss)),
                  "margin": max(0, math.floor((account.margin_free*risk.margin_fraction+1e-8)/margin)),
                  "broker_volume": broker_count,
                  "daily_quota": max(0, risk.max_trades-guard.entries) if risk.max_trades else None}
        count = min(n for n in limits.values() if n is not None)
        capacity_detail = {"sizing_mode": "capacity", "count_limits": limits,
                           "limiting_factors": [k for k, n in limits.items() if n == count]}
        if count < 1:
            raise Blocked(guard.reason(risk) or "No order fits the remaining basket capacity",
                          code=guard.reason(risk) or "basket_capacity",
                          details=capacity_detail | {"limits": entry_limits(account, guard, risk)})
    if broker_count is not None and count > broker_count:
        raise Blocked("Basket exceeds broker directional volume limit", code="broker_volume_limit",
                      details={"requested_orders": count, "available_orders": broker_count})
    try:
        detail = budget_check(account, guard, risk, loss*count, margin*count, count, fraction)
    except Blocked as error:
        error.details = target_detail | (error.details or {})
        raise
    return request, target_detail | detail | capacity_detail | {"protection_source": "confirmed_structure" if zones else "volatility_and_broker_limits", "lot_mode": sizing, "lot": lot, "total_lots": lot*count, "loss_per_order": loss,
                              "margin_per_order": margin, "sl": request["sl"], "tp": request["tp"]}


def child_order(memory, key):
    row = memory.db.execute("SELECT status,payload FROM orders WHERE key=?", (key,)).fetchone()
    return (row["status"], json.loads(row["payload"])) if row else (None, {})


def comment(key):
    return "fxb-"+hashlib.sha256(key.encode()).hexdigest()[:20]


def refresh_group(bridge, memory, group, positions, now):
    """Link by DEAL_POSITION_ID/comment, never by assuming order == position."""
    deals = bridge.history_deals(group["created_at"]-60, now+1)
    for child in group["children"]:
        status, sent = child_order(memory, child["key"])
        child["status"] = status
        if not status or status == "rejected":
            continue
        entries = [d for d in deals if d.symbol == group["symbol"] and d.magic == MAGIC and d.entry == 0
                   and d.time >= group["created_at"]-2 and
                   ((sent.get("order") and d.order == sent["order"])
                    or (sent.get("deal") and d.ticket == sent["deal"])
                    or d.comment == comment(child["key"]))]
        ids = {d.position_id for d in entries}
        # A fresh broker position comment can establish ownership while deal
        # history catches up. Settlement still requires complete deal history.
        ids.update(getattr(p, "identifier", p.ticket) for p in positions
                   if p.magic == MAGIC and p.symbol == group["symbol"] and
                   getattr(p, "comment", "") == comment(child["key"]))
        if len(ids) > 1:
            raise Blocked("One child order maps to multiple positions; reconcile")
        if ids:
            child["position_id"] = next(iter(ids))
        if status == "sent":
            child["result"] = sent
            if "risk_cash" not in child:
                # Recover the exact filled volume/price if the process died
                # after the durable receipt but before the basket state write.
                filled = child["request"] | {"volume": sent["volume"], "price": sent["price"]}
                child["risk_cash"], _, child["margin_cash"] = order_cost(bridge, filled, Spec(**group["spec"]))
    ids = {c.get("position_id") for c in group["children"]} - {None}
    own = [p for p in positions if getattr(p, "identifier", p.ticket) in ids
           and p.magic == MAGIC and p.symbol == group["symbol"]]
    memory.put_state("basket:"+group["id"], group)
    return own, deals


def validate_child(bridge, memory, basket_id, key, request, account, positions, pending, digest):
    """Narrow permission for one persisted child, including subsequent siblings."""
    group = memory.get_state("basket:"+basket_id)
    if (not group or group["account_key"] != account_key(account) or group["digest"] != digest
            or group["status"] != "submitting" or time.time() > group["deadline"]
            or memory.get_state("active-basket:"+account_key(account)) != basket_id):
        raise Blocked("Basket permission expired or does not match this account and model")
    child = next((c for c in group["children"] if c["key"] == key), None)
    if child is None or child["request"] != request or child_order(memory, key)[0] is not None:
        raise Blocked("Basket child is not authorized or was already submitted")
    if group["plan"]["requested_orders"] > 1 and getattr(account, "margin_mode", None) != 2:
        raise Blocked("Separate basket positions require hedging mode")
    ids = {c.get("position_id") for c in group["children"]} - {None}
    if pending or any(p.magic != MAGIC or p.symbol != group["symbol"] or
                      getattr(p, "identifier", p.ticket) not in ids for p in positions):
        raise Blocked("Exposure outside this basket prevents additional entries")
    if any(p.sl <= 0 or p.tp <= 0 or p.type != request["type"] for p in positions):
        raise Blocked("A sibling lost its protective stops or direction")
    risk, spec = Risk(**group["risk"]), Spec(**group["spec"])
    cap = volume_capacity(bridge.symbol(spec.symbol), group["side"], request["volume"], positions, pending)
    if cap is not None and cap < 1:
        raise Blocked("Broker directional volume capacity was exhausted", code="broker_volume_limit")
    open_ids = {getattr(p, "identifier", p.ticket) for p in positions}
    if any(c.get("position_id") not in open_ids for c in group["children"]
           if child_order(memory, c["key"])[0] == "sent"):
        raise Blocked("A filled sibling closed; remaining entries cancelled")
    guard = Guard(**memory.get_state("guard:"+group["account_key"], {}))
    guard.update(account.equity, day_key(time.time()), risk)
    loss, gain, margin = order_cost(bridge, request, spec)
    remaining = group["plan"]["requested_orders"] - sum(
        child_order(memory, c["key"])[0] is not None for c in group["children"])
    committed = sum(c.get("risk_cash", 0.) for c in group["children"] if child_order(memory, c["key"])[0] == "sent")
    used_margin = sum(c.get("margin_cash", group["plan"]["margin_per_order"])
                      for c in group["children"] if child_order(memory, c["key"])[0] == "sent")
    if used_margin+remaining*max(margin, group["plan"]["margin_per_order"]) > group["plan"]["margin_budget_cash"]+1e-8:
        raise Blocked("Repriced basket exceeds its original aggregate margin budget")
    if loss > account.equity*risk.fraction+1e-8 or gain+1e-8 < loss*risk.reward_risk:
        raise Blocked("Repriced child exceeds risk or net reward constraints", code="repriced_child_budget",
                      details={"loss_per_order": loss, "net_gain_per_order": gain,
                               "risk_budget_cash": account.equity*risk.fraction, "currency": account.currency})
    budget_check(account, guard, risk, committed+remaining*max(loss, group["plan"]["loss_per_order"]),
                 remaining*max(margin, group["plan"]["margin_per_order"]), remaining, group["fraction"])


def submit_basket(bridge, memory, account, meta, digest, guard, side, atr, lot, count,
                  fraction, options, deadline, vector=None, intent=None, cancelled=lambda: False,
                  zones=None, decision_trace=None):
    if cancelled():
        raise Blocked("Entry cancelled by operator")
    positions, pending = bridge.positions()
    if positions or pending:
        raise Blocked("Account already has exposure or pending orders")
    if memory.get_state("active-basket:"+account_key(account)):
        raise Blocked("Previous basket must finish reconciliation before another entry")
    if memory.unresolved():
        raise Blocked("Uncertain previous order requires reconciliation")
    spec = Spec(**meta["spec"])
    risk = Risk(**meta["risk"])
    request, plan = request_plan(bridge, account, guard, spec, risk, side, atr, lot, count, fraction, zones)
    count, lot = plan["requested_orders"], plan["lot"]
    if decision_trace is not None:
        decision_trace = decision_trace | {"plan": plan}
    # Read-only manual check calls the actual broker's order_check, without any
    # order claims. Aggregate budget is separately checked above.
    if intent and intent["check_only"]:
        check = bridge.api.order_check(request)
        details = {"retcode": getattr(check, "retcode", None), "comment": getattr(check, "comment", "No response")}
        return {"action": "CHECK", "status": "checked" if details["retcode"] == 0 else "rejected",
                "orders_sent": 0, "plan": plan, "broker": details, "broker_contacted": True,
                "reason": "manual_order_check"}
    if not options["execute"]:
        return {"action": "BUY" if side > 0 else "SELL", "status": "dry_run", "orders_sent": 0, "plan": plan}
    gid = str(uuid.uuid4())
    group = {"id": gid, "account_key": account_key(account), "scope": learning_scope(account, digest, meta),
             "digest": digest, "spec": asdict(spec), "risk": meta["risk"], "symbol": spec.symbol,
             "cfg": meta["scalp_config"], "side": side, "fraction": fraction, "created_at": time.time(),
             "deadline": deadline, "plan": plan, "status": "submitting", "manual": bool(intent),
             "micro_config": meta.get("micro_config", {}), "entry_atr": atr,
             "decision_trace": decision_trace,
             "vector": vector.tolist() if vector is not None else None,
             "children": []}
    # Both writes precede the first claim/send. A crash never resumes entries.
    memory.put_state("basket:"+gid, group)
    memory.put_state("active-basket:"+group["account_key"], gid)
    bridge.neural_status = {"phase": "submitting", "reason": "manual_entry" if intent else "neural_and_structure_entry_accepted",
                           "decision_id": decision_trace.get("id") if decision_trace else None}
    results, error = [], None
    try:
        for i in range(count):
            if cancelled() or time.time() > deadline:
                raise Blocked("Remaining basket entries cancelled or expired")
            # Journal lazily so even a large capacity cannot allocate millions
            # of child records before the opportunity expires.
            child = {"key": f"{gid}:{i}", "request": request.copy()}
            group["children"].append(child)
            positions, pending = bridge.positions()
            refresh_group(bridge, memory, group, positions, time.time())
            tick = bridge.tick(spec.symbol)
            child["request"]["price"] = tick.ask if side > 0 else tick.bid
            # Keep the original shared stops. Do not widen after a price move.
            if abs(child["request"]["price"]-request["price"]) > max(.15*atr, spec.tick_size):
                raise Blocked("Basket price moved too far; remaining children cancelled")
            memory.put_state("basket:"+gid, group)
            receipt = bridge.send_order(child["request"], child["key"], memory, account.login,
                **options, basket_id=gid, entry_deadline=deadline)
            results.append(receipt)
            if receipt["status"] == "sent":
                group.setdefault("first_fill_at", time.time())
                filled = child["request"] | {"volume": receipt["volume"], "price": receipt["price"]}
                child["risk_cash"], _, child["margin_cash"] = order_cost(bridge, filled, spec)
                guard.entries += 1
                memory.put_state("guard:"+group["account_key"], asdict(guard))
                memory.put_state("basket:"+gid, group)
            if receipt["status"] != "sent" or not math.isclose(receipt.get("volume", 0.), lot, abs_tol=1e-8):
                raise Blocked("Partial or unconfirmed fill; remaining children cancelled")
    except Blocked as exc:
        error = {"reason": str(exc), "code": exc.code, "details": exc.details}
    finally:
        group["status"] = "managing"
        group["submission_error"] = error
        memory.put_state("basket:"+gid, group)
    sent = sum(r["status"] == "sent" for r in results)
    return {"action": "BUY" if side > 0 else "SELL", "status": "sent" if sent == count else "partial" if sent else "blocked",
            "reason": "manual_entry" if intent else "micro_momentum", "basket_id": gid,
            "orders_sent": sent, "requested_orders": count, "plan": plan, "receipts": results, "error": error,
            "decision_trace": decision_trace}


def settle_group(memory, group, deals, own, now):
    if own:
        return False
    active = [c for c in group["children"] if child_order(memory, c["key"])[0] not in (None, "rejected")]
    if any(child_order(memory, c["key"])[0] in ("reserved", "ambiguous") for c in active):
        return False
    ids = {c.get("position_id") for c in active} - {None}
    if len(ids) != len(active):
        return False
    linked = [d for d in deals if d.position_id in ids and d.symbol == group["symbol"]]
    if active and (not linked or max(d.time for d in linked) > now-2):
        return False
    for pid in ids:
        ds = [d for d in linked if d.position_id == pid]
        opened = sum(d.volume for d in ds if d.entry == 0)
        closed = sum(d.volume for d in ds if d.entry in (1, 3))
        if opened <= 0 or not math.isclose(opened, closed, abs_tol=1e-8):
            return False
    external = any(d.magic != MAGIC or d.entry not in (0, 1, 3) for d in linked)
    for d in linked:
        if d.entry == 0 and not any(d.comment == comment(c["key"]) or
                d.order == child_order(memory, c["key"])[1].get("order") or
                d.ticket == child_order(memory, c["key"])[1].get("deal") for c in active):
            external = True
    net = float(sum(d.profit+d.commission+d.swap+d.fee for d in linked))
    if active and not group["manual"] and not external:
        risk_cash = sum(c.get("risk_cash", group["plan"]["loss_per_order"]) for c in active)
        learn_closed(memory, group["scope"], "basket:"+group["id"], group["symbol"], group["vector"],
                     net, risk_cash, min(d.time for d in linked), max(d.time for d in linked), now,
                     ScalpConfig(**group["cfg"]))
    group.update(status="closed", net_cashflow=net, closed_at=now, external_intervention=external)
    memory.put_state("basket:"+group["id"], group)
    memory.put_state("basket-last-exit:"+group["account_key"], now)
    memory.put_state("active-basket:"+group["account_key"], None)
    memory.record(group["scope"], group["symbol"], "basket_settled", group["created_at"], now,
                  group, "basket-settled:"+group["id"])
    return True


def manage_group(bridge, memory, group, account, guard, options, kill_file, pressure):
    positions, pending = bridge.positions()
    # Match already journaled IDs before fetching history: forced exits must
    # still run if deal history temporarily fails.
    # The final successful child might not have been observed before a crash.
    # Match its broker comment as well, without waiting for a history read.
    for child in group["children"]:
        status, sent = child_order(memory, child["key"])
        if status in (None, "rejected"):
            continue
        matches = [p for p in positions if p.magic == MAGIC and p.symbol == group["symbol"]
                   and getattr(p, "comment", "") == comment(child["key"])]
        if len(matches) == 1:
            child["position_id"] = getattr(matches[0], "identifier", matches[0].ticket)
    ids = {c.get("position_id") for c in group["children"]} - {None}
    own = [p for p in positions if getattr(p, "identifier", p.ticket) in ids
           and p.magic == MAGIC and p.symbol == group["symbol"]]
    now, risk, spec = time.time(), Risk(**group["risk"]), Spec(**group["spec"])
    if not own:
        own, deals = refresh_group(bridge, memory, group, positions, now)
        if settle_group(memory, group, deals, own, now):
            return {"action": "WAIT", "reason": "basket_settled", "basket_id": group["id"]}
    reason = group.get("close_reason")
    if own:
        if Path(kill_file).exists():
            reason = "operator_stop"
        elif guard.daily_halt or guard.total_halt:
            reason = "basket_risk_halt"
        elif any(p.sl <= 0 or p.tp <= 0 for p in own):
            reason = "basket_missing_protection"
        elif now-group.get("first_fill_at", group["created_at"]) >= risk.max_hold_bars*60:
            reason = "basket_holding_limit"
        elif pressure is not None:
            # Include already-closed siblings and charged entry fees; the open
            # positions alone cannot establish that the entire basket is ahead.
            try:
                own, deals = refresh_group(bridge, memory, group, positions, now)
                ids = {c.get("position_id") for c in group["children"]} - {None}
                linked = [d for d in deals if d.position_id in ids and d.symbol == spec.symbol]
                if any(getattr(p, "identifier", p.ticket) not in
                       {d.position_id for d in linked if d.entry == 0} for p in own):
                    raise Blocked("Waiting for entry fee history before profit-protection exit")
                total_lot = sum(p.volume for p in own)
                costs = total_lot*(spec.commission_per_lot_side +
                    math.ceil(spec.slippage_points)*spec.point/spec.tick_size*spec.tick_value)
                realized = sum(d.profit+d.commission+d.swap+d.fee for d in linked)
                net = realized+sum(p.profit+getattr(p, "swap", 0.) for p in own)-costs
                if weakening(group, pressure, net, now, total_lot*spec.tick_value):
                    reason = "favorable_quote_pressure_weakened"
                group["estimated_liquidation_net"] = net
            except Blocked as exc:
                group["pressure_exit_wait"] = str(exc)
                group.pop("weak_since", None)
                group.pop("estimated_liquidation_net", None)
        elif MicroConfig(**group.get("micro_config", {})).exit_policy == "confirmed":
            group.pop("weak_since", None)
            group.pop("weak_last_tick", None)
        if reason:
            group["close_reason"] = reason
    memory.put_state("basket:"+group["id"], group)
    results = []
    if reason:
        # Chronological order also respects hedging accounts with FIFO closing.
        for position in sorted(own, key=lambda p: (p.time, p.ticket)):
            key = f"{group['id']}:close:{position.ticket}:{position.volume}"
            status, previous = child_order(memory, key)
            if status == "rejected":
                key += f":{int(now//5)}"  # explicit no-fill only; no ambiguous retry
            if status == "sent":
                continue  # await broker snapshot/partial-volume update
            try:
                results.append(bridge.send_order(bridge.close_request(position), key, memory, account.login,
                    **(options | {"model_hash": group["digest"], "risk_config": group["risk"]})))
            except Blocked as exc:
                results.append({"status": "blocked", "reason": str(exc), "details": exc.details})
    return {"action": "CLOSE" if reason else "WAIT", "reason": reason or "managing_basket",
            "basket_id": group["id"], "open_positions": len(own), "receipts": results,
            "estimated_liquidation_net": group.get("estimated_liquidation_net"),
            "exit_diagnostics": group.get("exit_diagnostics")}


def execution_step(bridge, model_path, memory, expected_login, calendar=None, execute=False,
                   kill_file="STOP", mode="demo", expected_server=None, authorization=None,
                   entries_enabled=True, manual=None, cancelled=lambda: False):
    brain, meta, digest = policy(bridge, model_path)
    cfg, risk, spec = ScalpConfig(**meta["scalp_config"]), Risk(**meta["risk"]), Spec(**meta["spec"])
    account = bridge.account(expected_login, execution_mode=mode, expected_server=expected_server)
    ak, now = account_key(account), time.time()
    group_id = memory.get_state("active-basket:"+ak)
    if cfg.style != "micro" and not manual and not group_id:
        return scalp_cycle(bridge, model_path, memory, expected_login, calendar, execute, kill_file,
                           mode, expected_server, authorization, entries_enabled)
    if account.currency != spec.currency or meta["synthetic"]:
        raise Blocked("Basket profile must match the real account currency")
    options = dict(execute=execute, mode=mode, expected_server=expected_server, authorization=authorization,
                   model_hash=digest, risk_config=meta["risk"], entry_period_seconds=60)
    guard = Guard(**memory.get_state("guard:"+ak, {}))
    group = memory.get_state("basket:"+group_id) if group_id else None
    guard.update(account.equity, day_key(now), Risk(**group["risk"]) if group else risk)
    memory.put_state("guard:"+ak, asdict(guard))
    histories = getattr(bridge, "_micro_histories", {})
    history = histories.setdefault(spec.symbol, [])
    bridge._micro_histories = histories
    try:
        pressure = quote_pressure(history, bridge.tick(spec.symbol), now)
    except Blocked:
        history.clear()
        pressure = None
    if group:
        bridge.neural_status = {"phase": "managing_positions", "reason": "Open-basket exits take priority over new neural input"}
        result = manage_group(bridge, memory, group, account, guard, options, kill_file, pressure)
        if result.get("reason") == "basket_settled":
            bridge.neural_status = {"phase": "settled", "reason": "basket_settled"}
        group_learning = learning_state(memory, group["scope"])
        bridge.scalp_status = (getattr(bridge, "scalp_status", None) or {}) | {
            "symbol": group["symbol"], "style": "micro" if not group["manual"] else cfg.style,
            "risk": group["risk"], "risk_budget_cash": account.equity*Risk(**group["risk"]).fraction,
            "micro_config": group.get("micro_config", asdict(MicroConfig())), "entries_today": guard.entries,
            "learning": {k: v for k, v in group_learning.items() if k not in ("weights", "tick_weights")},
            "basket": {k: group[k] for k in ("id", "status", "plan", "manual")},
            "pressure": pressure, "momentum": None, "last_decision": result}
        if manual:
            raise Blocked("Existing basket is still being managed; manual entry was not sent")
        return result
    if manual and (now > manual["expires_at"] or cancelled()):
        raise Blocked("Manual intent expired or was cancelled; no order sent")
    if Path(kill_file).exists() or (not manual and (not entries_enabled or cancelled())):
        bridge.neural_status = {"phase": "paused", "reason": "entries_paused"}
        return {"action": "WAIT", "reason": "entries_paused"}
    if not bridge.clock.compatible(meta["time_basis"]["broker_clock"]):
        raise Blocked("Profile and broker clock policies differ")
    midnight = datetime.fromtimestamp(now, KABUL).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    deals = bridge.history_deals(midnight, now+1)
    guard.entries = max(guard.entries, len({d.order for d in deals if d.entry in (0, 2)}))
    memory.put_state("guard:"+ak, asdict(guard))
    # Cache only closed-bar context. Tick pressure and exits run every poll.
    cache = getattr(bridge, "_micro_context", {})
    context = cache.get(digest)
    if context is None or int(context["closed_at"]//60) != int(now//60):
        bridge.neural_status = {"phase": "waiting_for_data", "reason": "Waiting for fresh, contiguous M1 bars"}
        bars = bridge.bars(spec, 500)
        market = closed_bar_status(bars.time[-1], 60, time.time())
        if bars.synthetic or len(bars.time) != 500 or not market["ready"] or np.any(np.diff(bars.time[-20:]) != 60):
            raise Blocked("Micro entry requires 500 real M1 bars with fresh, contiguous recent history")
        raw, atr, signals, structure = analyze(bars, cfg)
        context = {"bars": bars, "raw": raw, "atr": float(atr[-1]), "structure": structure[-1],
                   "closed_at": float(bars.time[-1]+60), "encoded": None, "signal": signals[-1]}
        cache[digest] = context
        bridge._micro_context = cache
    scope = learning_scope(account, digest, meta)
    learned = learning_state(memory, scope)
    bridge.scalp_status = {"symbol": spec.symbol, "style": cfg.style, "context_bars": 500,
        "risk": meta["risk"], "entries_today": guard.entries, "structure": context["structure"],
        "risk_budget_cash": account.equity*risk.fraction, "micro_config": meta.get("micro_config"),
        "risk_peak_equity": guard.peak, "day_start_equity": guard.day_start,
        "learning": {k: v for k, v in learned.items() if k not in ("weights", "tick_weights")}, "pressure": pressure,
        "market_data": closed_bar_status(context["closed_at"]-60, 60, time.time())}
    positions, pending = bridge.positions()
    gate = (guard.reason(risk) or ("uncertain_order_requires_reconciliation" if memory.unresolved() else
            "account_has_exposure" if positions or pending else ""))
    bridge.scalp_status["entry_gate"] = gate
    bridge.scalp_status["entry_limits"] = entry_limits(account, guard, risk)
    updated_context = False
    if not manual and context["encoded"] is None:
        # Entry permission must not disable observation of valid market data.
        # Bounded batches still yield to manual requests and position management.
        bridge.neural_status = {"phase": "warming_up", "reason": "Processing closed market bars"}
        if brain.last_time is not None and brain.last_time < context["bars"].time[0]:
            brain.reset()
        context["encoded"] = brain.observe(context["raw"], context["bars"].time, max_bars=4)
        ready = context["encoded"] is not None
        bridge.neural_status = {"phase": "waiting_for_ticks" if ready else "warming_up",
                               "reason": "neural_context_updated_waiting_for_fresh_ticks" if ready else "neural_warmup",
                               **getattr(brain, "observation_progress", {})}
        # Neither warmup nor a blocked entry may bank an old tick impulse.
        history.clear()
        updated_context = True
    if gate and not manual:
        history.clear()
        bridge.neural_status = {"phase": "entry_blocked", "reason": gate,
                               **getattr(brain, "observation_progress", {})}
        result = {"action": "WAIT", "reason": gate, "details": {"limits": bridge.scalp_status["entry_limits"]}}
        bridge.scalp_status["last_decision"] = result
        return result
    micro = MicroConfig(**meta["micro_config"]) if cfg.style == "micro" else None
    decision = {"action": "WAIT", "reason": "micro_waiting_for_quote_momentum", "bar_closed_at": context["closed_at"]}
    vector, zones, trace = None, None, None
    if manual:
        bridge.neural_status = {"phase": "manual_request", "reason": "Manual execution does not require a neural signal"}
        side, count, lot = manual["side"], manual["count"], manual["lot"]
        if not micro and count != 1:
            raise Blocked("Choose a Micro profile to send multiple orders")
        fraction = micro.basket_risk if micro else risk.fraction
        deadline = manual["expires_at"]
    else:
        if updated_context:
            decision["reason"] = bridge.neural_status["reason"]
            bridge.scalp_status["last_decision"] = decision
            return decision
        bridge.neural_status = {"phase": "waiting_for_ticks", "reason": "micro_waiting_for_quote_momentum"}
        if pressure is None or time.time()-pressure["time"] > 2:
            bridge.scalp_status["last_decision"] = decision
            return decision
        cost_price = 2*(spec.commission_per_lot_side*spec.tick_size/spec.tick_value +
                        math.ceil(spec.slippage_points)*spec.point)
        signal, momentum = assess_micro_signal(pressure, context["atr"], spec.tick_size, cost_price, micro.direction_policy)
        bridge.scalp_status["momentum"] = momentum
        decision["details"] = {"momentum": momentum}
        if not signal:
            decision["reason"] = momentum["reason"]
            bridge.neural_status["reason"] = momentum["reason"]
            bridge.scalp_status["last_decision"] = decision
            return decision
        if micro.entry_policy == "structure":
            from .market_plan import structure_plan
            try:
                zones = structure_plan(context["bars"], context["structure"], signal["side"],
                    pressure["bid"], context["atr"], spec.tick_size, context.get("signal"))
            except Blocked as error:
                decision.update(reason=error.code, details={"momentum": momentum, **(error.details or {})})
                bridge.neural_status = {"phase": "waiting_for_setup", "reason": error.code}
                bridge.scalp_status["last_decision"] = decision
                return decision
            signal["structure_kind"] = zones["kind"]
        vector = (micro_readout_context(context["encoded"], signal, context["atr"])
                  if meta.get("learning_version", 1) == 2 else readout_context(context["encoded"], signal))
        score, adjustment = entry_score(learned, vector, signal["score"], cfg)
        from .scalping import decision_attribution
        trace = decision_attribution(learned, vector, signal["score"], cfg) | {
            "id": str(uuid.uuid4()), "source": "autonomous", "symbol": spec.symbol,
            "bar_closed_at": context["closed_at"], "quote_time": pressure["time"],
            "evaluated_at": time.time(), "setup": signal.get("structure_kind", signal["kind"]),
            "side": signal["side"], "zones": zones,
            "momentum": momentum,
            "neural_window_ms": getattr(brain, "latest", {}).get("window_ms"),
            "neural_spikes": getattr(brain, "latest", {}).get("spikes", 0),
            "neural_observed_at": getattr(brain, "latest", {}).get("observed_at"),
            "input_features": context["raw"][-1].tolist()}
        bridge.scalp_status["decision_trace"] = trace
        bridge.last_evaluation = trace
        decision.update(setup=signal, entry_score=score, learned_adjustment=adjustment)
        last_exit = max(learned["last_exit"], memory.get_state("basket-last-exit:"+ak, 0.))
        if time.time()-last_exit < cfg.cooldown_seconds:
            decision["reason"] = "post_basket_cooldown"
        elif learned["loss_streak"] >= cfg.max_consecutive_losses and time.time()-learned["last_exit"] < cfg.loss_pause_seconds:
            decision["reason"] = "consecutive_loss_pause"
        elif score < cfg.min_score:
            decision["reason"] = "learned_entry_filter"
        else:
            decision["reason"] = "ready"
        if decision["reason"] != "ready":
            bridge.neural_status = {"phase": "entry_filtered", "reason": decision["reason"], "decision_id": trace["id"]}
            bridge.scalp_status["last_decision"] = decision
            return decision
        side, count, lot = signal["side"], 0 if micro.sizing_mode == "capacity" else micro.orders, None if micro.lot_mode == "risk" else micro.lot
        fraction, deadline = micro.basket_risk, signal["expires_at"]
    if not (manual and manual["check_only"]):
        NewsGate(calendar).check(time.time(), risk.max_hold_bars*60)
    # A manual intent selects the side explicitly. Its volatility/broker plan
    # must reach the same risk, margin, net-RR and broker checks regardless of
    # the autonomous profile's signal policy; it is never a learning sample.
    bridge.neural_status = {"phase": "checking_order",
                           "reason": "manual_entry" if manual else "neural_and_structure_entry_accepted",
                           "decision_id": trace["id"] if trace else manual.get("request_id") if manual else None}
    result = submit_basket(bridge, memory, account, meta, digest, guard, side, context["atr"], lot, count,
                          fraction, options, deadline, vector, manual, cancelled, zones, trace)
    if not manual:
        result["details"] = (result.get("details") or {}) | {"momentum": momentum}
    bridge.neural_status = {"phase": "managing_positions" if result.get("orders_sent", 0) else
                           "entry_blocked" if result.get("status") in ("blocked", "rejected") else "waiting_for_ticks",
                           "reason": result.get("reason"), "last_submission": result.get("status"),
                           "orders_sent": result.get("orders_sent", 0), "decision_id": trace["id"] if trace else None}
    bridge.scalp_status["last_decision"] = result
    if result.get("orders_sent", 0):
        bridge.last_entry_trace = result.get("decision_trace") or {
            "source": "manual", "side": side, "evaluated_at": time.time(), "plan": result.get("plan")}
    memory.record(scope, spec.symbol, "manual_result" if manual else "micro_decision", now, time.time(), result)
    return result
