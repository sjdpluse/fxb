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
from .risk import Spec, Risk, Guard
from .mt5 import Blocked, MAGIC, NewsGate, closed_bar_status
from .scalping import (ScalpConfig, analyze, readout_context, learning_state, entry_score, learn_closed)
from .scalper_runtime import policy, learning_scope, scalp_cycle
from .micro_scalping import MicroConfig, quote_pressure, micro_signal, weakening


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


def budget_check(account, guard, risk, risk_cash, margin_cash, count, fraction):
    available = min(account.equity*fraction,
                    account.equity-guard.day_start*(1-risk.daily_loss),
                    account.equity-guard.peak*(1-risk.max_drawdown))
    detail = {"requested_orders": count, "total_risk_cash": risk_cash,
              "risk_budget_cash": max(0., available), "total_margin_cash": margin_cash,
              "margin_budget_cash": account.margin_free*risk.margin_fraction, "currency": account.currency}
    reason = guard.reason(risk)
    if reason or guard.entries+count > risk.max_trades:
        raise Blocked(reason or "Basket exceeds remaining daily entry quota", code="basket_quota", details=detail)
    if risk_cash > available+1e-8 or margin_cash > account.margin_free*risk.margin_fraction+1e-8:
        raise Blocked("Entire requested basket exceeds risk or margin budget", code="basket_budget", details=detail)
    return detail


def request_plan(bridge, account, guard, spec, risk, side, atr, lot, count, fraction):
    if type(count) is not int or not 1 <= count <= 10 or type(side) is not int or side not in (-1, 1):
        raise ValueError("Choose BUY/SELL and 1..10 orders")
    if type(lot) not in (int, float) or lot not in (.01, .02):
        raise ValueError("Choose 0.01 or 0.02 lot")
    if count > 1 and getattr(account, "margin_mode", None) != 2:
        raise Blocked("Separate basket positions require an MT5 hedging account")
    tick, info = bridge.tick(spec.symbol), bridge.symbol(spec.symbol)
    spread = tick.ask-tick.bid
    # Shorter stop than one full M1 ATR, but never narrower than spread or
    # the broker's minimum closing-quote distance. Each order retains SL/TP.
    distance = max(.45*atr, 4.1*spread, info.trade_stops_level*info.point+spread+spec.tick_size)
    entry = tick.ask if side > 0 else tick.bid
    request = bridge.request(spec.symbol, side, atr, spec, risk, account,
                             structural_stop=entry-side*distance, fixed_volume=lot)
    if request["type_filling"] not in (bridge.api.ORDER_FILLING_FOK, bridge.api.ORDER_FILLING_IOC):
        raise Blocked("Baskets require FOK or IOC; unfilled volume must not remain pending")
    loss, gain, margin = order_cost(bridge, request, spec)
    if gain < loss:
        raise Blocked("Scalp target does not cover stop risk after configured costs")
    detail = budget_check(account, guard, risk, loss*count, margin*count, count, fraction)
    return request, detail | {"lot": lot, "total_lots": lot*count, "loss_per_order": loss,
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
    if len(group["children"]) > 1 and getattr(account, "margin_mode", None) != 2:
        raise Blocked("Separate basket positions require hedging mode")
    ids = {c.get("position_id") for c in group["children"]} - {None}
    if pending or any(p.magic != MAGIC or p.symbol != group["symbol"] or
                      getattr(p, "identifier", p.ticket) not in ids for p in positions):
        raise Blocked("Exposure outside this basket prevents additional entries")
    if any(p.sl <= 0 or p.tp <= 0 or p.type != request["type"] for p in positions):
        raise Blocked("A sibling lost its protective stops or direction")
    risk, spec = Risk(**group["risk"]), Spec(**group["spec"])
    guard = Guard(**memory.get_state("guard:"+group["account_key"], {}))
    guard.update(account.equity, day_key(time.time()), risk)
    loss, gain, margin = order_cost(bridge, request, spec)
    remaining = sum(child_order(memory, c["key"])[0] is None for c in group["children"])
    committed = sum(c.get("risk_cash", 0.) for c in group["children"] if child_order(memory, c["key"])[0] == "sent")
    used_margin = sum(c.get("margin_cash", group["plan"]["margin_per_order"])
                      for c in group["children"] if child_order(memory, c["key"])[0] == "sent")
    if used_margin+remaining*max(margin, group["plan"]["margin_per_order"]) > group["plan"]["margin_budget_cash"]+1e-8:
        raise Blocked("Repriced basket exceeds its original aggregate margin budget")
    if loss > account.equity*risk.fraction+1e-8 or gain < loss:
        raise Blocked("Repriced child exceeds risk or net reward constraints")
    budget_check(account, guard, risk, committed+remaining*max(loss, group["plan"]["loss_per_order"]),
                 remaining*max(margin, group["plan"]["margin_per_order"]), remaining, group["fraction"])


def submit_basket(bridge, memory, account, meta, digest, guard, side, atr, lot, count,
                  fraction, options, deadline, vector=None, intent=None, cancelled=lambda: False):
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
    request, plan = request_plan(bridge, account, guard, spec, risk, side, atr, lot, count, fraction)
    # Read-only manual check calls the actual broker's order_check, without any
    # order claims. Aggregate budget is separately checked above.
    if intent and intent["check_only"]:
        check = bridge.api.order_check(request)
        details = {"retcode": getattr(check, "retcode", None), "comment": getattr(check, "comment", "No response")}
        return {"action": "CHECK", "status": "checked" if details["retcode"] == 0 else "rejected",
                "orders_sent": 0, "plan": plan, "broker": details, "reason": "manual_order_check"}
    if not options["execute"]:
        return {"action": "BUY" if side > 0 else "SELL", "status": "dry_run", "orders_sent": 0, "plan": plan}
    gid = str(uuid.uuid4())
    group = {"id": gid, "account_key": account_key(account), "scope": learning_scope(account, digest),
             "digest": digest, "spec": asdict(spec), "risk": meta["risk"], "symbol": spec.symbol,
             "cfg": meta["scalp_config"], "side": side, "fraction": fraction, "created_at": time.time(),
             "deadline": deadline, "plan": plan, "status": "submitting", "manual": bool(intent),
             "vector": vector.tolist() if vector is not None else None,
             "children": [{"key": f"{gid}:{i}", "request": request.copy()} for i in range(count)]}
    # Both writes precede the first claim/send. A crash never resumes entries.
    memory.put_state("basket:"+gid, group)
    memory.put_state("active-basket:"+group["account_key"], gid)
    results, error = [], None
    try:
        for child in group["children"]:
            if cancelled() or time.time() > deadline:
                raise Blocked("Remaining basket entries cancelled or expired")
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
            "orders_sent": sent, "requested_orders": count, "plan": plan, "receipts": results, "error": error}


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
        if (Path(kill_file).exists() or guard.daily_halt or guard.total_halt or
                any(p.sl <= 0 or p.tp <= 0 for p in own) or
                now-group["created_at"] >= risk.max_hold_bars*60):
            reason = "basket_time_risk_or_stop"
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
            "estimated_liquidation_net": group.get("estimated_liquidation_net")}


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
        result = manage_group(bridge, memory, group, account, guard, options, kill_file, pressure)
        bridge.scalp_status = (getattr(bridge, "scalp_status", None) or {}) | {
            "symbol": group["symbol"], "style": "micro" if not group["manual"] else cfg.style,
            "risk": group["risk"], "risk_budget_cash": account.equity*risk.fraction,
            "micro_config": meta.get("micro_config"), "entries_today": guard.entries,
            "basket": {k: group[k] for k in ("id", "status", "plan", "manual")},
            "pressure": pressure, "last_decision": result}
        if manual:
            raise Blocked("Existing basket is still being managed; manual entry was not sent")
        return result
    if manual and (now > manual["expires_at"] or cancelled()):
        raise Blocked("Manual intent expired or was cancelled; no order sent")
    if Path(kill_file).exists() or (not manual and (not entries_enabled or cancelled())):
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
        bars = bridge.bars(spec, 500)
        market = closed_bar_status(bars.time[-1], 60, time.time())
        if bars.synthetic or len(bars.time) != 500 or not market["ready"] or np.any(np.diff(bars.time[-20:]) != 60):
            raise Blocked("Micro entry requires 500 real M1 bars with fresh, contiguous recent history")
        raw, atr, signals, structure = analyze(bars, cfg)
        context = {"bars": bars, "raw": raw, "atr": float(atr[-1]), "structure": structure[-1],
                   "closed_at": float(bars.time[-1]+60), "encoded": None}
        cache[digest] = context
        bridge._micro_context = cache
    scope = learning_scope(account, digest)
    learned = learning_state(memory, scope)
    bridge.scalp_status = {"symbol": spec.symbol, "style": cfg.style, "context_bars": 500,
        "risk": meta["risk"], "entries_today": guard.entries, "structure": context["structure"],
        "risk_budget_cash": account.equity*risk.fraction, "micro_config": meta.get("micro_config"),
        "risk_peak_equity": guard.peak, "day_start_equity": guard.day_start,
        "learning": {k: v for k, v in learned.items() if k != "weights"}, "pressure": pressure,
        "market_data": closed_bar_status(context["closed_at"]-60, 60, time.time())}
    positions, pending = bridge.positions()
    gate = (guard.reason(risk) or ("uncertain_order_requires_reconciliation" if memory.unresolved() else
            "account_has_exposure" if positions or pending else ""))
    bridge.scalp_status["entry_gate"] = gate
    if gate and not manual:
        result = {"action": "WAIT", "reason": gate}
        bridge.scalp_status["last_decision"] = result
        return result
    micro = MicroConfig(**meta["micro_config"]) if cfg.style == "micro" else None
    decision = {"action": "WAIT", "reason": "micro_waiting_for_quote_momentum", "bar_closed_at": context["closed_at"]}
    vector = None
    if manual:
        side, count, lot = manual["side"], manual["count"], manual["lot"]
        if not micro and count != 1:
            raise Blocked("Choose a Micro profile to send multiple orders")
        fraction = micro.basket_risk if micro else risk.fraction
        deadline = manual["expires_at"]
    else:
        if context["encoded"] is None:
            if brain.last_time is not None and brain.last_time < context["bars"].time[0]:
                brain.reset()
            context["encoded"] = brain.observe(context["raw"], context["bars"].time)
            # Initial graph warmup must never turn an old impulse into an entry.
            history.clear()
            decision["reason"] = "neural_context_updated_waiting_for_fresh_ticks"
            bridge.scalp_status["last_decision"] = decision
            return decision
        if pressure is None or time.time()-pressure["time"] > 2:
            bridge.scalp_status["last_decision"] = decision
            return decision
        cost_price = 2*(spec.commission_per_lot_side*spec.tick_size/spec.tick_value +
                        math.ceil(spec.slippage_points)*spec.point)
        signal = micro_signal(pressure, context["atr"], spec.tick_size, cost_price)
        if not signal:
            bridge.scalp_status["last_decision"] = decision
            return decision
        vector = readout_context(context["encoded"], signal)
        score, adjustment = entry_score(learned, vector, signal["score"], cfg)
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
            bridge.scalp_status["last_decision"] = decision
            return decision
        side, count, lot = signal["side"], micro.orders, micro.lot
        fraction, deadline = micro.basket_risk, signal["expires_at"]
    if not (manual and manual["check_only"]):
        NewsGate(calendar).check(time.time(), risk.max_hold_bars*60)
    result = submit_basket(bridge, memory, account, meta, digest, guard, side, context["atr"], lot, count,
                          fraction, options, deadline, vector, manual, cancelled)
    bridge.scalp_status["last_decision"] = result
    memory.record(scope, spec.symbol, "manual_result" if manual else "micro_decision", now, time.time(), result)
    return result
