"""M1 gold deployment, owned-position management and settled-deal learning."""
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
import re
import time
import numpy as np
from .connectome import sha256
from .data import KABUL, day_key, save_bars
from .risk import Spec, Risk, Guard
from .mt5 import Blocked, MAGIC, NewsGate, closed_bar_status
from .scalping import (VERSION, ACTIVE_VERSION, FEATURES, ScalpConfig, scalp_risk, analyze,
                       readout_context, entry_score, learning_state, learn_closed)
from .micro_scalping import MicroConfig, VERSION as MICRO_VERSION


def strategy_version(style):
    return {"structure": VERSION, "active": ACTIVE_VERSION, "micro": MICRO_VERSION}[style]


def create_profile(bridge, symbol, output, commission, slippage, overnight, config=None, risk=None, micro=None):
    """Read 500 closed M1 bars; create an untrained adaptive profile, not a DQN."""
    from .fast_brain import FastWholeBrain
    cfg = config or ScalpConfig()
    risk = risk or scalp_risk(cfg.style)
    if risk != scalp_risk(cfg.style, risk.fraction, risk.max_trades, risk.daily_loss, risk.max_hold_bars):
        raise ValueError("Unsupported scalper risk configuration")
    if not symbol or any(c in symbol for c in "/\\\0") or "XAU" not in symbol.upper():
        raise ValueError("Choose the broker's exact XAU gold symbol")
    spec = bridge.spec(symbol, commission, slippage, overnight, timeframe_seconds=60)
    bars = bridge.bars(spec, cfg.context_bars)
    if bars.synthetic or len(bars.time) != cfg.context_bars:
        raise ValueError("The scalper needs 500 actual closed M1 bars")
    market = closed_bar_status(bars.time[-1], 60, time.time())
    if not market["ready"]:
        raise Blocked(f"{market['reason']}; close={market['bar_close_utc']}, age={market['age_seconds']:.3f}s")
    raw, atr, signals, structures = analyze(bars, cfg)
    brain = FastWholeBrain()  # manifest + mapping only; graph remains lazy
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_bars(bars, output / "context.csv")
    meta = {"strategy": strategy_version(cfg.style),
            "spec": asdict(spec), "risk": asdict(risk),
            "scalp_config": asdict(cfg), "features": FEATURES, "encoder_type": "scalp-full-brain",
            "topology": brain.provenance, "time_basis": bars.time_basis, "synthetic": False,
            "data_source": bars.source, "run_id": output.name, "created_at": time.time(),
            "context_sha256": bars.digest(), "initial_policy": "explicit_rules_with_zero_online_samples",
            "learning": "bounded_reward_readout_after_fully_closed_broker_positions"}
    if cfg.style == "micro":
        micro = micro or MicroConfig(direction_policy="price_path")
        if micro.basket_risk > risk.daily_loss:
            raise ValueError("Basket risk cannot exceed the daily loss limit")
        if micro.exit_policy == "confirmed" and micro.min_hold_seconds >= risk.max_hold_bars*60:
            raise ValueError("Minimum protection hold must be shorter than maximum holding time")
        meta["micro_config"] = asdict(micro)
        meta["learning"] = "one_reward_per_fully_closed_autonomous_basket"
        meta["learning_version"] = 2
    with (output / "model.npz").open("wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta, allow_nan=False)))
    return {"model": str(output / "model.npz"), "symbol": symbol, "timeframe": "M1",
            "context_bars": len(raw), "initial_online_samples": 0, "orders_sent": 0,
            "structure": structures[-1], "market_data": market, "profitability_established": False}


def policy(bridge, path):
    from .fast_brain import FastWholeBrain
    path = Path(path).resolve()
    stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    cache = getattr(bridge, "_scalp_policies", {})
    if stamp not in cache:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
        cfg = ScalpConfig(**meta["scalp_config"])
        risk = Risk(**meta["risk"])
        if (meta.get("strategy") != strategy_version(cfg.style)
                or meta.get("features") != FEATURES
                or meta["spec"]["timeframe_seconds"] != 60 or "XAU" not in meta["spec"]["symbol"].upper()
                or risk != scalp_risk(cfg.style, risk.fraction, risk.max_trades, risk.daily_loss, risk.max_hold_bars)):
            raise Blocked("Unsupported or changed M1 scalper profile")
        if cfg.style == "micro":
            micro = MicroConfig(**meta["micro_config"])
            if micro.basket_risk > risk.daily_loss:
                raise Blocked("Basket risk exceeds daily limit")
            if micro.exit_policy == "confirmed" and micro.min_hold_seconds >= risk.max_hold_bars*60:
                raise Blocked("Invalid Micro holding interval")
            validate_learning_key(meta)
            if meta.get("learning_version", 1) not in (1, 2):
                raise Blocked("Unknown Micro learning context version")
        brain = FastWholeBrain()
        if brain.provenance != meta["topology"]:
            raise Blocked("The scalper's connectome provenance changed; create a new profile")
        cache = {k: v for k, v in cache.items() if k[0] != str(path)}
        cache[stamp] = (brain, meta, sha256(path))
        bridge._scalp_policies = cache
    return cache[stamp]


def validate_learning_key(meta):
    key = meta.get("learning_key")
    if key is not None and (not isinstance(key, str) or re.fullmatch("[0-9a-f]{64}", key) is None):
        raise ValueError("Invalid learning lineage identifier")


def learning_scope(account, model_hash, meta=None):
    if meta is not None:
        validate_learning_key(meta)
        model_hash = meta.get("learning_key", model_hash)
    return hashlib.sha256(f"{account.server}:{account.login}:{account.trade_mode}:{model_hash}".encode()).hexdigest()


def upgrade_micro_profile(source, output):
    """Create a separately activatable profile while retaining account-scoped learning.

    No guard, historical record, original model or current basket is rewritten.
    A live authorization remains bound to the old file hash until reactivated.
    """
    source, output = Path(source), Path(output)
    with np.load(source, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
    cfg, risk, spec = ScalpConfig(**meta["scalp_config"]), Risk(**meta["risk"]), Spec(**meta["spec"])
    if (cfg.style != "micro" or meta.get("strategy") != MICRO_VERSION or meta.get("synthetic") is not False
            or meta.get("features") != FEATURES or spec.timeframe_seconds != 60 or "XAU" not in spec.symbol.upper()
            or risk != scalp_risk(cfg.style, risk.fraction, risk.max_trades, risk.daily_loss, risk.max_hold_bars)):
        raise ValueError("Choose an existing real-market Micro profile")
    validate_learning_key(meta)
    micro = MicroConfig(**meta["micro_config"])
    if micro.basket_risk > risk.daily_loss:
        raise ValueError("Basket risk exceeds daily limit")
    parent = sha256(source)
    new_risk = replace(risk, max_trades=0, max_hold_bars=5)
    new_micro = replace(micro, sizing_mode="capacity", exit_policy="confirmed", entry_policy="structure", lot_mode="risk",
                        direction_policy="price_path")
    meta.update(risk=asdict(new_risk), micro_config=asdict(new_micro), learning_version=2,
        learning_key=meta.get("learning_key", parent), parent_model_sha256=parent,
        run_id=output.name, created_at=time.time(),
        learning="one_reward_per_fully_closed_autonomous_basket_with_entry_tick_context")
    output.mkdir(parents=True, exist_ok=False)
    with (output/"model.npz").open("wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta, allow_nan=False)))
    return {"status": "upgraded", "model": str(output/"model.npz"), "parent_model_sha256": parent,
            "learning_key": meta["learning_key"], "learning_preserved": True,
            "risk": meta["risk"], "micro_config": meta["micro_config"],
            "orders_sent": 0, "requires_activation": True}


def decision_stats(memory, scope, decision, now):
    """Count distinct bars/reasons, not two-second polls; keep today's totals on disk."""
    key = "scalp-stats:" + scope + ":" + day_key(now)
    stats = memory.get_state(key, {"bars": 0, "signal_bars": 0, "reasons": {}, "last_bar": None})
    before = json.dumps(stats, sort_keys=True)
    bar = decision["bar_closed_at"]
    if stats["last_bar"] != bar:
        stats.update(last_bar=bar, bars=stats["bars"]+1, seen_reasons=[], signal_seen=False)
    if decision.get("setup") and not stats["signal_seen"]:
        stats["signal_bars"] += 1
        stats["signal_seen"] = True
    reason = decision.get("reason_code", decision["reason"])
    if decision["action"] == "WAIT" and reason not in stats["seen_reasons"]:
        stats["seen_reasons"].append(reason)
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
    if json.dumps(stats, sort_keys=True) != before:
        memory.put_state(key, stats)
    return {k: stats[k] for k in ("bars", "signal_bars", "reasons")}


def settle_attempts(memory, scope, symbol, deals, positions, now, config):
    """Link entry orders to position IDs. No floating, partial or foreign rewards."""
    rows = memory.db.execute("SELECT key,payload FROM state WHERE key LIKE ?",
                             ("scalp-attempt:" + scope + ":%",)).fetchall()
    open_ids = {getattr(p, "identifier", p.ticket) for p in positions}
    learned = 0
    for row in rows:
        attempt = json.loads(row["payload"])
        if attempt.get("settled"):
            continue
        order = memory.db.execute("SELECT status,payload FROM orders WHERE key=?", (attempt["order_key"],)).fetchone()
        if order is None or order["status"] == "rejected":
            # Without a durable order claim, send_order() could never submit.
            memory.put_state(row["key"], attempt | {"settled": "no_confirmed_submission"})
            continue
        sent = json.loads(order["payload"])
        comment = "fxb-" + hashlib.sha256(attempt["order_key"].encode()).hexdigest()[:20]
        def matches(d):
            return ((sent.get("order") and d.order == sent["order"]) or
                    (sent.get("deal") and d.ticket == sent["deal"]) or d.comment == comment)
        entries = [d for d in deals if d.symbol == symbol and d.magic == MAGIC and d.entry == 0
                   and d.time >= attempt["created_at"]-2 and matches(d)]
        ids = {d.position_id for d in entries}
        if len(ids) != 1 or not next(iter(ids), 0):
            continue
        pid = next(iter(ids))
        if pid in open_ids:
            continue
        linked = [d for d in deals if d.position_id == pid and d.symbol == symbol]
        if not linked or max(d.time for d in linked) > now - 2:
            continue
        # Reversals, other strategies adding volume, and manual intervention
        # are not examples of this policy's complete autonomous trade.
        if any(d.entry not in (0, 1, 3) or d.magic != MAGIC or (d.entry == 0 and not matches(d)) for d in linked):
            memory.put_state(row["key"], attempt | {"settled": "excluded_external_intervention"})
            continue
        opened = sum(d.volume for d in linked if d.entry == 0)
        closed = sum(d.volume for d in linked if d.entry in (1, 3))
        if opened <= 0 or not math.isclose(opened, closed, rel_tol=1e-7, abs_tol=1e-8):
            continue
        net = float(sum(d.profit + d.commission + d.swap + d.fee for d in linked))
        exit_time = max(d.time for d in linked)
        applied = learn_closed(memory, scope, pid, symbol, attempt["state"], net,
            attempt["risk_per_lot"] * opened, min(d.time for d in entries), exit_time, now, config)
        # A crash between these writes replays learn_closed(), which is idempotent.
        memory.put_state(row["key"], attempt | {"settled": True, "position_id": pid})
        learned += int(applied)
    return learned


def scalp_cycle(bridge, model_path, memory, expected_login, calendar=None, execute=False,
                kill_file="STOP", mode="demo", expected_server=None, authorization=None,
                entries_enabled=True):
    brain, meta, digest = policy(bridge, model_path)
    spec, risk, cfg = Spec(**meta["spec"]), Risk(**meta["risk"]), ScalpConfig(**meta["scalp_config"])
    now = time.time()
    account = bridge.account(expected_login, execution_mode=mode, expected_server=expected_server)
    if spec.currency != account.currency or meta["synthetic"]:
        raise Blocked("Scalper profile does not match the real broker contract currency")
    account_key = hashlib.sha256(f"{account.server}:{account.login}".encode()).hexdigest()[:20]
    scope = learning_scope(account, digest)
    send_options = dict(execute=execute, mode=mode, expected_server=expected_server,
                        authorization=authorization, model_hash=digest, risk_config=meta["risk"],
                        entry_period_seconds=60)
    guard_key = "guard:" + account_key
    guard = Guard(**memory.get_state(guard_key, {}))
    guard.update(account.equity, day_key(now), risk)
    memory.put_state(guard_key, asdict(guard))
    positions, pending = bridge.positions()
    own = [p for p in positions if p.magic == MAGIC and p.symbol == spec.symbol]
    if len(own) > 1:
        raise Blocked("Multiple owned positions require reconciliation")
    p = own[0] if own else None
    stopped = Path(kill_file).exists()
    # Risk/time exits run on every poll, before history, model warmup or news checks.
    if p and (stopped or guard.daily_halt or guard.total_halt or p.sl <= 0 or p.tp <= 0
              or now - p.time >= risk.max_hold_bars * 60):
        exit_key = f"{account_key}:{p.ticket}:scalp-risk-close:{int(now//60)}"
        result = bridge.send_order(bridge.close_request(p),
            exit_key, memory, expected_login, **send_options)
        decision = {"action": "CLOSE", "reason": "M1_time_risk_or_stop", **result}
        memory.record(scope, spec.symbol, "scalp_exit", now, max(now, time.time()), decision,
                      hashlib.sha256(f"{exit_key}:{execute}".encode()).hexdigest())
        return decision
    if stopped:
        return {"action": "WAIT", "reason": "stop_file"}
    source_clock = (meta.get("time_basis") or {}).get("broker_clock")
    if source_clock is None or not bridge.clock.compatible(source_clock):
        raise Blocked("M1 profile and broker clock policies differ; prepare a fresh profile")
    midnight = datetime.fromtimestamp(now, KABUL).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    # Include each unconsumed attempt, even following a long restart. A clock
    # policy must cover its true entry date before it can become a reward.
    attempts = memory.db.execute("SELECT payload FROM state WHERE key LIKE ?", ("scalp-attempt:"+scope+":%",)).fetchall()
    earliest = min([midnight] + [json.loads(r[0])["created_at"]-60 for r in attempts if not json.loads(r[0]).get("settled")])
    deals = bridge.history_deals(earliest, now + 1)
    guard.entries = max(guard.entries, len({d.order for d in deals if d.time >= midnight and d.entry in (0, 2)}))
    memory.put_state(guard_key, asdict(guard))
    settle_attempts(memory, scope, spec.symbol, deals, positions, now, cfg)
    learned = learning_state(memory, scope)
    public_learning = {k: v for k, v in learned.items() if k != "weights"}
    entry_gate = ("entries_paused" if not entries_enabled else "account_has_exposure" if positions or pending
                  else "uncertain_order_requires_reconciliation" if memory.unresolved() else guard.reason(risk))
    bridge.scalp_status = (getattr(bridge, "scalp_status", None) or {}) | {
                           "symbol": spec.symbol, "timeframe": "M1", "context_bars": 500,
                           "learning": public_learning, "risk": meta["risk"], "execution": execute,
                           "style": cfg.style, "entries_today": guard.entries,
                           "risk_budget_cash": account.equity*risk.fraction,
                           "entry_gate": entry_gate, "risk_peak_equity": guard.peak,
                           "day_start_equity": guard.day_start}
    bars = bridge.bars(spec, cfg.context_bars)
    if bars.synthetic or len(bars.time) != cfg.context_bars:
        raise Blocked("500 real closed M1 bars are required")
    closed_at = float(bars.time[-1] + 60)
    # History/API work may cross a minute boundary. The start-of-cycle clock
    # must not label a newly closed bar as future data or accept an old bar.
    market = closed_bar_status(bars.time[-1], 60, time.time())
    bridge.scalp_status["market_data"] = market
    if not market["ready"]:
        bridge.scalp_status["last_decision"] = {"action": "WAIT", "reason": market["status"]}
        raise Blocked(market["reason"])
    key = f"{account_key}:{digest}:M1:{int(closed_at)}"
    decision_key = "decision:" + key + (":execute" if execute else ":observe")
    previous_decision = memory.get_state(decision_key)
    if previous_decision and (cfg.style != "active" or previous_decision.get("action") in ("BUY", "SELL", "CLOSE")):
        return {"action": "WAIT", "reason": "already_processed_bar"}
    raw, atr, signals, structure = analyze(bars, cfg)
    encoded = None
    if not p:
        if brain.last_time is not None and brain.last_time < bars.time[0]:
            brain.reset()
        encoded = brain.observe(raw, bars.time)
    decision = {"action": "WAIT", "reason": entry_gate or "no_confirmed_setup", "bar_closed_at": closed_at,
                "strategy": meta["strategy"], "structure": structure[-1], "learning": public_learning,
                "features": dict(zip(FEATURES, raw[-1].tolist()))}
    signal = signals[-1]
    now = time.time()  # graph initialization is not allowed to make an entry late
    if cfg.style == "active" and not p:
        from .active_scalping import live_signal
        histories = getattr(bridge, "_scalp_tick_history", {})
        history = histories.setdefault(digest, [])
        bridge._scalp_tick_history = histories
        try:
            fresh = bridge.tick(spec.symbol)
            live = live_signal(bars, atr[-1], structure[-1].get("m5_bias", 0), fresh, history, time.time())
            if live and (signal is None or now > closed_at+cfg.entry_window_seconds):
                signal = live
        except Blocked as error:
            history.clear()
            if signal is None and not entry_gate:
                decision.update(reason=str(error), status="blocked")
    if p:
        side = 1 if p.type == bridge.api.POSITION_TYPE_BUY else -1
        if structure[-1]["structure_break"] == -side:
            decision.update(action="CLOSE", reason="opposite_M1_structure_break")
            memory.put_state(decision_key, decision)
            decision.update(bridge.send_order(bridge.close_request(p), key, memory, expected_login, **send_options))
            memory.record(scope, spec.symbol, "scalp_exit", closed_at, max(closed_at, time.time()), decision,
                          hashlib.sha256(decision_key.encode()).hexdigest())
            bridge.scalp_status.update(structure=structure[-1], last_decision=decision)
            return decision
        decision["reason"] = "managing_broker_SL_TP"
    elif signal:
        vector = readout_context(encoded, signal)
        score, adjustment = entry_score(learned, vector, signal["score"], cfg)
        decision.update(setup=signal, entry_score=score, learned_adjustment=adjustment,
                        score_is_probability=False)
        reason = ""
        deadline = min(closed_at+60, signal["expires_at"]) if "expires_at" in signal else closed_at+cfg.entry_window_seconds
        last_exit = max([learned["last_exit"]] + [d.time for d in deals if d.symbol == spec.symbol
                         and d.magic == MAGIC and d.entry in (1, 3)])
        if not entries_enabled: reason = "entries_paused"
        elif positions or pending: reason = "account_has_exposure"
        elif memory.unresolved(): reason = "uncertain_order_requires_reconciliation"
        elif guard.reason(risk): reason = guard.reason(risk)
        elif now > deadline: reason = "entry_window_expired"
        elif np.any(np.diff(bars.time[-20:]) != 60): reason = "recent_M1_history_gap"
        elif now-last_exit < cfg.cooldown_seconds: reason = "post_trade_cooldown"
        elif learned["loss_streak"] >= cfg.max_consecutive_losses and now-learned["last_exit"] < cfg.loss_pause_seconds:
            reason = "consecutive_loss_pause"
        elif memory.get_state("scalp-used:"+scope+":"+signal["setup_id"]): reason = "setup_already_used"
        elif score < cfg.min_score: reason = "learned_entry_filter"
        if not reason:
            try:
                NewsGate(calendar).check(now, risk.max_hold_bars * 60)
                tick = bridge.tick(spec.symbol)
                # Stops come from bid-chart structure; a sell exits at ask.
                stop = signal["stop_bid"] + (tick.ask-tick.bid if signal["side"] < 0 else 0)
                request = bridge.request(spec.symbol, signal["side"], atr[-1], spec, risk, account,
                                         structural_stop=stop)
                entry = request["price"]
                if "entry_reference" in signal:
                    if abs(entry-signal["entry_reference"]) > signal["max_entry_drift_atr"]*atr[-1]:
                        raise Blocked("Price moved too far from the scalp signal")
                else:
                    edge = signal["gap_high"] if signal["side"] > 0 else signal["gap_low"]
                    if abs(entry-edge) > .6*atr[-1]:
                        raise Blocked("Price moved too far from the FVG retest")
                ref = request["volume"]
                loss = bridge.api.order_calc_profit(request["type"], spec.symbol, ref, entry, request["sl"])
                gain = bridge.api.order_calc_profit(request["type"], spec.symbol, ref, entry, request["tp"])
                # Both sides' commission and adverse slippage enter the stored R budget.
                costs = 2 * (spec.commission_per_lot_side + math.ceil(spec.slippage_points)*spec.point/spec.tick_size*spec.tick_value)
                if loss is None or gain is None or not np.isfinite([loss, gain]).all() or loss >= 0:
                    raise Blocked("Cannot verify structural risk with the broker")
                risk_per_lot = -loss/ref + costs
                if gain/ref-costs < risk_per_lot:
                    raise Blocked("Scalp reward does not cover risk after configured costs")
                decision.update(action="BUY" if signal["side"] > 0 else "SELL", reason=signal["kind"])
                if execute:
                    memory.put_state(decision_key, decision)
                    memory.put_state("scalp-attempt:"+scope+":"+str(int(closed_at)),
                        {"order_key": key, "state": vector.tolist(), "risk_per_lot": risk_per_lot,
                         "created_at": now, "setup": signal})
                result = bridge.send_order(request, key, memory, expected_login,
                    **send_options, entry_deadline=deadline)
                if result["status"] == "sent":
                    guard.entries += 1
                    memory.put_state(guard_key, asdict(guard))
                    memory.put_state("scalp-used:"+scope+":"+signal["setup_id"], True)
                decision.update(result)
            except Blocked as e:
                reason = str(e)
                if getattr(e, "details", None):
                    decision["sizing"] = e.details
                    decision["reason_code"] = e.code
        if reason:
            decision.update(action="WAIT", reason=reason, status="blocked")
    memory.put_state(decision_key, decision)
    signature = lambda d: (d.get("action"), d.get("reason"), d.get("status"))
    event_key = decision_key + (json.dumps(signature(decision)) if cfg.style == "active" else "")
    memory.record(scope, spec.symbol, "scalp_decision", closed_at, max(closed_at, time.time()), decision,
                  hashlib.sha256(event_key.encode()).hexdigest())
    stats = decision_stats(memory, scope, decision, time.time())
    bridge.scalp_status.update(structure=structure[-1], last_decision=decision,
                              entries_today=guard.entries, activity_stats=stats)
    if previous_decision and signature(previous_decision) == signature(decision):
        decision["repeat"] = True
    return decision
