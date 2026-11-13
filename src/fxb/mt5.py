"""MT5 transport with separate dry-run, demo and explicitly activated live paths."""
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib
import json
import math
import os
import time
import uuid
from types import SimpleNamespace
import numpy as np
from .clock import BrokerClock
from .data import Bars, KABUL, day_key, timestamp, features
from .risk import Spec, Guard, Risk, levels, size_lots
from .simulator import portfolio_state, ACTION_NAMES
from .research import load_checkpoint

MAGIC = 9261109
# MQL5 SYMBOL_ORDER_MODE flags. These are not consistently exported by the
# Python wheel; their values come from the official Symbol Properties table.
ORDER_CAPABILITY_MARKET = 1
ORDER_CAPABILITY_SL = 16
ORDER_CAPABILITY_TP = 32


class Blocked(RuntimeError):
    pass


class WorkerLock:
    """Single worker per state directory; a crash leaves a fail-closed lock."""
    def __init__(self, database):
        self.path = Path(str(database) + ".worker.lock")

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise Blocked("Worker lock exists; verify no worker is running before removing it") from None
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        return self

    def __exit__(self, *args):
        self.path.unlink()


class NewsGate:
    def __init__(self, path):
        self.path = path

    def check(self, now, horizon=0):
        if not self.path:
            raise Blocked("A verified economic-calendar file is required for new entries")
        try:
            doc = json.loads(Path(self.path).read_text(encoding="utf-8"))
            start, end, updated = [timestamp(doc[k]) for k in ("coverage_start", "coverage_end", "updated_at")]
            events = doc["events"]
            if (not isinstance(events, list) or not isinstance(doc["source_url"], str)
                    or not doc["source_url"].startswith("https://")):
                raise ValueError("Missing calendar provenance")
            if not start <= now or not now + horizon < end or not 0 <= now - updated <= 86400:
                raise Blocked("Economic-calendar coverage is missing or older than 24 hours")
            for event in events:
                if (not isinstance(event, dict) or event.get("impact") not in ("low", "medium", "high")
                        or not isinstance(event.get("currency"), str)):
                    raise ValueError("Invalid event schema")
                at = timestamp(event["time"])
                if event["currency"] == "USD" and event["impact"] == "high" and now - 900 <= at <= now + max(900, horizon):
                    raise Blocked("High-impact USD news blackout")
        except (KeyError, ValueError, TypeError, OSError) as e:
            raise Blocked("Invalid economic-calendar file") from e


class MT5:
    def __init__(self, api=None, terminal=None, clock=None):
        self.api = api
        self.terminal = terminal
        self.clock = clock if isinstance(clock, BrokerClock) else BrokerClock.from_config(clock)

    def utc(self, value):
        try:
            return self.clock.to_utc(value)
        except ValueError as e:
            raise Blocked(str(e)) from e

    def timed_record(self, record, fields):
        changes = {}
        for field in fields:
            value = getattr(record, field, None)
            if value is not None and value > 0:
                unit = 1000 if field.endswith("_msc") else 1
                changes[field] = int(round(self.utc(value / unit) * unit))
        if hasattr(record, "_replace"):
            return record._replace(**changes)
        # MT5's native Tick/TradePosition/TradeDeal records expose _asdict(),
        # but do not have the __dict__ or _replace of our Python test doubles.
        as_dict = getattr(record, "_asdict", None)
        if callable(as_dict):
            values = as_dict()
        elif hasattr(record, "__dict__"):
            values = vars(record)
        else:
            raise Blocked(f"Unsupported MT5 record type: {type(record).__name__}")
        return SimpleNamespace(**(values | changes))

    def history_deals(self, start, end):
        try:
            begin, finish = self.clock.query_bounds(start, end)
        except ValueError as e:
            raise Blocked(str(e)) from e
        raw = self.api.history_deals_get(begin, finish)
        if raw is None:
            raise Blocked("Trade history unavailable; cannot reconcile realized outcomes")
        deals = []
        low = min(self.clock.from_utc(start), self.clock.from_utc(end))
        high = max(self.clock.from_utc(start), self.clock.from_utc(end))
        if self.clock.zone not in ("UTC",) and not self.clock.zone.startswith(("+", "-")):
            low, high = low - 3600, high + 3600
        for row in raw:
            # Padded API queries may include rows outside the validated clock
            # interval. Only skip those outside the requested raw-time bounds;
            # an ambiguous time inside the interval must still fail closed.
            if not low <= row.time <= high:
                continue
            normalized = self.timed_record(row, ("time", "time_msc"))
            if start <= normalized.time <= end:
                deals.append(normalized)
        return tuple(deals)

    def connect(self):
        if self.api is None:
            if os.name != "nt":
                raise Blocked("MT5 execution requires 64-bit Windows and a locally installed MT5 terminal")
            try:
                import MetaTrader5
            except ImportError:
                raise Blocked('Install the Windows extra: pip install -e ".[mt5]"') from None
            self.api = MetaTrader5
        ok = self.api.initialize(self.terminal) if self.terminal else self.api.initialize()
        if not ok:
            raise Blocked("MT5 initialization failed")
        return self

    def close(self):
        if self.api is not None:
            self.api.shutdown()

    def account(self, expected_login=None, demo=False, execution_mode=None, expected_server=None):
        account = self.api.account_info()
        terminal = self.api.terminal_info()
        if account is None or terminal is None or not terminal.connected:
            raise Blocked("MT5 account or terminal unavailable")
        if expected_login is not None and account.login != expected_login:
            raise Blocked("Connected MT5 login differs from the configured account")
        if expected_server is not None and account.server != expected_server:
            raise Blocked("Connected MT5 server differs from the configured broker server")
        mode = execution_mode or ("demo" if demo else None)
        if mode:
            if mode not in ("demo", "live"):
                raise Blocked("Invalid execution mode")
            required_mode = self.api.ACCOUNT_TRADE_MODE_DEMO if mode == "demo" else self.api.ACCOUNT_TRADE_MODE_REAL
            if expected_login is None or account.trade_mode != required_mode:
                raise Blocked(f"Execution requires the configured {mode.upper()} account")
            if mode == "live" and not expected_server:
                raise Blocked("The exact broker server is required for live execution")
            if not account.trade_allowed or not account.trade_expert or not terminal.trade_allowed or terminal.tradeapi_disabled:
                raise Blocked("Terminal/account algorithmic trading permission is disabled")
        # Exhausted margin/equity must block new risk, not an owned-position close.
        if not all(math.isfinite(v) for v in (account.equity, account.margin_free)):
            raise Blocked("Invalid equity or free margin")
        return account

    def symbol(self, name):
        if not self.api.symbol_select(name, True):
            raise Blocked("Exact broker symbol could not be selected")
        info = self.api.symbol_info(name)
        if info is None:
            raise Blocked("Symbol metadata unavailable")
        if info.chart_mode != self.api.SYMBOL_CHART_MODE_BID:
            raise Blocked("This release only supports bid-chart CFDs, not last-price futures")
        return info

    def tick(self, symbol, now=None):
        tick = self.api.symbol_info_tick(symbol)
        now = time.time() if now is None else now
        return self.checked_tick(tick, now)

    def checked_tick(self, tick, now):
        if (tick is None or not all(math.isfinite(v) for v in (tick.bid, tick.ask, tick.time_msc))
                or tick.bid <= 0 or tick.ask < tick.bid):
            raise Blocked("Missing, stale or invalid executable quote")
        tick = self.timed_record(tick, ("time", "time_msc"))
        if not -2 <= now - tick.time_msc / 1000 <= 10:
            raise Blocked("Quote time differs from UTC or is stale; verify Windows clock and the configured broker clock policy")
        return tick

    def clock_snapshot(self, symbol):
        self.symbol(symbol)
        raw = self.api.symbol_info_tick(symbol)
        now = time.time()
        result = {"symbol": symbol, "windows_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                  "broker_clock": self.clock.config(), "ready": False}
        if raw is not None:
            result["raw_time_msc"] = raw.time_msc
        try:
            quote = self.checked_tick(raw, now)
            result.update(ready=True, quote_utc=datetime.fromtimestamp(quote.time_msc / 1000, timezone.utc).isoformat(),
                          quote_age_seconds=now - quote.time_msc / 1000)
        except (Blocked, ValueError) as e:
            result["reason"] = str(e)
        return result

    def positions(self):
        positions, orders = self.api.positions_get(), self.api.orders_get()
        if positions is None or orders is None:
            raise Blocked("Cannot reconcile positions or pending orders")
        positions = tuple(self.timed_record(p, ("time", "time_msc", "time_update", "time_update_msc")) for p in positions)
        return positions, orders

    def spec(self, symbol, commission, slippage, overnight, timeframe_seconds=300):
        info, account = self.symbol(symbol), self.account()
        tick = self.tick(symbol)
        ref = max(info.volume_min, min(1.0, info.volume_max))
        margin = self.api.order_calc_margin(self.api.ORDER_TYPE_BUY, symbol, ref, tick.ask)
        if margin is None or not math.isfinite(margin) or margin <= 0:
            raise Blocked("Cannot obtain margin in the account currency")
        return Spec(symbol, account.currency, timeframe_seconds, info.point, info.trade_tick_size,
                    info.trade_tick_value_loss, info.volume_min, info.volume_max, info.volume_step,
                    margin / ref, commission, slippage, overnight, info.trade_stops_level * info.point)

    def bars(self, spec, count=512):
        timeframe = self.api.TIMEFRAME_M1 if spec.timeframe_seconds == 60 else self.api.TIMEFRAME_M5
        raw = self.api.copy_rates_from_pos(spec.symbol, timeframe, 1, count)
        if raw is None or len(raw) < min(count, 512):
            raise Blocked(f"Insufficient closed M{spec.timeframe_seconds // 60} history; load more bars in MT5")
        times = np.asarray([self.utc(t) for t in raw["time"]], dtype=float)
        return Bars(times, raw["open"], raw["high"], raw["low"], raw["close"],
                    raw["tick_volume"].astype(float), raw["spread"] * spec.point, spec,
                    "MT5 broker bid bars; margin/tick-value snapshot; bar spread approximation", False,
                    {"storage": "UTC", "broker_clock": self.clock.config()}).validate()

    def filling(self, info):
        if info.filling_mode & 1:
            return self.api.ORDER_FILLING_FOK
        if info.filling_mode & 2:
            return self.api.ORDER_FILLING_IOC
        if info.trade_exemode != self.api.SYMBOL_TRADE_EXECUTION_MARKET:
            return self.api.ORDER_FILLING_RETURN
        raise Blocked("No supported market-order filling policy")

    def request(self, symbol, side, atr, spec, risk, account, now=None, structural_stop=None):
        info, tick = self.symbol(symbol), self.tick(symbol, now)
        if info.trade_mode != self.api.SYMBOL_TRADE_MODE_FULL:
            raise Blocked("Symbol does not permit full trading")
        required = ORDER_CAPABILITY_MARKET | ORDER_CAPABILITY_SL | ORDER_CAPABILITY_TP
        if info.order_mode & required != required:
            raise Blocked("Broker-side SL and TP are mandatory")
        for name, expected in (("point", spec.point), ("trade_tick_size", spec.tick_size),
                                ("volume_min", spec.lot_min), ("volume_step", spec.lot_step), ("volume_max", spec.lot_max)):
            if not math.isclose(getattr(info, name), expected, rel_tol=1e-9):
                raise Blocked("Broker contract changed; re-export metadata and revalidate the model")
        spread = tick.ask - tick.bid
        if not math.isfinite(atr) or atr <= 0 or spread > risk.max_spread_atr * atr:
            raise Blocked("Spread is too large relative to ATR")
        entry = tick.ask if side > 0 else tick.bid
        # Include spread when respecting the broker's stop distance from its closing quote.
        active_spec = Spec(**(asdict(spec) | {"stops_level": info.trade_stops_level * info.point + spread}))
        if structural_stop is None:
            stop, target = levels(side, entry, atr, active_spec, risk)
        else:
            from .risk import round_price
            if not math.isfinite(structural_stop) or structural_stop <= 0:
                raise Blocked("Invalid structural stop")
            stop = round_price(structural_stop, spec.tick_size, up=side < 0)
            distance = side * (entry - stop)
            if distance <= active_spec.stops_level or distance > 2.5 * atr or spread > .25 * distance:
                raise Blocked("Structural stop conflicts with broker distance, spread or M1 volatility")
            target = round_price(entry + side * distance * risk.reward_risk, spec.tick_size, up=side > 0)
        order_type = self.api.ORDER_TYPE_BUY if side > 0 else self.api.ORDER_TYPE_SELL
        ref = max(info.volume_min, min(1.0, info.volume_max))
        slip = math.ceil(spec.slippage_points) * spec.point
        loss = self.api.order_calc_profit(order_type, symbol, ref, entry + side * slip, stop - side * slip)
        margin = self.api.order_calc_margin(order_type, symbol, ref, entry + side * slip)
        if loss is None or margin is None or not math.isfinite(loss) or loss >= 0:
            raise Blocked("Broker could not price stop-loss risk")
        loss_per_lot = -loss / ref + 2 * spec.commission_per_lot_side
        lots = size_lots(account.equity, loss_per_lot, account.margin_free, margin / ref, active_spec, risk)
        if not lots:
            raise Blocked("Minimum lot would exceed risk or margin budget")
        return {"action": self.api.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lots,
                "type": order_type, "price": entry, "sl": stop, "tp": target,
                "deviation": int(math.ceil(spec.slippage_points)), "magic": MAGIC,
                "type_time": self.api.ORDER_TIME_GTC, "type_filling": self.filling(info)}

    def close_request(self, position):
        info, tick = self.symbol(position.symbol), self.tick(position.symbol)
        buy = position.type == self.api.POSITION_TYPE_BUY
        return {"action": self.api.TRADE_ACTION_DEAL, "position": position.ticket,
                "symbol": position.symbol, "volume": position.volume,
                "type": self.api.ORDER_TYPE_SELL if buy else self.api.ORDER_TYPE_BUY,
                "price": tick.bid if buy else tick.ask, "deviation": 20, "magic": MAGIC,
                "type_time": self.api.ORDER_TIME_GTC, "type_filling": self.filling(info)}

    def send_order(self, request, key, memory, expected_login, execute=False, mode="demo",
                   expected_server=None, authorization=None, model_hash=None, risk_config=None,
                   entry_period_seconds=300, entry_deadline=None):
        if entry_period_seconds not in (60, 300):
            raise Blocked("Unsupported entry interval")
        account = self.account(expected_login, execution_mode=mode, expected_server=expected_server)
        if mode == "live" and execute:
            if authorization is None:
                raise Blocked("Live trading has not been activated by the operator")
            try:
                authorization.validate(account.login, account.server, model_hash, risk_config,
                                       reducing_risk="position" in request)
            except ValueError as e:
                raise Blocked(str(e)) from e
        positions, orders = self.positions()
        if "position" in request:
            own = [p for p in positions if p.ticket == request["position"] and p.magic == MAGIC
                   and p.symbol == request["symbol"]]
            if len(own) != 1 or request["volume"] > own[0].volume:
                raise Blocked("Closing request does not match an owned position")
        else:
            if positions or orders:
                raise Blocked("Account already has exposure or pending orders")
            buy = request["type"] == self.api.ORDER_TYPE_BUY
            entry, stop, target = request["price"], request.get("sl", 0), request.get("tp", 0)
            if not (0 < stop < entry < target if buy else 0 < target < entry < stop):
                raise Blocked("Opening orders require valid broker-side SL and TP")
            slot = (f"entry-slot:{expected_login}:{int(time.time() // 300)}" if entry_period_seconds == 300 else
                    f"entry-slot-m1:{expected_login}:{int(time.time() // 60)}")
            if execute and memory.get_state(slot) not in (None, key):
                raise Blocked("Another policy already reserved this account's entry for the current bar")
            if entry_deadline is not None and time.time() > entry_deadline:
                raise Blocked("Scalp entry window expired")
        if not execute:
            return {"status": "dry_run", "request": request}
        if memory.unresolved():
            raise Blocked("Uncertain previous order exists; reconcile it in MT5 before continuing")
        request = request | {"comment": "fxb-" + hashlib.sha256(key.encode()).hexdigest()[:20]}
        check = self.api.order_check(request)
        if check is None or check.retcode != 0:
            raise Blocked("Broker order_check rejected the request")
        if not memory.claim_order(key, request):
            return {"status": "duplicate_blocked"}
        if "position" not in request:
            memory.put_state(slot, key)
        # Recheck the account and quote immediately before the only submission call.
        try:
            self.account(expected_login, execution_mode=mode, expected_server=expected_server)
            if "position" not in request and entry_deadline is not None and time.time() > entry_deadline:
                raise Blocked("Scalp entry window expired before submission")
            if mode == "live":
                authorization.validate(account.login, account.server, model_hash, risk_config,
                                       reducing_risk="position" in request)
            tick = self.tick(request["symbol"])
            quote = tick.ask if request["type"] == self.api.ORDER_TYPE_BUY else tick.bid
            point = self.symbol(request["symbol"]).point
            if abs(quote - request["price"]) > request["deviation"] * point:
                memory.order_status(key, "rejected", {"reason": "quote_moved"})
                raise Blocked("Quote moved outside the configured deviation; no send")
        except Exception:
            memory.order_status(key, "rejected", {"reason": "pre_send_check_failed"})
            raise
        try:
            result = self.api.order_send(request)
        except Exception:
            memory.order_status(key, "ambiguous", {"request": request})
            raise Blocked("Order outcome is unknown; automatic retry is disabled") from None
        if result is None:
            memory.order_status(key, "ambiguous", {"request": request})
            raise Blocked("MT5 returned no order result; reconcile before restarting")
        if result.retcode not in (self.api.TRADE_RETCODE_DONE, self.api.TRADE_RETCODE_DONE_PARTIAL):
            # Even a server timeout or PLACED response is ambiguous: never send again.
            memory.order_status(key, "ambiguous", {"request": request, "retcode": result.retcode})
            raise Blocked(f"Unconfirmed MT5 result {result.retcode}; reconcile before restarting")
        payload = {"request": request, "retcode": result.retcode, "order": result.order,
                   "deal": result.deal, "volume": result.volume, "price": result.price}
        memory.order_status(key, "sent", payload)
        return {"status": "sent", **payload}

    def send_demo(self, request, key, memory, expected_login, execute=False):
        """Backwards-compatible demo entry point, with no live-account fallback."""
        return self.send_order(request, key, memory, expected_login, execute, mode="demo")

    def policy(self, model_path):
        path = Path(model_path).resolve()
        stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        cache = getattr(self, "_policies", {})
        if stamp not in cache:
            cache = {k: v for k, v in cache.items() if k[0] != str(path)}
            cache[stamp] = (*load_checkpoint(path), hashlib.sha256(path.read_bytes()).hexdigest())
            self._policies = cache
        return cache[stamp]


def collect_deals(bridge, memory, account_key, now):
    deals = bridge.history_deals(max(now - 7 * 86400, bridge.clock.lower), now + 1)
    for d in deals:
        if d.magic == MAGIC:
            payload = {k: getattr(d, k) for k in ("ticket", "position_id", "symbol", "type", "entry",
                       "volume", "price", "profit", "commission", "swap", "fee", "comment")}
            payload["net_cashflow"] = d.profit + d.commission + d.swap + d.fee
            memory.record(account_key, d.symbol, "broker_deal", d.time, max(d.time, now), payload,
                          str(uuid.uuid5(uuid.NAMESPACE_URL, f"{account_key}:deal:{d.ticket}")))


def cycle(bridge, model_path, memory, expected_login, calendar=None, execute=False, kill_file="STOP",
          mode="demo", expected_server=None, authorization=None, entries_enabled=True):
    agent, reservoir, scaler, meta, model_hash = bridge.policy(model_path)
    spec, risk = Spec(**meta["spec"]), Risk(**meta["risk"])
    now = time.time()
    account = bridge.account(expected_login, execution_mode=mode, expected_server=expected_server)
    if spec.currency != account.currency:
        raise Blocked("Account currency differs from training metadata")
    if execute and meta["synthetic"]:
        raise Blocked("A model trained on synthetic prices cannot send broker orders")
    send_options = dict(execute=execute, mode=mode, expected_server=expected_server,
                        authorization=authorization, model_hash=model_hash, risk_config=meta["risk"])
    account_key = hashlib.sha256(f"{account.server}:{account.login}".encode()).hexdigest()[:20]
    collect_deals(bridge, memory, account_key, now)
    guard_key = "guard:" + account_key
    guard = Guard(**memory.get_state(guard_key, {}))
    guard.update(account.equity, day_key(now), risk)
    # Preserve a journaled count, but reconcile broker entries after a process crash.
    midnight = datetime.fromtimestamp(now, KABUL).replace(hour=0, minute=0, second=0, microsecond=0)
    day_deals = bridge.history_deals(midnight.timestamp(), now + 1)
    entry_orders = {d.order for d in day_deals if d.entry in (bridge.api.DEAL_ENTRY_IN, bridge.api.DEAL_ENTRY_INOUT)}
    guard.entries = max(guard.entries, len(entry_orders))
    memory.put_state(guard_key, asdict(guard))
    positions, pending = bridge.positions()
    owned = [p for p in positions if p.magic == MAGIC and p.symbol == spec.symbol]
    if len(owned) > 1:
        raise Blocked("Multiple owned positions require manual reconciliation")
    p = owned[0] if owned else None
    age = max(0, int((now - p.time) // 300)) if p else 0
    stop_requested = Path(kill_file).exists()
    # Risk exits do not wait for candles or a valid news calendar.
    force_close = bool(p and (stop_requested or guard.total_halt or guard.daily_halt
                             or age >= risk.max_hold_bars or p.sl <= 0 or p.tp <= 0))
    if force_close:
        key = f"{account_key}:{p.ticket}:risk-close:{int(now // 300)}"
        result = bridge.send_order(bridge.close_request(p), key, memory, expected_login, **send_options)
        return {"action": "CLOSE", "reason": "risk_time_or_stop_file", **result}
    if stop_requested:
        return {"action": "WAIT", "reason": "stop_file"}
    expected_clock = (meta.get("time_basis") or {}).get("broker_clock")
    if expected_clock is None:
        if bridge.clock.zone != "UTC":
            raise Blocked("This model has no broker clock provenance; re-export data and retrain after correcting time")
    elif not bridge.clock.compatible(expected_clock):
        raise Blocked("Model and connected broker use different clock policies; revalidate before execution")
    bars = bridge.bars(spec)
    closed_at = bars.time[-1] + 300
    if not 0 <= now - closed_at < 300:
        raise Blocked("Latest closed bar is stale or in the future")
    key = f"{account_key}:{meta['run_id']}:{spec.symbol}:{int(closed_at)}"
    if execute and memory.get_state("decision:" + key):
        return {"action": "WAIT", "reason": "already_processed_bar"}
    raw, atr = features(bars)
    normalized = scaler.transform(raw)
    encoded = reservoir.observe(normalized, bars.time) if hasattr(reservoir, "observe") else reservoir.encode(normalized)[-1]
    side = (1 if p.type == bridge.api.POSITION_TYPE_BUY else -1) if p else 0
    state = np.r_[encoded, portfolio_state(side, age, p.profit if p else 0, account.equity, guard, risk)]
    slot = f"entry-slot:{expected_login}:{int(now // 300)}"
    can_enter = (entries_enabled and not positions and not pending and not guard.reason(risk)
                 and (not execute or memory.get_state(slot) is None))
    if mode == "live" and execute and can_enter:
        try:
            if authorization is None:
                raise ValueError("Live trading requires explicit activation")
            authorization.validate(account.login, account.server, model_hash, meta["risk"])
        except ValueError:
            can_enter = False
    mask = np.asarray([True, can_enter, can_enter, bool(p)])
    action = agent.act(state, mask)
    decision = {"action": ACTION_NAMES[action], "q_values": agent.online.predict(state).tolist(),
                "reason": guard.reason(risk) or "frozen_policy", "synthetic_model": meta["synthetic"],
                "model_run": meta["run_id"], "bar_closed_at": closed_at,
                "features": dict(zip(meta["features"], raw[-1].tolist()))}
    # Persist decision before submission; a crash cannot cause a new action for this bar.
    if execute:
        memory.put_state("decision:" + key, decision)
    memory.record(meta["run_id"], spec.symbol, "decision", closed_at, now, decision,
                  str(uuid.uuid5(uuid.NAMESPACE_URL, key + (":execute" if execute else ":dry"))))
    if action in (1, 2):
        try:
            NewsGate(calendar).check(now, risk.max_hold_bars * 300)
            request = bridge.request(spec.symbol, 1 if action == 1 else -1, atr[-1], spec, risk, account)
        except Blocked as e:
            veto = decision | {"status": "blocked", "reason": str(e)}
            memory.record(meta["run_id"], spec.symbol, "decision", closed_at, now, veto,
                          str(uuid.uuid5(uuid.NAMESPACE_URL, key + ":veto")))
            return veto
        result = bridge.send_order(request, key, memory, expected_login, **send_options)
        if result["status"] == "sent":
            guard.entries += 1
            memory.put_state(guard_key, asdict(guard))
    elif action == 3:
        result = bridge.send_order(bridge.close_request(p), key, memory, expected_login, **send_options)
    else:
        result = {"status": "no_order"}
    return decision | result


def reconcile(bridge, memory, expected_login, mode="demo", expected_server=None):
    account = bridge.account(expected_login, execution_mode=mode, expected_server=expected_server)
    account_key = hashlib.sha256(f"{account.server}:{account.login}".encode()).hexdigest()[:20]
    now = time.time()
    unresolved = memory.unresolved()
    if not unresolved:
        return {"resolved": [], "remaining": 0, "note": "No uncertain orders to reconcile."}
    deals = bridge.history_deals(max(now - 30 * 86400, bridge.clock.lower), now + 1)
    resolved = []
    for row in memory.unresolved():
        if not row["key"].startswith(account_key + ":"):
            continue
        payload = json.loads(row["payload"])
        request = payload.get("request", payload)
        matching = [d for d in deals if d.magic == MAGIC and d.symbol == request.get("symbol")
                    and d.comment == request.get("comment")]
        if matching:
            memory.order_status(row["key"], "sent", {"request": request, "reconciled_deals": [d.ticket for d in matching]})
            resolved.append(row["key"])
    return {"resolved": resolved, "remaining": len(memory.unresolved()),
            "note": "Absence of a matching deal does not prove rejection; unresolved records remain blocked."}
