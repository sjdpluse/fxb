"""Local workstation controller: authentic telemetry, background jobs, trading control.

No account credentials are accepted or stored; MT5 uses its local terminal session.
The HTTP layer queues expiring manual intents; only the controlled worker sends orders.
"""
from dataclasses import asdict
from pathlib import Path
import gc
import hashlib
import json
import os
import re
import threading
import time
import uuid
import numpy as np
from . import __version__
from .connectome import WholeBrain, install, read_manifest, geometry, sha256
from .data import load_bars, save_bars, timestamp
from .clock import BrokerClock
from .execution import LiveAuthorization
from .memory import Memory, SupabaseSink
from .mt5 import MT5, Blocked, WorkerLock, cycle, reconcile, MAGIC, NewsGate
from .research import train


def metadata(path):
    with np.load(path, allow_pickle=False) as f:
        return json.loads(str(f["meta"]))


class Workstation:
    def __init__(self, root, preview=False):
        self.root = Path(root).resolve()
        self.state_dir = self.root / ("state/preview" if preview else "state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.brain_dir = self.root / "data/connectome"
        os.environ["FXB_BRAIN_DIRECTORY"] = str(self.brain_dir)
        self.db_path = self.state_dir / "runtime.sqlite"
        self.preview = preview
        self.lock = threading.RLock()
        self.job = {"status": "idle", "kind": None}
        self.cancel_job = threading.Event()
        self.worker_stop = threading.Event()
        self.calendar_stop = threading.Event()
        self.calendar_wake = threading.Event()
        self.calendar_thread = None
        self.calendar_refresh = {"status": "disabled"}
        self.bridge = None
        self.brain = None
        self.worker = None
        self.worker_state = {"phase": "stopped", "reason": None}
        self.authorization = None
        self.entries_enabled = False
        self.mode = "dry-run"
        self.account_data = None
        self.positions = []
        self.quotes = {}
        self.last_decisions = []
        self.events = []
        self.last_error = None
        self.active_models = []
        self.manual_pending = None
        memory = Memory(self.db_path)
        try:
            self.manual_result = memory.get_state("manual-last", None)
            if self.manual_result and self.manual_result.get("status") in ("queued", "processing"):
                try:
                    with WorkerLock(self.db_path):
                        self.recover_manual(memory)
                except Blocked:
                    pass  # A second UI must not rewrite an active worker's intent.
        finally:
            memory.close()
        self.config = {"login": None, "server": "", "terminal": "", "calendar": "data/news.json", "calendar_auto": False,
                       "broker_clock": BrokerClock().config()}
        path = self.state_dir / "workstation.json"
        if path.exists():
            self.config.update(json.loads(path.read_text(encoding="utf-8-sig")))

    def recover_manual(self, memory):
        """Caller holds WorkerLock; re-read so another UI cannot overwrite a receipt."""
        current = memory.get_state("manual-last")
        if current and current.get("status") in ("queued", "processing"):
            current = current | {"status": "interrupted",
                                 "reason": "Worker restarted; inspect broker history. No automatic replay."}
            memory.put_state("manual:"+current["request_id"], current)
            memory.put_state("manual-last", current)
        self.manual_result = current
        self.manual_pending = None

    def note(self, message, level="info", **extra):
        with self.lock:
            self.events.append({"time": time.time(), "message": message, "level": level, **extra})
            self.events = self.events[-100:]

    def resolve(self, value, parent=None, must_exist=True):
        base = self.root if parent is None else self.root / parent
        path = (self.root / value).resolve()
        if not path.is_relative_to(base.resolve()) or (must_exist and not path.is_file()):
            raise ValueError("Choose an existing file inside the workstation project")
        return path

    def models(self):
        result = []
        for path in sorted((self.root / "runs").glob("**/model.npz")):
            try:
                meta = metadata(path)
                report_path = path.parent / "report.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else {}
                result.append({"path": str(path.relative_to(self.root)), "name": path.parent.name,
                    "symbol": meta["spec"]["symbol"], "synthetic": meta["synthetic"],
                    "encoder": meta.get("encoder_type", meta.get("topology", {}).get("kind", "control")),
                    "run_id": meta["run_id"], "risk": meta["risk"], "source": meta["data_source"],
                    "strategy": meta.get("strategy"), "timeframe_seconds": meta["spec"]["timeframe_seconds"],
                    "scalp_style": meta.get("scalp_config", {}).get("style", "structure"),
                    "micro_config": meta.get("micro_config"),
                    "holdout": report.get("holdout"), "selected_epoch": meta.get("selected_epoch"),
                    "sha256": sha256(path)})
            except (ValueError, KeyError, OSError):
                continue
        return result

    def datasets(self):
        result = []
        for path in sorted((self.root / "data").glob("**/*.meta.json")):
            csv_path = path.with_name(path.name.replace(".meta.json", ".csv"))
            if csv_path.exists():
                meta = json.loads(path.read_text(encoding="utf-8"))
                result.append({"path": str(csv_path.relative_to(self.root)), "symbol": meta["spec"]["symbol"],
                               "synthetic": meta["synthetic"], "source": meta["source"], "spec": meta["spec"]})
        return result

    def status(self):
        try:
            manifest = read_manifest(self.brain_dir)
        except (OSError, ValueError):
            manifest = None
        activity = self.brain.latest if self.brain is not None else {"status": "idle", "active_neurons": 0, "spikes": 0}
        if self.bridge is not None:
            for policy in getattr(self.bridge, "_policies", {}).values():
                if hasattr(policy[1], "latest") and policy[1].net is not None:
                    activity = policy[1].latest | {"input_source": "closed_market_bars"}
            for policy in getattr(self.bridge, "_scalp_policies", {}).values():
                if policy[0].net is not None:
                    activity = policy[0].latest | {"input_source": "closed_market_bars"}
            activity = activity | {"execution": getattr(self.bridge, "neural_status", None)}
        memory = Memory(self.db_path)
        try:
            journal = memory.stats()
        finally:
            memory.close()
        try:
            horizon = 300 if getattr(self.bridge, "_scalp_policies", {}) else 3600
            NewsGate(str(self.root / self.config["calendar"])).check(time.time(), horizon)
            news = {"ready": True, "reason": f"Calendar covers the next {horizon//60} minutes"}
        except (Blocked, OSError, ValueError) as e:
            news = {"ready": False, "reason": str(e)}
        news["automatic"] = bool(self.config.get("calendar_auto"))
        news["refresh"] = dict(self.calendar_refresh)
        return {"version": __version__, "preview": self.preview, "platform": os.name, "news": news,
                "running": bool(self.worker and self.worker.is_alive()), "entries_enabled": self.entries_enabled,
                "worker": dict(self.worker_state),
                "mode": self.mode, "account": self.account_data, "positions": self.positions, "quotes": self.quotes,
                "models": self.models(), "datasets": self.datasets(), "active_models": self.active_models,
                "brain": {"manifest": manifest, "activity": activity}, "job": dict(self.job),
                "journal": journal, "decisions": self.last_decisions[-20:], "events": self.events[-25:],
                "scalper": getattr(self.bridge, "scalp_status", None),
                "manual_order": self.manual_result,
                "error": self.last_error, "live_activation": self.authorization.public_status() if self.authorization else None,
                "config": self.config, "connected": self.bridge is not None}

    def launch_job(self, kind, function):
        with self.lock:
            if self.job["status"] == "running":
                raise ValueError("Another background operation is still running")
            if self.worker and self.worker.is_alive() and kind in ("brain_install", "brain_stimulate", "train", "scalp_prepare"):
                raise ValueError("Stop the trading worker before changing or stimulating its neural engine")
            self.cancel_job.clear()
            self.job = {"status": "running", "kind": kind, "started_at": time.time(), "progress": {}}
        def update(value):
            self.job["progress"] = value
            if self.cancel_job.is_set():
                raise InterruptedError("Background operation cancelled")
        def run():
            try:
                result = function(update)
                self.job.update(status="completed", result=result, finished_at=time.time())
                self.note(f"{kind} completed")
            except InterruptedError:
                self.job.update(status="cancelled", finished_at=time.time())
            except Exception as e:
                self.job.update(status="failed", error=str(e), finished_at=time.time())
                self.note(str(e), "error", operation=kind)
        threading.Thread(target=run, daemon=True, name=f"fxb-{kind}").start()
        return {"status": "started", "kind": kind}

    def install_brain(self):
        return self.launch_job("brain_install", lambda update: install(self.brain_dir, update))

    def stimulate(self, channels="all", frames=8, rate=150):
        if channels not in ("all", "positive", "negative") or not 1 <= frames <= 30 or not 0 <= rate <= 300:
            raise ValueError("Invalid stimulation parameters")
        def work(update):
            if self.brain is None:
                update({"stage": "initializing", "message": "Loading all neurons and connections"})
                self.brain = WholeBrain(self.brain_dir)
            self.brain.initialize()
            indices = list(range(152)) if channels == "all" else list(range(76)) if channels == "positive" else list(range(76, 152))
            for i in range(frames):
                update({"stage": "simulating", "frame": i + 1, "frames": frames})
                self.brain.step(stimulus_indices=indices, rate_hz=rate)
            return {"frames": frames, "activity_source": "manual_input_to_original_full_model"}
        return self.launch_job("brain_stimulate", work)

    def configure(self, data):
        if self.job["status"] == "running" or self.worker and self.worker.is_alive():
            raise ValueError("Stop execution before changing the connected account")
        allowed = {k: data[k] for k in ("login", "server", "terminal", "calendar", "broker_clock") if k in data}
        if "login" in allowed and (type(allowed["login"]) is not int or allowed["login"] <= 0):
            raise ValueError("A positive MT5 account number is required")
        if "calendar" in allowed:
            self.resolve(allowed["calendar"], "data", must_exist=False)
            allowed["calendar_auto"] = False
        if "broker_clock" in allowed:
            allowed["broker_clock"] = BrokerClock.from_config(allowed["broker_clock"]).config()
        if self.bridge:
            self.bridge.close()
            self.bridge = None
            self.account_data = None
            self.positions, self.quotes = [], {}
        self.authorization = None
        self.config.update(allowed)
        (self.state_dir / "workstation.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        return self.config

    def enable_calendar_auto(self):
        if self.preview:
            raise Blocked("Automatic calendar updates require the local workstation")
        from .news import atomic_json, SOURCE
        self.config.update(calendar="data/news-auto.json", calendar_auto=True)
        atomic_json(self.state_dir / "workstation.json", self.config)
        self.start_calendar_refresh()
        self.calendar_wake.set()
        return {"status": "enabled", "source": SOURCE, "orders_sent": 0}

    def start_calendar_refresh(self):
        if self.preview or not self.config.get("calendar_auto"):
            return
        if self.calendar_thread and self.calendar_thread.is_alive():
            return
        from .news import load_weekly, atomic_json, REFRESH_SECONDS
        self.calendar_stop.clear()
        def run():
            while not self.calendar_stop.is_set():
                self.calendar_wake.clear()
                with self.lock:
                    enabled = self.config.get("calendar_auto", False)
                    selected = self.config["calendar"]
                delay = 60
                if enabled:
                    try:
                        path = self.resolve(selected, "data", must_exist=False)
                        self.calendar_refresh = {"status": "refreshing"}
                        doc, cached = load_weekly(path)
                        with self.lock:
                            if (not self.calendar_stop.is_set() and self.config.get("calendar_auto")
                                    and self.config["calendar"] == selected):
                                if not cached:
                                    atomic_json(path, doc)
                                self.calendar_refresh = {"status": "cached" if cached else "updated",
                                    "updated_at": doc["updated_at"], "events": len(doc["events"])}
                        delay = max(30, REFRESH_SECONDS - (time.time() - timestamp(doc["fetched_at"])))
                    except Exception as e:
                        # No network call runs in the trading loop. A failed
                        # download keeps the old file and its original expiry.
                        self.calendar_refresh = {"status": "error", "reason": str(e)}
                        delay = 300
                        retry = (getattr(e, "headers", None) or {}).get("Retry-After", "")
                        if str(retry).isdigit():
                            delay = max(delay, int(retry))
                        elif retry:
                            from email.utils import parsedate_to_datetime
                            try:
                                delay = max(delay, parsedate_to_datetime(retry).timestamp() - time.time())
                            except (ValueError, TypeError, OverflowError):
                                pass
                self.calendar_wake.wait(delay)
        self.calendar_thread = threading.Thread(target=run, daemon=True, name="fxb-calendar")
        self.calendar_thread.start()

    def stop_calendar_refresh(self):
        self.calendar_stop.set()
        self.calendar_wake.set()
        if self.calendar_thread:
            self.calendar_thread.join(timeout=1)

    def connect(self):
        if self.preview:
            raise Blocked("Broker control is available in your local Windows installation")
        if self.job["status"] == "running" or self.worker and self.worker.is_alive():
            raise ValueError("The trading worker is already connected")
        if not self.config["login"]:
            raise ValueError("Configure the MT5 account number first")
        if self.bridge:
            self.bridge.close()
            self.bridge = None
            self.account_data = None
        bridge = MT5(terminal=self.config["terminal"] or None, clock=self.config["broker_clock"])
        bridge.connect()
        try:
            account = bridge.account(self.config["login"], expected_server=self.config["server"] or None)
            self.config["server"] = account.server
            self.bridge = bridge
            self.refresh_account()
            self.note("MT5 terminal connected", account_mode=self.account_data["mode"])
            return self.account_data
        except Exception:
            self.bridge = None
            self.account_data = None
            bridge.close()
            raise

    def refresh_account(self):
        if self.bridge is None:
            return
        a = self.bridge.account(self.config["login"], expected_server=self.config["server"])
        positions, pending = self.bridge.positions()
        self.account_data = {k: getattr(a, k) for k in ("login", "server", "currency", "balance", "equity", "margin", "margin_free", "profit", "leverage")}
        self.account_data.update(mode="demo" if a.trade_mode == 0 else "live" if a.trade_mode == 2 else "contest", updated_at=time.time())
        self.positions = [{**{k: getattr(p, k) for k in ("ticket", "symbol", "type", "volume", "price_open", "price_current", "sl", "tp", "profit", "magic", "time")}, "owned": p.magic == MAGIC} for p in positions]
        self.pending_orders = len(pending)

    def export(self, symbol, bars, commission, slippage, overnight):
        if self.preview or self.bridge is None:
            raise Blocked("Connect the local MT5 terminal before exporting market history")
        if self.worker and self.worker.is_alive():
            raise ValueError("Pause and stop execution before exporting a training dataset")
        if not symbol or any(c in symbol for c in "/\\\0") or not 800 <= bars <= 500000:
            raise ValueError("Invalid broker symbol or bar count")
        def work(update):
            update({"stage": "exporting", "symbol": symbol})
            spec = self.bridge.spec(symbol, commission, slippage, overnight)
            data = self.bridge.bars(spec, bars)
            name = f"{symbol}-{int(time.time())}.csv"
            dest = self.root / "data" / name
            save_bars(data, dest)
            return {"path": str(dest.relative_to(self.root)), "bars": len(data.time), "synthetic": False}
        return self.launch_job("export", work)

    def start_training(self, dataset, epochs=10, encoder="whole-brain"):
        data = self.resolve(dataset, "data")
        if encoder not in ("whole-brain", "fast-brain", "plain") or not 1 <= epochs <= 200:
            raise ValueError("Invalid training configuration")
        bars = load_bars(data)
        if bars.synthetic:
            raise ValueError("Workstation training requires exported market data; CLI supports synthetic software fixtures")
        clock = BrokerClock.from_config(self.config["broker_clock"])
        source_clock = (bars.time_basis or {}).get("broker_clock")
        if (source_clock is None and clock.zone != "UTC") or (source_clock is not None and not clock.compatible(source_clock)):
            raise ValueError("Dataset clock differs from connection settings; export fresh data before training")
        symbol = re.sub(r"[^a-zA-Z0-9_.-]", "_", bars.spec.symbol)
        output = self.root / "runs" / f"{symbol}-{time.time_ns()}"
        def work(update):
            self.brain = None
            gc.collect()
            result = train(bars, output, epochs=epochs, full_brain=encoder == "whole-brain",
                           plain=encoder == "plain", fast_brain=encoder == "fast-brain", progress=update)
            return {"model": str((output / "model.npz").relative_to(self.root)), "holdout": result["holdout"]}
        return self.launch_job("train", work)

    def prepare_scalper(self, symbol, commission, slippage, overnight, style="structure",
                        risk_percent=None, max_trades=None, daily_loss_percent=None,
                        basket_orders=5, basket_lot=.01, basket_risk_percent=2.5):
        from .scalper_runtime import create_profile
        from .scalping import scalp_config, scalp_risk
        if self.preview or self.bridge is None:
            raise Blocked("Connect the local MT5 terminal before preparing the M1 scalper")
        if self.worker and self.worker.is_alive():
            raise ValueError("Stop execution before preparing another trading profile")
        if any(value is not None and type(value) not in (int, float)
               for value in (risk_percent, daily_loss_percent)):
            raise ValueError("Risk percentages must be numbers")
        config = scalp_config(style)
        risk = scalp_risk(style, None if risk_percent is None else risk_percent/100,
                          max_trades, None if daily_loss_percent is None else daily_loss_percent/100)
        from .micro_scalping import MicroConfig
        if type(basket_risk_percent) not in (int, float):
            raise ValueError("Basket risk percentage must be a number")
        micro = MicroConfig(basket_orders, basket_lot, basket_risk_percent/100) if style == "micro" else None
        if micro and micro.basket_risk > risk.daily_loss:
            raise ValueError("Basket risk cannot exceed daily loss limit")
        safe_symbol = re.sub(r"[^a-zA-Z0-9_.-]", "_", symbol)
        suffix = style if style in ("active", "micro") else "scalper"
        output = self.root / "runs" / f"{safe_symbol}-M1-{suffix}-{time.time_ns()}"
        def work(update):
            update({"stage": "reading_M1", "message": "Reading 500 closed gold bars; no historical DQN training"})
            result = create_profile(self.bridge, symbol, output, commission, slippage, overnight, config, risk, micro)
            result["model"] = str(Path(result["model"]).relative_to(self.root))
            return result
        return self.launch_job("scalp_prepare", work)

    def manual_order(self, request_id, side, lot, count=1, check_only=True):
        """At most one queued intent; repeated IDs cannot create another trade."""
        if not isinstance(request_id, str) or str(uuid.UUID(request_id)) != request_id:
            raise ValueError("A canonical UUID request_id is required")
        if (type(side) is not int or side not in (-1, 1) or type(count) is not int or not 1 <= count <= 10
                or type(lot) not in (int, float) or lot not in (.01, .02) or type(check_only) is not bool):
            raise ValueError("Invalid manual order side, lot, count or check mode")
        with self.lock:
            memory = Memory(self.db_path)
            fields = dict(request_id=request_id, side=side, lot=lot, count=count, check_only=check_only)
            try:
                previous = memory.get_state("manual:"+request_id)
                if previous:
                    if any(previous.get(k) != v for k, v in fields.items()):
                        raise ValueError("This request ID already belongs to different order parameters")
                    return previous
                if self.preview or not self.worker or not self.worker.is_alive() or self.worker_stop.is_set():
                    raise Blocked("Start the selected M1 worker before a manual order check or send")
                if len(self.active_models) != 1 or not metadata(self.root/self.active_models[0]).get("strategy"):
                    raise Blocked("Manual orders require one active M1 scalper profile")
                if self.manual_pending or (self.manual_result and self.manual_result["status"] == "processing"):
                    raise Blocked("Wait for the current manual request to finish")
                if not check_only and self.mode == "dry-run":
                    raise Blocked("Dry-run cannot send an order; select Demo or activate Live")
                intent = fields | {"status": "queued", "created_at": time.time(), "expires_at": time.time()+10,
                                   "model": self.active_models[0], "mode": self.mode}
                memory.put_state("manual:"+request_id, intent)
                memory.put_state("manual-last", intent)
                self.manual_pending = intent
                self.manual_result = intent.copy()
                return intent.copy()
            finally:
                memory.close()

    def finish_manual(self, memory, intent, result):
        if intent:
            with self.lock:
                self.manual_result = intent | result | {"completed_at": time.time()}
                memory.put_state("manual:"+intent["request_id"], self.manual_result)
                memory.put_state("manual-last", self.manual_result)

    def start(self, model_paths, mode="dry-run", acknowledged_login=None, hours=8):
        with self.lock:
            return self._start(model_paths, mode, acknowledged_login, hours)

    def _start(self, model_paths, mode, acknowledged_login, hours):
        if self.preview:
            raise Blocked("The preview cannot activate a trading account")
        if self.bridge is None:
            raise Blocked("Connect MT5 first")
        if self.worker and self.worker.is_alive():
            raise ValueError("The worker is already running")
        if self.job["status"] == "running":
            raise ValueError("Wait for the current background operation to finish")
        if mode not in ("dry-run", "demo", "live") or not 1 <= len(model_paths) <= 2:
            raise ValueError("Choose a mode and one or two deployment models")
        paths = [self.resolve(p, "runs") for p in model_paths]
        metas = [metadata(p) for p in paths]
        if any(m.get("strategy") for m in metas) and len(metas) != 1:
            raise ValueError("Run the gold M1 scalper alone on this account worker")
        if len({m["spec"]["symbol"] for m in metas}) != len(metas):
            raise ValueError("Choose only one model per symbol")
        if any(m["risk"] != metas[0]["risk"] for m in metas):
            raise ValueError("Account models must share the same risk limits")
        if mode != "dry-run" and any(m["synthetic"] for m in metas):
            raise Blocked("Synthetic models cannot trade a broker account")
        self.refresh_account()
        broker_mode = self.account_data["mode"]
        if mode != "dry-run" and mode != broker_mode:
            raise Blocked("Chosen execution mode does not match the connected account")
        try:
            lease = WorkerLock(self.db_path).__enter__()
        except Blocked as error:
            self.entries_enabled = False
            self.last_error = str(error)
            self.worker_state = {"phase": "blocked", "reason": str(error),
                                 "reason_code": error.code, "details": error.details}
            raise
        try:
            # Validate recovery under the same lease used by the new worker.
            memory = Memory(self.db_path)
            try:
                ak = hashlib.sha256(f"{self.config['server']}:{self.config['login']}".encode()).hexdigest()[:20]
                gid = memory.get_state("active-basket:"+ak)
                group = memory.get_state("basket:"+gid) if gid else None
                if group and (len(paths) != 1 or sha256(paths[0]) != group["digest"]):
                    raise Blocked("Restart the basket's original profile to finish management and reconciliation")
                self.recover_manual(memory)
            finally:
                memory.close()
            return self._start_worker(paths, metas, mode, broker_mode, acknowledged_login, hours, lease)
        except BaseException as error:
            lease.__exit__()
            self.entries_enabled = False
            self.last_error = str(error)
            self.worker_state = {"phase": "failed", "reason": str(error)}
            raise

    def _start_worker(self, paths, metas, mode, broker_mode, acknowledged_login, hours, lease):
        self.authorization = None
        if mode == "live":
            self.authorization = LiveAuthorization.activate(self.config["login"], self.config["server"],
                [sha256(p) for p in paths], metas[0]["risk"], acknowledged_login, hours)
        self.mode, self.entries_enabled, self.active_models = mode, True, [str(p.relative_to(self.root)) for p in paths]
        self.last_error = None
        self.worker_stop.clear()
        self.brain = None
        # Switching strategies must not retain several full 15M-edge graphs.
        self.bridge._policies = {}
        self.bridge._scalp_policies = {}
        self.bridge._scalp_tick_history = {}
        self.bridge._micro_histories = {}
        self.bridge._micro_context = {}
        self.bridge.scalp_status = None
        self.bridge.neural_status = {"phase": "waiting_for_data", "reason": "Waiting for fresh M1 data"}
        self.worker_state = {"phase": "starting", "reason": None, "lock_recovered": lease.recovered}
        gc.collect()
        (self.root / "STOP").unlink(missing_ok=True)
        def run():
            memory = None
            try:
                memory = Memory(self.db_path)
                self.note("Trading worker started", mode=mode)
                blocked_reasons = {}
                while not self.worker_stop.is_set():
                    self.worker_state["cycle_started_at"] = time.time()
                    for path in paths:
                        with self.lock:
                            intent, self.manual_pending = self.manual_pending, None
                            if intent:
                                self.manual_result = intent | {"status": "processing"}
                        try:
                            from .basket_runtime import execution_step
                            is_scalper = bool(metas[paths.index(path)].get("strategy"))
                            runner = execution_step if is_scalper else cycle
                            extra = {"manual": intent, "cancelled": lambda: self.worker_stop.is_set() or
                                (self.root/"STOP").exists() or (not intent and not self.entries_enabled)} if is_scalper else {}
                            result = runner(self.bridge, path, memory, self.config["login"],
                                calendar=str(self.root / self.config["calendar"]), execute=mode != "dry-run",
                                kill_file=str(self.root / "STOP"), mode=broker_mode,
                                expected_server=self.config["server"], authorization=self.authorization,
                                entries_enabled=self.entries_enabled, **extra)
                            self.finish_manual(memory, intent, result)
                            blocked_reasons.pop(str(path), None)
                            failure = result.get("error") or {}
                            blocked = result.get("status") in ("blocked", "rejected")
                            self.last_error = (failure.get("reason") or result.get("reason")) if blocked else None
                            self.worker_state.update(phase="blocked" if blocked else "waiting" if result.get("action") == "WAIT" else "running",
                                                     reason=failure.get("reason") or result.get("reason"),
                                                     reason_code=failure.get("code"), details=failure.get("details"))
                        except Blocked as e:
                            self.finish_manual(memory, intent, {"status": "blocked", "reason": str(e), "reason_code": e.code, "details": e.details, "orders_sent": 0})
                            self.last_error = str(e)
                            self.worker_state.update(phase="blocked", reason=str(e), reason_code=e.code, details=e.details)
                            if is_scalper and getattr(self.bridge, "scalp_status", None):
                                self.bridge.scalp_status["last_decision"] = {
                                    "action": "WAIT", "reason": str(e), "reason_code": e.code, "details": e.details}
                            if memory.unresolved():
                                self.entries_enabled = False
                            if blocked_reasons.get(str(path)) == str(e):
                                continue
                            blocked_reasons[str(path)] = str(e)
                            self.note(str(e), "warning", operation="execution_wait")
                            result = {"action": "WAIT", "status": "blocked", "reason": str(e)}
                        if result.get("reason") != "already_processed_bar" and not result.get("repeat"):
                            self.last_decisions.append(result | {"time": time.time(), "model": path.parent.name})
                            self.last_decisions = self.last_decisions[-100:]
                    try:
                        self.refresh_account()
                    except Blocked as e:
                        self.last_error = str(e)
                    self.worker_state["cycle_completed_at"] = time.time()
                    self.worker_stop.wait(.5 if any(m.get("scalp_config", {}).get("style") == "micro" for m in metas) else 2)
            except Exception as e:
                self.last_error = str(e)
                self.entries_enabled = False
                self.worker_state.update(phase="failed", reason=str(e))
                self.note(str(e), "error", operation="execution")
            finally:
                try:
                    with self.lock:
                        unfinished = self.manual_pending or (self.manual_result if self.manual_result and
                            self.manual_result.get("status") == "processing" else None)
                        self.manual_pending = None
                        if memory is not None:
                            self.finish_manual(memory, unfinished, {"status": "interrupted", "reason": "Worker stopped; no automatic replay"})
                        self.entries_enabled = False
                        if self.worker_state["phase"] != "failed":
                            self.worker_state.update(phase="stopped", reason=None)
                finally:
                    try:
                        if memory is not None:
                            memory.close()
                    finally:
                        lease.__exit__()
                self.note("Trading worker stopped; inspect any remaining broker positions")
        self.worker = threading.Thread(target=run, daemon=True, name="fxb-execution")
        self.worker.start()
        return {"status": "started", "mode": mode}

    def pause(self):
        self.entries_enabled = False
        self.note("New entries paused; open-position management remains active")
        return {"entries_enabled": False}

    def reconcile_orders(self):
        if self.bridge is None or self.worker and self.worker.is_alive():
            raise Blocked("Connect MT5 and stop the worker before reconciliation")
        def work(update):
            self.refresh_account()
            memory = Memory(self.db_path)
            try:
                return reconcile(self.bridge, memory, self.config["login"], self.account_data["mode"], self.config["server"])
            finally:
                memory.close()
        return self.launch_job("reconcile", work)

    def save_calendar(self, value):
        path = self.state_dir / "calendar-check.json"
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        try:
            NewsGate(str(path)).check(time.time(), 0)
        except Blocked as e:
            # A valid calendar can correctly block an active news window.
            if str(e) != "High-impact USD news blackout":
                raise
        target = self.resolve(self.config["calendar"], "data", must_exist=False)
        # A manual upload owns the selected calendar until auto is re-enabled.
        from .news import atomic_json
        self.config["calendar_auto"] = False
        atomic_json(self.state_dir / "workstation.json", self.config)
        self.calendar_wake.set()
        target.parent.mkdir(parents=True, exist_ok=True)
        path.replace(target)
        return {"saved": self.config["calendar"]}

    def stop(self, flatten=False):
        if flatten and self.worker and self.worker.is_alive():
            # Risk exits are executed by the existing worker and the existing authorization.
            (self.root / "STOP").write_text("operator requested flatten", encoding="utf-8")
            self.entries_enabled = False
            self.note("Flatten requested; awaiting broker confirmation")
            return {"status": "flatten_requested"}
        self.worker_stop.set()
        self.entries_enabled = False
        if self.worker and self.worker.is_alive():
            self.worker_state.update(phase="stopping", reason=None)
        return {"status": "stopping", "positions_closed": False}

    def journal(self):
        memory = Memory(self.db_path)
        try:
            return memory.recent_events(100)
        finally:
            memory.close()

    def sync(self):
        def work(update):
            memory = Memory(self.db_path)
            try:
                sink = SupabaseSink(os.environ.get("FXB_SUPABASE_URL", ""), os.environ.get("FXB_SUPABASE_SECRET_KEY", ""))
                total = 0
                while count := memory.sync(sink):
                    total += count
                    update({"stage": "sync", "events": total})
                return {"events": total}
            finally:
                memory.close()
        return self.launch_job("memory_sync", work)
