"""Local workstation controller: authentic telemetry, background jobs, trading control.

No account credentials are accepted or stored; MT5 uses its local terminal session.
The HTTP layer cannot submit individual orders. It can activate the controlled worker.
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
from .data import load_bars, save_bars
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
        self.bridge = None
        self.brain = None
        self.worker = None
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
        self.config = {"login": None, "server": "", "terminal": "", "calendar": "data/news.json",
                       "broker_clock": BrokerClock().config()}
        path = self.state_dir / "workstation.json"
        if path.exists():
            self.config.update(json.loads(path.read_text(encoding="utf-8")))

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
                               "synthetic": meta["synthetic"], "source": meta["source"]})
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
        memory = Memory(self.db_path)
        try:
            journal = memory.stats()
        finally:
            memory.close()
        try:
            NewsGate(str(self.root / self.config["calendar"])).check(time.time(), 3600)
            news = {"ready": True, "reason": "Calendar covers the next hour"}
        except (Blocked, OSError, ValueError) as e:
            news = {"ready": False, "reason": str(e)}
        return {"version": __version__, "preview": self.preview, "platform": os.name, "news": news,
                "running": bool(self.worker and self.worker.is_alive()), "entries_enabled": self.entries_enabled,
                "mode": self.mode, "account": self.account_data, "positions": self.positions, "quotes": self.quotes,
                "models": self.models(), "datasets": self.datasets(), "active_models": self.active_models,
                "brain": {"manifest": manifest, "activity": activity}, "job": dict(self.job),
                "journal": journal, "decisions": self.last_decisions[-20:], "events": self.events[-25:],
                "error": self.last_error, "live_activation": self.authorization.public_status() if self.authorization else None,
                "config": self.config, "connected": self.bridge is not None}

    def launch_job(self, kind, function):
        with self.lock:
            if self.job["status"] == "running":
                raise ValueError("Another background operation is still running")
            if self.worker and self.worker.is_alive() and kind in ("brain_install", "brain_stimulate", "train"):
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

    def start(self, model_paths, mode="dry-run", acknowledged_login=None, hours=8):
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
        self.authorization = None
        if mode == "live":
            self.authorization = LiveAuthorization.activate(self.config["login"], self.config["server"],
                [sha256(p) for p in paths], metas[0]["risk"], acknowledged_login, hours)
        self.mode, self.entries_enabled, self.active_models = mode, True, [str(p.relative_to(self.root)) for p in paths]
        self.last_error = None
        self.worker_stop.clear()
        self.brain = None
        gc.collect()
        (self.root / "STOP").unlink(missing_ok=True)
        def run():
            memory = Memory(self.db_path)
            try:
                with WorkerLock(self.db_path):
                    self.note("Trading worker started", mode=mode)
                    blocked_reasons = {}
                    while not self.worker_stop.is_set():
                        for path in paths:
                            try:
                                result = cycle(self.bridge, path, memory, self.config["login"],
                                    calendar=str(self.root / self.config["calendar"]), execute=mode != "dry-run",
                                    kill_file=str(self.root / "STOP"), mode=broker_mode,
                                    expected_server=self.config["server"], authorization=self.authorization,
                                    entries_enabled=self.entries_enabled)
                                blocked_reasons.pop(str(path), None)
                                self.last_error = None
                            except Blocked as e:
                                self.last_error = str(e)
                                if memory.unresolved():
                                    self.entries_enabled = False
                                if blocked_reasons.get(str(path)) == str(e):
                                    continue
                                blocked_reasons[str(path)] = str(e)
                                self.note(str(e), "warning", operation="execution_wait")
                                result = {"action": "WAIT", "status": "blocked", "reason": str(e)}
                            if result.get("reason") != "already_processed_bar":
                                self.last_decisions.append(result | {"time": time.time(), "model": path.parent.name})
                                self.last_decisions = self.last_decisions[-100:]
                        try:
                            self.refresh_account()
                        except Blocked as e:
                            self.last_error = str(e)
                        self.worker_stop.wait(2)
            except Exception as e:
                self.last_error = str(e)
                self.entries_enabled = False
                self.note(str(e), "error", operation="execution")
            finally:
                memory.close()
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
