"""Workstation, full connectome, training, market data and controlled execution."""
import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from .data import load_bars, save_bars, synthetic_bars, FEATURE_NAMES
from .brain import import_connectome
from .research import train, evaluate_file, walk_forward
from .memory import Memory, SupabaseSink
from .mt5 import MT5, Blocked, WorkerLock, cycle, reconcile
from .clock import BrokerClock


def parser():
    p = argparse.ArgumentParser(description="FXB neural trading workstation")
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("serve", help="Open the local trading workstation")
    c.add_argument("--port", type=int, default=8765)
    c.add_argument("--root", default=".")
    c.add_argument("--no-browser", action="store_true")
    c = commands.add_parser("brain-install", help="Download the pinned full published connectome and coordinates")
    c.add_argument("--directory", default="data/connectome")
    c = commands.add_parser("clock-check", help="Read-only UTC/MT5 timestamp check using the saved workstation account")
    c.add_argument("--root", default=".")
    c.add_argument("--symbols", nargs="+", required=True)
    c = commands.add_parser("clock-set", help="Save a broker clock rule; no date expiry unless both dates are supplied. Sends no orders")
    c.add_argument("--root", default=".")
    c.add_argument("--zone", required=True, help="Explicit fixed offset, UTC, or a confirmed IANA/NY+7 rule")
    c.add_argument("--valid-from")
    c.add_argument("--valid-until")
    c = commands.add_parser("scalp-prepare", help="Prepare the gold M1 online profile from 500 recent closed bars; sends no orders")
    c.add_argument("--root", default=".")
    c.add_argument("--symbol", default="XAUUSD.m")
    c.add_argument("--commission", type=float, required=True)
    c.add_argument("--slippage-points", type=float, required=True)
    c.add_argument("--overnight-cost", type=float, required=True)
    c = commands.add_parser("synth", help="Generate clearly labelled synthetic test bars")
    c.add_argument("--out", required=True)
    c.add_argument("--bars", type=int, default=2400)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--symbol", choices=["XAUUSD", "US100"], default="XAUUSD")
    for name in ("train", "walk-forward"):
        c = commands.add_parser(name)
        c.add_argument("--data", required=True)
        c.add_argument("--out", required=True)
        c.add_argument("--epochs", type=int, default=6)
        c.add_argument("--seed", type=int, default=7)
        c.add_argument("--initial-balance", type=float, default=10_000,
                       help="Starting account-currency balance for each independent partition")
        c.add_argument("--encoding-cache", help="Optional directory for resumable compiled-brain encoding")
        g = c.add_mutually_exclusive_group()
        g.add_argument("--graph")
        g.add_argument("--plain", action="store_true", help="Train the same DQN without recurrence as a control")
        g.add_argument("--full-brain", action="store_true", help="Use all 138,639 neurons of the published Brian2 model")
        g.add_argument("--fast-brain", action="store_true", help="Compiled full Shiu LIF graph; resumes cached neural encoding")
        if name == "walk-forward":
            c.add_argument("--folds", type=int, default=3)
    c = commands.add_parser("evaluate")
    c.add_argument("--data", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--forward", action="store_true")
    c = commands.add_parser("import-connectome")
    c.add_argument("--edges", required=True)
    c.add_argument("--provenance", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--seed", type=int, default=7)
    c = commands.add_parser("export-mt5", help="Read closed M5 bars and exact contract metadata")
    c.add_argument("--symbol", required=True, help="Exact name from this broker's Market Watch")
    c.add_argument("--out", required=True)
    c.add_argument("--bars", type=int, default=50000)
    c.add_argument("--terminal")
    c.add_argument("--commission", type=float, required=True, help="Account-currency commission per lot per side")
    c.add_argument("--slippage-points", type=float, required=True)
    c.add_argument("--overnight-cost", type=float, required=True, help="Conservative account-currency overnight cost per lot")
    c = commands.add_parser("demo", help="Dry-run by default; --execute-demo can ONLY send to DEMO accounts")
    c.add_argument("--model", action="append", required=True, help="Repeat for gold and US100; one worker manages the account")
    c.add_argument("--login", type=int, required=True)
    c.add_argument("--terminal")
    c.add_argument("--db", default="state/demo.sqlite")
    c.add_argument("--calendar")
    c.add_argument("--kill-file", default="STOP")
    c.add_argument("--execute-demo", action="store_true")
    c.add_argument("--once", action="store_true")
    c = commands.add_parser("reconcile", help="Match uncertain submissions to broker deal history; never resubmit")
    c.add_argument("--login", type=int, required=True)
    c.add_argument("--terminal")
    c.add_argument("--db", default="state/demo.sqlite")
    c.add_argument("--mode", choices=["demo", "live"], default="demo")
    c.add_argument("--server")
    c = commands.add_parser("memory", help="Inspect the local journal or sync its pending events to Supabase")
    c.add_argument("--db", required=True)
    c.add_argument("--sync", action="store_true")
    for name in ("export-mt5", "demo", "reconcile"):
        c = commands.choices[name]
        c.add_argument("--broker-zone", default="UTC", help="UTC, an explicit offset such as +02:00, a confirmed IANA zone, or NY+7")
        c.add_argument("--clock-valid-from", help="UTC start of the configured clock interval, inclusive")
        c.add_argument("--clock-valid-until", help="UTC end of the configured clock interval, exclusive")
    return p


def run(args):
    if args.command == "clock-set":
        clock = BrokerClock(args.zone, args.valid_from, args.valid_until)
        state = Path(args.root).resolve() / "state"
        path = state / "workstation.json"
        with WorkerLock(state / "runtime.sqlite"):
            original = path.read_bytes()
            config = json.loads(original.decode("utf-8-sig"))
            if not isinstance(config, dict) or not config.get("login") or not config.get("server"):
                raise ValueError("Save the exact MT5 account and server in the workstation first")
            config["broker_clock"] = clock.config()
            backup = state / f"workstation.clock-backup-{uuid.uuid4().hex}.json"
            backup.write_bytes(original)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=state,
                                                 prefix=".clock-", suffix=".tmp", delete=False) as f:
                    temporary = Path(f.name)
                    json.dump(config, f, indent=2, allow_nan=False)
                    f.flush()
                    os.fsync(f.fileno())
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return {"status": "saved", "broker_clock": clock.config(), "backup": str(backup),
                "orders_sent": 0, "restart_workstation": True}
    if args.command == "scalp-prepare":
        from .workstation import Workstation
        from .scalper_runtime import create_profile
        controller = Workstation(args.root)
        try:
            with WorkerLock(controller.db_path):
                controller.connect()
                output = controller.root / "runs" / f"gold-M1-scalper-{time.time_ns()}"
                result = create_profile(controller.bridge, args.symbol, output, args.commission,
                                        args.slippage_points, args.overnight_cost)
                result["model"] = str(Path(result["model"]).relative_to(controller.root))
                return result
        finally:
            if controller.bridge: controller.bridge.close()
    if args.command == "clock-check":
        config = json.loads((Path(args.root) / "state/workstation.json").read_text(encoding="utf-8-sig"))
        if not config.get("login") or not config.get("server"):
            raise ValueError("Save the exact MT5 account and server in the workstation first")
        bridge = MT5(terminal=config.get("terminal") or None, clock=config.get("broker_clock"))
        try:
            bridge.connect()
            bridge.account(config["login"], expected_server=config["server"])
            quotes = [bridge.clock_snapshot(symbol) for symbol in args.symbols]
            return {"ready": all(q["ready"] for q in quotes), "quotes": quotes, "orders_sent": 0}
        finally:
            bridge.close()
    if args.command == "serve":
        from .ui import serve
        serve(args.root, args.port, open_browser=not args.no_browser)
        return {"status": "stopped"}
    if args.command == "brain-install":
        from .connectome import install
        return install(args.directory, lambda p: print(json.dumps(p), flush=True))
    if args.command == "synth":
        save_bars(synthetic_bars(args.bars, args.seed, args.symbol), args.out)
        return {"data": args.out, "synthetic": True, "purpose": "software verification only"}
    if args.command in ("train", "walk-forward"):
        def progress(event):
            if event["stage"] == "brain_encoding":
                print(json.dumps({k: v for k, v in event.items() if k != "activity"}), flush=True)
        kw = dict(epochs=args.epochs, seed=args.seed, graph=args.graph, plain=args.plain,
                  full_brain=args.full_brain, initial=args.initial_balance, progress=progress,
                  fast_brain=args.fast_brain, encoding_cache=args.encoding_cache)
        if args.command == "walk-forward":
            r = walk_forward(load_bars(args.data), args.out, folds=args.folds, **kw)
            return r
        r = train(load_bars(args.data), args.out, **kw)
        return {"output": args.out, "holdout": r["holdout"], "profitability_established": False}
    if args.command == "evaluate":
        return evaluate_file(load_bars(args.data), args.model, args.forward)
    if args.command == "import-connectome":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        return import_connectome(args.edges, args.provenance, args.out, len(FEATURE_NAMES), args.seed)
    if args.command == "export-mt5":
        bridge = MT5(terminal=args.terminal, clock=BrokerClock(args.broker_zone, args.clock_valid_from, args.clock_valid_until))
        try:
            bridge.connect()
            spec = bridge.spec(args.symbol, args.commission, args.slippage_points, args.overnight_cost)
            bars = bridge.bars(spec, args.bars)
            save_bars(bars, args.out)
            return {"data": args.out, "bars": len(bars.time), "synthetic": False}
        finally:
            bridge.close()
    if args.command in ("demo", "reconcile"):
        with WorkerLock(args.db):
            bridge = MT5(terminal=args.terminal, clock=BrokerClock(args.broker_zone, args.clock_valid_from, args.clock_valid_until))
            memory = Memory(args.db)
            try:
                bridge.connect()
                if args.command == "reconcile":
                    return reconcile(bridge, memory, args.login, args.mode, args.server)
                # One policy per symbol prevents two checkpoints contending for one position.
                from .research import load_checkpoint
                models = [load_checkpoint(path)[3] for path in args.model]
                symbols = [m["spec"]["symbol"] for m in models]
                if len(set(symbols)) != len(symbols):
                    raise ValueError("Use one model per exact broker symbol")
                if any(m["risk"] != models[0]["risk"] for m in models):
                    raise ValueError("Models sharing an account must use the same independent risk limits")
                while True:
                    for model in args.model:
                        result = cycle(bridge, model, memory, args.login, args.calendar,
                                       args.execute_demo, args.kill_file)
                        print(json.dumps(result, allow_nan=False), flush=True)
                    if args.once:
                        return {"status": "cycle_complete", "execution": args.execute_demo}
                    time.sleep(2)
            finally:
                memory.close()
                bridge.close()
    if args.command == "memory":
        memory = Memory(args.db)
        try:
            synced = 0
            if args.sync:
                sink = SupabaseSink(os.environ.get("FXB_SUPABASE_URL", ""), os.environ.get("FXB_SUPABASE_SECRET_KEY", ""))
                while True:
                    n = memory.sync(sink)
                    synced += n
                    if not n:
                        break
            return memory.stats() | {"synced_this_run": synced}
        finally:
            memory.close()
    raise ValueError("Unknown command")


def main():
    args = parser().parse_args()
    try:
        print(json.dumps(run(args), indent=2, allow_nan=False))
    except KeyboardInterrupt:
        print("Stopped. Inspect MT5 positions; attached broker SL/TP remain active.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, KeyError) as e:
        print(f"FXB blocked: {e}", file=sys.stderr)
        raise SystemExit(2)
