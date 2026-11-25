"""Bounded local snapshots of rejected structural entries and read-only replay.

No MT5 connection, order submission, learning update or cloud outbox is used.
Only the first rejection per account/profile/bar/side/reason is retained, with
at most twelve reports. The report is a recorded decision, never a live quote.
"""
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid
import numpy as np
from . import __version__
from .data import Bars, KABUL, iso
from .market_plan import structure_plan
from .micro_scalping import MicroConfig, assess_micro_signal, quote_pressure
from .mt5 import Blocked
from .risk import Spec
from .scalping import ScalpConfig, analyze

SCHEMA = "fxb-entry-rejection-v1"
INDEX = "entry-diagnostics-index"
PREFIX = "entry-diagnostic:"
MAX_REPORTS = 12
COLUMNS = ("time", "open", "high", "low", "close", "volume", "spread")
SNAPSHOT_SOURCE = "captured M1 decision context"
PUBLIC_FIELDS = ("id", "captured_at", "captured_utc", "captured_kabul", "symbol", "side",
                 "reason", "account_mode", "model_sha256")


def reports(memory):
    return [{k: row[k] for k in PUBLIC_FIELDS} for row in memory.get_state(INDEX, [])]


def store_snapshot(memory, report, index, sampling_key):
    row = {k: report[k] for k in PUBLIC_FIELDS} | {"sampling_key": sampling_key}
    updated = [row] + index[:MAX_REPORTS-1]
    encoded = json.dumps(report, allow_nan=False, separators=(",", ":"))
    with memory.db:
        memory.db.execute("INSERT INTO state VALUES(?,?)", (PREFIX+report["id"], encoded))
        memory.db.execute("INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                          (INDEX, json.dumps(updated, allow_nan=False)))
        memory.db.executemany("DELETE FROM state WHERE key=?",
                             [(PREFIX+r["id"],) for r in index[MAX_REPORTS-1:]])
    return {k: row[k] for k in PUBLIC_FIELDS}


def capture_rejection(bridge, memory, context, meta, digest, account_scope, mode, decision, history):
    """Called only after momentum passes and the structure planner rejects.

    A diagnostic write failure must not change that rejection or stop position
    management. Credentials, account identifiers, balances and weights are not
    part of the export. Account scope is used only in the internal sample key.
    """
    try:
        momentum = decision["details"]["momentum"]
        if not momentum["ready"] or decision["reason"] not in (
                "no_structure_setup", "structure_stop_distance", "no_structure_target"):
            return None
        sampling_key = hashlib.sha256(json.dumps([account_scope, digest, context["closed_at"],
            momentum["side"], decision["reason"]]).encode()).hexdigest()
        index = memory.get_state(INDEX, [])
        existing = next((r for r in index if r["sampling_key"] == sampling_key), None)
        if existing:
            return {k: existing[k] for k in PUBLIC_FIELDS}
        bars, now = context["bars"], time.time()
        report = {"schema": SCHEMA, "id": str(uuid.uuid4()), "fxb_version": __version__,
            "captured_at": now, "captured_utc": iso(now),
            "captured_kabul": datetime.fromtimestamp(now, KABUL).isoformat(),
            "symbol": bars.spec.symbol, "side": momentum["side"], "account_mode": mode,
            "model_sha256": digest, "reason": decision["reason"], "orders_sent": 0,
            "scope": "momentum_and_structure_rejection", "decision": decision,
            "profile": {"scalp_config": meta["scalp_config"], "micro_config": meta["micro_config"]},
            "market": {"spec": asdict(bars.spec), "columns": list(COLUMNS),
                "rows": np.column_stack([getattr(bars, k) for k in COLUMNS]).tolist(),
                "price_basis": "bid", "spread_unit": "price", "synthetic": bars.synthetic,
                "time_basis": bars.time_basis, "bars_sha256": replace(bars, source=SNAPSHOT_SOURCE,
                    rollover=None, assumptions=None).digest()},
            "tick_columns": ["time", "bid", "ask"], "ticks": [list(row) for row in history],
            "structure": context["structure"], "closed_bar_signal": context.get("signal"),
            "atr": context["atr"]}
        result = store_snapshot(memory, report, index, sampling_key)
        bridge.entry_diagnostic_error = None
        return result
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as error:
        bridge.entry_diagnostic_error = f"{type(error).__name__}: entry diagnostic could not be saved"
        return None


def load_report(memory, report_id=None):
    index = memory.get_state(INDEX, [])
    if report_id is None:
        if not index:
            raise FileNotFoundError("No recorded structural rejection yet; wait for a Micro entry assessment")
        report_id = index[0]["id"]
    if str(uuid.UUID(report_id)) != report_id:
        raise ValueError("Invalid diagnostic report identifier")
    if not any(row["id"] == report_id for row in index):
        raise FileNotFoundError("This report has expired from the twelve-report history; select a current report")
    report = memory.get_state(PREFIX+report_id)
    if report is None:
        raise FileNotFoundError("Diagnostic report is unavailable")
    return report


def replay_report(report):
    """Recompute the two gates with captured data, without trading authority.

    Matching means reproducible logic, not that rejecting the trade was optimal.
    Account/broker gates and the trained entry readout are outside this replay.
    """
    if report.get("schema") != SCHEMA:
        raise ValueError("Unsupported diagnostic schema")
    market = report["market"]
    if market.get("price_basis") != "bid" or market.get("spread_unit") != "price":
        raise ValueError("Diagnostic prices must use bid bars and price-distance spreads")
    if market["columns"] != list(COLUMNS) or len(market["rows"]) != 500:
        raise ValueError("Expected exactly 500 captured M1 bars")
    matrix = np.asarray(market["rows"], dtype=float)
    if matrix.shape != (500, len(COLUMNS)):
        raise ValueError("Invalid diagnostic bar columns")
    spec = Spec(**market["spec"])
    bars = Bars(**{k: matrix[:, i] for i, k in enumerate(COLUMNS)}, spec=spec,
                source=SNAPSHOT_SOURCE, synthetic=market["synthetic"],
                time_basis=market["time_basis"]).validate()
    if spec.timeframe_seconds != 60 or bars.digest() != market["bars_sha256"]:
        raise ValueError("Captured bar integrity check failed")
    cfg, micro = ScalpConfig(**report["profile"]["scalp_config"]), MicroConfig(**report["profile"]["micro_config"])
    _, atr, signals, structures = analyze(bars, cfg)
    ticks = report["ticks"]
    if report["tick_columns"] != ["time", "bid", "ask"] or not 4 <= len(ticks) <= 64:
        raise ValueError("Expected the recorded quote window")
    from types import SimpleNamespace
    history, pressure = [], None
    for row in ticks:
        if len(row) != 3 or not all(math.isfinite(v) for v in row):
            raise ValueError("Invalid captured quote")
        t, bid, ask = row
        pressure = quote_pressure(history, SimpleNamespace(time_msc=t*1000, bid=bid, ask=ask), t)
    cost = 2*(spec.commission_per_lot_side*spec.tick_size/spec.tick_value + math.ceil(spec.slippage_points)*spec.point)
    signal, momentum = assess_micro_signal(pressure, float(atr[-1]), spec.tick_size, cost, micro.direction_policy)
    plan, assessment, reason = None, None, momentum["reason"]
    if signal:
        try:
            plan = structure_plan(bars, structures[-1], signal["side"], pressure["bid"],
                                  float(atr[-1]), spec.tick_size, signals[-1])
            assessment, reason = plan["setup_assessment"], "structure_plan_ready"
        except Blocked as error:
            assessment, reason = (error.details or {}).get("setup_assessment"), error.code
    return {"report_id": report["id"], "captured_reason": report["reason"], "replayed_reason": reason,
            "reason_matches": reason == report["reason"], "code_version_matches": report["fxb_version"] == __version__,
            "scope": "momentum_and_structure_only", "momentum": momentum,
            "setup_assessment": assessment, "plan": plan, "orders_sent": 0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reproduce a recorded entry rejection without connecting to MT5")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    path = Path(args.report)
    if path.stat().st_size > 2_000_000:
        parser.error("Diagnostic report exceeds 2 MB")
    result = replay_report(json.loads(path.read_text(encoding="utf-8-sig")))
    print(json.dumps(result, indent=2, allow_nan=False))
