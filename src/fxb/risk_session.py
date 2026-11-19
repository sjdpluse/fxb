"""Explicit, audited demo-account risk periods; no broker orders or auto reset."""
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import math
import time
from .data import KABUL, day_key
from .mt5 import Blocked
from .risk import Guard


def account_scope(login, server):
    return hashlib.sha256(f"{server}:{login}".encode()).hexdigest()[:20]


def demo_snapshot(bridge, memory, login, server):
    """Caller holds WorkerLock; validate the actual terminal, never the UI mode."""
    if not login or not server:
        raise Blocked("Connect the configured demo account before starting a new risk period")
    account = bridge.account(login, execution_mode="demo", expected_server=server)
    positions, pending = bridge.positions()
    if positions or pending:
        raise Blocked("Close positions and cancel pending orders before a new demo risk period",
                      code="demo_session_exposure")
    if memory.unresolved():
        raise Blocked("Reconcile uncertain orders before a new demo risk period",
                      code="demo_session_unresolved")
    scope = account_scope(account.login, account.server)
    if memory.get_state("active-basket:"+scope):
        raise Blocked("Finish basket management and settlement before a new demo risk period",
                      code="demo_session_basket")
    if any(not math.isfinite(v) or v <= 0 for v in (account.equity, account.balance)):
        raise Blocked("Demo balance and equity must be positive and finite")
    now = time.time()
    day = day_key(now)
    midnight = datetime.fromtimestamp(now, KABUL).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    deals = bridge.history_deals(midnight, now+1)
    # Balance/credit records are not entries. Preserve the larger current-day
    # counter so restarting a demo period cannot manufacture additional quota.
    entries = len({d.order for d in deals if d.time >= midnight and d.entry in (0, 2)
                   and d.type in (0, 1) and d.order > 0})
    previous = memory.get_state("guard:"+scope, {})
    old = Guard(**previous)
    entries = max(entries, old.entries if old.day == day else 0)
    proposed = Guard(day=day, day_start=account.equity, peak=account.equity, entries=entries)
    # Re-read the account after history retrieval; switching/resetting the
    # terminal during the review must invalidate the snapshot.
    final = bridge.account(login, execution_mode="demo", expected_server=server)
    if (final.currency, final.equity, final.balance) != (account.currency, account.equity, account.balance):
        raise Blocked("Demo account values changed during review; request a fresh preview",
                      code="demo_session_changed")
    positions, pending = bridge.positions()
    if positions or pending:
        raise Blocked("Account exposure changed during review", code="demo_session_exposure")
    return {"account_scope": scope, "login": account.login, "server": account.server,
            "currency": account.currency, "mode": "demo", "balance": account.balance,
            "equity": account.equity, "day": day, "previous": previous,
            "proposed": asdict(proposed)}


def snapshot_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def receipt(memory, token, login, server):
    row = memory.db.execute("SELECT kind,payload FROM events WHERE id=?", (token,)).fetchone()
    if row is None:
        return None
    value = json.loads(row["payload"])
    if row["kind"] != "demo_risk_session" or (value["login"], value["server"]) != (login, server):
        raise Blocked("Risk-period request belongs to another account")
    return value


def commit_demo_session(memory, snapshot, token):
    """Publish the new guard and its immutable audit event in one transaction."""
    now = time.time()
    result = snapshot | {"status": "restarted", "request_id": token, "completed_at": now,
                         "reason": "operator_started_new_demo_period", "orders_sent": 0}
    key = "guard:"+snapshot["account_scope"]
    encoded = json.dumps(result, allow_nan=False)
    with memory.db:
        memory.db.execute("BEGIN IMMEDIATE")
        if memory.get_state(key, {}) != snapshot["previous"]:
            raise Blocked("Risk state changed after review; request a fresh preview",
                          code="demo_session_changed")
        memory.db.execute("INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                          (key, json.dumps(snapshot["proposed"], allow_nan=False)))
        memory.db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,0)",
                          (token, "account:"+snapshot["account_scope"], "ACCOUNT", "demo_risk_session", now, now, encoded))
    return result
