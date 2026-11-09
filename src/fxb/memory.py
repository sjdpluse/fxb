"""Durable experience journal, causal recall, and retryable Supabase outbox."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid
import urllib.request
import urllib.error
from urllib.parse import urlparse
import numpy as np


class Memory:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, symbol TEXT NOT NULL,
            kind TEXT NOT NULL, occurred_at REAL NOT NULL, available_at REAL NOT NULL,
            payload TEXT NOT NULL, synced INTEGER NOT NULL DEFAULT 0);
          CREATE INDEX IF NOT EXISTS events_recall ON events(run_id,symbol,kind,available_at);
          CREATE TABLE IF NOT EXISTS orders (
            key TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
        """)

    def close(self):
        self.db.close()

    def record(self, run_id, symbol, kind, occurred_at, available_at, payload, event_id=None):
        if (not np.isfinite([occurred_at, available_at]).all() or available_at < occurred_at):
            raise ValueError("An outcome cannot be available before the observation")
        key = event_id or str(uuid.uuid4())
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?,0)",
                            (key, run_id, symbol, kind, occurred_at, available_at, encoded))
        return key

    def recall(self, state, symbol, as_of, run_id, limit=5):
        """Explicit run + time boundary prevents cross-run and future-label leakage."""
        query = np.asarray(state, dtype=float)
        if not np.isfinite(query).all() or not np.isfinite(as_of) or limit < 1:
            raise ValueError("Invalid recall query")
        rows = self.db.execute("""SELECT * FROM events WHERE run_id=? AND symbol=?
            AND kind='experience' AND available_at < ? ORDER BY available_at DESC LIMIT 10000""",
            (run_id, symbol, as_of)).fetchall()
        ranked = []
        for row in rows:
            p = json.loads(row["payload"])
            candidate = np.asarray(p["state"], dtype=float)
            if candidate.shape == query.shape:
                distance = float(np.linalg.norm(query - candidate))
                ranked.append({"id": row["id"], "distance": distance, "available_at": row["available_at"], **p})
        return sorted(ranked, key=lambda p: p["distance"])[:limit]

    def get_state(self, key, default=None):
        row = self.db.execute("SELECT payload FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put_state(self, key, payload):
        with self.db:
            self.db.execute("INSERT INTO state VALUES(?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
                            (key, json.dumps(payload, allow_nan=False)))

    def claim_order(self, key, payload):
        """Commit before contacting MT5; an uncertain send is never blindly retried."""
        try:
            with self.db:
                self.db.execute("INSERT INTO orders VALUES(?,?,?)", (key, "reserved", json.dumps(payload, allow_nan=False)))
            return True
        except sqlite3.IntegrityError:
            return False

    def order_status(self, key, status, payload):
        if status not in ("sent", "rejected", "ambiguous"):
            raise ValueError("Invalid order status")
        with self.db:
            self.db.execute("UPDATE orders SET status=?,payload=? WHERE key=?",
                            (status, json.dumps(payload, allow_nan=False), key))

    def unresolved(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM orders WHERE status IN ('reserved','ambiguous')")]

    def stats(self):
        return {"events": self.db.execute("SELECT count(*) FROM events").fetchone()[0],
                "pending_sync": self.db.execute("SELECT count(*) FROM events WHERE synced=0").fetchone()[0],
                "unresolved_orders": len(self.unresolved())}

    def sync(self, transport, batch=100):
        rows = self.db.execute("SELECT * FROM events WHERE synced=0 ORDER BY rowid LIMIT ?", (batch,)).fetchall()
        if not rows:
            return 0
        payload = [{k: row[k] for k in ("id", "run_id", "symbol", "kind")} | {
            "occurred_at": datetime.fromtimestamp(row["occurred_at"], timezone.utc).isoformat(),
            "available_at": datetime.fromtimestamp(row["available_at"], timezone.utc).isoformat(),
            "payload": json.loads(row["payload"])} for row in rows]
        transport(payload)  # Errors leave every row pending; UUID upserts make retries safe.
        with self.db:
            self.db.executemany("UPDATE events SET synced=1 WHERE id=?", [(r["id"],) for r in rows])
        return len(rows)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Supabase redirects are refused to protect backend credentials")


class SupabaseSink:
    def __init__(self, url, secret):
        parsed = urlparse(url)
        if (parsed.scheme != "https" or not (parsed.hostname or "").endswith(".supabase.co")
                or parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment
                or parsed.path not in ("", "/")):
            raise ValueError("Use the HTTPS project URL at <ref>.supabase.co")
        if not secret.startswith("sb_secret_"):
            raise ValueError("Set a backend sb_secret_ API key; publishable/anon keys are not supported")
        self.url = url.rstrip("/") + "/rest/v1/fxb_events?on_conflict=id"
        self.secret = secret

    def __call__(self, payload):
        request = urllib.request.Request(self.url, method="POST",
            data=json.dumps(payload, allow_nan=False).encode(), headers={
                "apikey": self.secret, "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal"})
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=15) as response:
                if not 200 <= response.status < 300:
                    raise RuntimeError("Supabase rejected the batch")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Supabase HTTP {e.code}; batch remains pending") from None
        except urllib.error.URLError:
            raise RuntimeError("Supabase connection failed; batch remains pending") from None
