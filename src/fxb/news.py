"""Hourly public calendar import. Invalid downloads never replace valid data."""
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
from .data import iso, timestamp

SOURCE = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
REFRESH_SECONDS = 3600
MAX_BYTES = 2_000_000


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".fxb-", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(document, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def week_bounds(now):
    local = datetime.fromtimestamp(now, ZoneInfo("America/New_York"))
    start = local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=(local.weekday()+1) % 7)
    return start.timestamp(), (start + timedelta(days=7)).timestamp()


def parse_weekly(rows, now, modified):
    start, end = week_bounds(now)
    if not -5 <= now - modified <= 86400:
        raise ValueError("Calendar source is stale or its timestamp is in the future")
    if not isinstance(rows, list) or not rows or len(rows) > 10000:
        raise ValueError("Calendar feed must contain a nonempty event list")
    events, seen = [], {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid calendar event")
        title, currency, impact = row.get("title"), row.get("country"), row.get("impact")
        if (not isinstance(title, str) or not title.strip() or len(title) > 500
                or not isinstance(currency, str) or len(currency) != 3 or not currency.isalpha()
                or not isinstance(row.get("date"), str)
                or impact not in ("High", "Medium", "Low", "Non-Economic")):
            raise ValueError("Unknown calendar event schema or impact; refresh was not applied")
        at = timestamp(row["date"])
        if not start <= at < end:
            raise ValueError("Calendar feed is not for the current week")
        impact = "low" if impact == "Non-Economic" else impact.lower()
        key = (at, currency.upper(), title.strip())
        if key in seen:
            if seen[key] != impact:
                raise ValueError("Conflicting duplicate calendar events")
            continue
        seen[key] = impact
        events.append({"time": iso(at), "currency": currency.upper(), "impact": impact, "title": title.strip()})
    if not any(e["currency"] == "USD" for e in events):
        raise ValueError("Calendar feed contains no USD coverage")
    return {"source_url": SOURCE, "provider": "Forex Factory weekly export",
            "coverage_start": iso(start), "coverage_end": iso(end),
            "updated_at": iso(min(now, modified)), "fetched_at": iso(now),
            "events": sorted(events, key=lambda e: (e["time"], e["currency"], e["title"]))}


def load_weekly(path, now=None):
    """Return validated cached data or download once; caller commits atomically."""
    now = time.time() if now is None else now
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if (doc.get("source_url") == SOURCE and 0 <= now-timestamp(doc["fetched_at"]) < REFRESH_SECONDS
                and 0 <= now-timestamp(doc["updated_at"]) < 86400
                and timestamp(doc["coverage_start"]) <= now < timestamp(doc["coverage_end"])
                and isinstance(doc.get("events"), list) and doc["events"]):
            return doc, True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    request = Request(SOURCE, headers={"User-Agent": "FXB/0.3.2 (+https://github.com/sjdpluse/fxb)",
                                      "Accept": "application/json"})
    with urlopen(request, timeout=15) as response:
        if response.status != 200 or "json" not in response.headers.get("Content-Type", "").lower():
            raise ValueError("Calendar provider did not return JSON; existing calendar was preserved")
        modified_text = response.headers.get("Last-Modified")
        if not modified_text:
            raise ValueError("Calendar response has no source update timestamp")
        modified = parsedate_to_datetime(modified_text)
        if modified.tzinfo is None:
            raise ValueError("Calendar source timestamp has no time zone")
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Calendar response is too large")
    # Read a new clock after the request; downloads do not extend source age.
    return parse_weekly(json.loads(payload), time.time(), modified.timestamp()), False


def refresh_calendar(path):
    doc, cached = load_weekly(path)
    if not cached:
        atomic_json(path, doc)
    return {"status": "cached" if cached else "updated", "source": SOURCE,
            "updated_at": doc["updated_at"], "coverage_end": doc["coverage_end"],
            "events": len(doc["events"]), "orders_sent": 0}
