"""Calendar adapter tests use explicit software fixtures, never invented market news."""
from datetime import datetime, timezone
from email.utils import format_datetime
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from fxb.clock import instant
from fxb.cli import parser, run
from fxb.data import iso
from fxb.mt5 import NewsGate, Blocked
from fxb.news import parse_weekly, week_bounds, load_weekly, refresh_calendar, atomic_json, SOURCE
from fxb.workstation import Workstation


NOW = instant("2026-11-16T12:00:00Z")
ROWS = [{"title": "Software test event", "country": "USD", "impact": "High",
         "date": "2026-11-16T07:10:00-05:00"}]


class Response(io.BytesIO):
    def __init__(self, rows=ROWS, modified=NOW-60, content_type="application/json"):
        super().__init__(json.dumps(rows).encode())
        self.status = 200
        self.headers = {"Content-Type": content_type,
                        "Last-Modified": format_datetime(datetime.fromtimestamp(modified, timezone.utc), usegmt=True)}


class CalendarTests(unittest.TestCase):
    def test_holiday_rows_import_without_suppressing_high_impact_events(self):
        rows=ROWS+[{"title":"Software holiday fixture", "country":currency,
                    "date":"2026-11-16T07:30:00-05:00", "impact":"Holiday"}
                   for currency in ("JPY","USD")]
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"news.json"
            with patch("fxb.news.urlopen",return_value=Response(rows=rows)), \
                 patch("time.time",return_value=NOW):
                result=refresh_calendar(path)
            self.assertEqual(result["events"],3)
            doc=json.loads(path.read_text())
            holidays=[e for e in doc['events'] if e.get('source_impact')=='Holiday']
            self.assertEqual(len(holidays),2)
            self.assertTrue(all(e['impact']=='low' for e in holidays))
            with self.assertRaisesRegex(Blocked,"news blackout"):
                NewsGate(path).check(NOW,300)
            NewsGate(path).check(NOW+1800,300)
            with self.assertRaisesRegex(Blocked,"coverage"):
                NewsGate(path).check(NOW+86401,300)

    def test_saving_unchanged_calendar_with_new_clock_keeps_automatic_refresh(self):
        with tempfile.TemporaryDirectory() as temp:
            c=Workstation(temp)
            c.config.update(calendar="data/news-auto.json",calendar_auto=True)
            # A client can include the current calendar in a full config update.
            c.configure({"login":123,"server":"test","terminal":"",
                         "calendar":"data/news-auto.json","broker_clock":{"zone":"+02:00"}})
            saved=Workstation(temp).config
            self.assertTrue(saved['calendar_auto'])
            self.assertEqual(saved['broker_clock']['zone'],'+02:00')
            c.configure({"calendar":"data/manual-calendar.json"})
            self.assertFalse(Workstation(temp).config['calendar_auto'])

    def test_timezone_impact_and_source_update_survive_import(self):
        doc = parse_weekly(ROWS, NOW, NOW-60)
        self.assertEqual(doc["events"][0]["time"], "2026-11-16T12:10:00+00:00")
        self.assertEqual(doc["events"][0]["impact"], "high")
        self.assertEqual(doc["updated_at"], iso(NOW-60))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"news.json";atomic_json(path,doc)
            with self.assertRaisesRegex(Blocked,"news blackout"):
                NewsGate(path).check(NOW,300)

    def test_wrong_week_empty_missing_usd_and_unknown_impact_are_rejected(self):
        cases = [[], [ROWS[0]|{"date":"2026-11-09T12:00:00Z"}],
                 [ROWS[0]|{"country":"EUR"}], [ROWS[0]|{"impact":"unknown"}],
                 [ROWS[0]|{"date":"2026-11-16T12:10:00"}], [ROWS[0]|{"date":None}],
                 ROWS+[ROWS[0]|{"impact":"Low"}]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                parse_weekly(rows,NOW,NOW-60)
        for modified in (NOW-86401,NOW+60):
            with self.assertRaises(ValueError):parse_weekly(ROWS,NOW,modified)

    def test_week_bounds_handle_both_dst_transitions(self):
        start,end=week_bounds(instant("2026-03-09T12:00:00Z"))
        self.assertEqual(end-start,167*3600)
        start,end=week_bounds(instant("2026-11-02T12:00:00Z"))
        self.assertEqual(end-start,169*3600)

    def test_failed_download_preserves_old_calendar_and_its_expiry(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"news.json";old={"previous":"must survive"};atomic_json(path,old)
            for response in (Response(content_type="text/html"),Response(modified=NOW-86401),
                             Response([ROWS[0]|{"date":"2026-11-09T12:00:00Z"}])):
                with patch("fxb.news.urlopen",return_value=response),patch("time.time",return_value=NOW):
                    with self.assertRaises(ValueError):refresh_calendar(path)
                self.assertEqual(json.loads(path.read_text()),old)

    def test_hourly_cache_does_not_refresh_source_timestamp_or_call_network(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"news.json"
            with patch("fxb.news.urlopen",return_value=Response()) as call, patch("time.time",return_value=NOW):
                result=refresh_calendar(path)
            self.assertEqual(result["status"],"updated");call.assert_called_once()
            saved=path.read_bytes()
            with patch("fxb.news.urlopen",side_effect=AssertionError("cache must avoid a request")), \
                 patch("time.time",return_value=NOW+1800):
                result=refresh_calendar(path)
            self.assertEqual(result["status"],"cached")
            self.assertEqual(path.read_bytes(),saved)
            with patch("fxb.news.urlopen",return_value=Response(modified=NOW+3600)) as call, \
                 patch("time.time",return_value=NOW+3700):
                self.assertEqual(refresh_calendar(path)["status"],"updated")
            call.assert_called_once()

    def test_atomic_write_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"news.json";atomic_json(path,{"old":True})
            with patch.object(Path,"replace",side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):atomic_json(path,{"new":True})
            self.assertEqual(json.loads(path.read_text()),{"old":True})
            self.assertFalse(list(Path(temp).glob("*.tmp")))

    def test_cli_enables_auto_only_after_a_successful_import_and_preserves_account(self):
        with tempfile.TemporaryDirectory() as temp:
            c=Workstation(temp)
            c.configure({"login":123,"server":"test","broker_clock":{"zone":"+02:00"}})
            path=Path(temp)/"state/workstation.json";before=path.read_bytes()
            args=parser().parse_args(["calendar-update","--root",temp,"--auto"])
            with patch("fxb.news.urlopen",side_effect=OSError("offline")):
                with self.assertRaises(OSError):run(args)
            self.assertEqual(path.read_bytes(),before)
            with patch("fxb.news.urlopen",return_value=Response()),patch("time.time",return_value=NOW), \
                 patch("fxb.cli.MT5",side_effect=AssertionError("No broker calls for a calendar")):
                result=run(args)
            config=Workstation(temp).config
            self.assertEqual(config["login"],123)
            self.assertEqual(config["broker_clock"]["zone"],"+02:00")
            self.assertEqual(config["calendar"],"data/news-auto.json")
            self.assertTrue(config["calendar_auto"])
            self.assertEqual(result["orders_sent"],0)

    def test_background_import_is_separate_and_manual_calendar_disables_it(self):
        called=threading.Event()
        doc=parse_weekly(ROWS,NOW,NOW-60)
        def load(path):
            self.assertNotEqual(threading.current_thread(),threading.main_thread())
            called.set()
            return doc,False
        with tempfile.TemporaryDirectory() as temp:
            c=Workstation(temp)
            try:
                with patch("fxb.news.load_weekly",side_effect=load):
                    c.enable_calendar_auto()
                    self.assertTrue(called.wait(2))
                    deadline=time.monotonic()+2
                    while c.calendar_refresh["status"] not in ("updated","error") and time.monotonic()<deadline:
                        time.sleep(.01)
                    self.assertEqual(c.calendar_refresh["status"],"updated")
                self.assertTrue(Workstation(temp).config["calendar_auto"])
                manual={"source_url":SOURCE,"updated_at":iso(time.time()-1),
                        "coverage_start":iso(time.time()-60),"coverage_end":iso(time.time()+3600),"events":[]}
                with c.lock:c.save_calendar(manual)
                self.assertFalse(Workstation(temp).config["calendar_auto"])
            finally:c.stop_calendar_refresh()
            self.assertFalse(c.calendar_thread.is_alive())


if __name__=="__main__":unittest.main()
