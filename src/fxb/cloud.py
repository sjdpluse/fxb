"""Private FXB-only credentials and a retryable background journal outbox.

The local database/learner remains authoritative. Network I/O never runs in
the trading worker. Cloud events do not authorize or initiate orders.
"""
import base64
import ctypes
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse
from .memory import Memory, NoRedirect, SupabaseSink


def project_url(value):
    parsed = urlparse(str(value).strip())
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port
            or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTPS Supabase project URL or its dashboard project link")
    if parsed.hostname == "supabase.com":
        match = re.fullmatch(r"/dashboard/project/([a-z0-9]{3,64})/?", parsed.path)
        if match:
            return f"https://{match[1]}.supabase.co"
    if re.fullmatch(r"[a-z0-9]{3,64}\.supabase\.co", parsed.hostname or "") and parsed.path in ("", "/"):
        return "https://"+parsed.hostname
    raise ValueError("Use the HTTPS project URL at <ref>.supabase.co")


class DeviceSink:
    def __init__(self, url, token):
        self.url = project_url(url)+"/functions/v1/fxb-journal"
        if not isinstance(token, str) or not re.fullmatch(r"fxb_device_[0-9a-f]{64}", token):
            raise ValueError("Import a valid FXB device connection file")
        self.token = token

    def __call__(self, events):
        data = json.dumps({"events": events}, allow_nan=False).encode()
        request = urllib.request.Request(self.url, data=data, headers={
            "Authorization": "Bearer "+self.token, "Content-Type": "application/json"})
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=15) as response:
                result = json.loads(response.read(8192))
                if (response.status != 200 or not isinstance(result, dict)
                        or type(result.get("accepted")) is not int or result["accepted"] != len(events)):
                    raise RuntimeError("Cloud did not acknowledge the complete batch; events remain pending")
                return {"accepted": result["accepted"], "device_id": result.get("device_id")}
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Supabase HTTP {error.code}; events remain pending") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise RuntimeError("Supabase is unavailable; events remain pending") from None


def protect(data, decrypt=False):
    """Windows user-bound DPAPI, with no added credential dependencies."""
    class Blob(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_ubyte))]
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = Blob()
    operation = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    # UI forbidden; only the current Windows user can decrypt the stored token.
    ok = operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target))
    if not ok:
        raise ValueError("Windows could not unlock the FXB device credential")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        ctypes.windll.kernel32.LocalFree(target.data)


class CloudJournal:
    def __init__(self, directory, database):
        self.path = Path(directory)/"cloud.json"
        self.database = database
        self.stop_event, self.wake = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.state = {"status": "configured" if self.path.exists() else "not_configured", "last_success": None}

    def public_status(self):
        return dict(self.state)

    def sink(self):
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            token = value["credential"]
            if value["protection"] == "windows-dpapi":
                if os.name != "nt":
                    raise ValueError("Import the original connection file on this computer")
                token = protect(base64.b64decode(token), decrypt=True).decode()
            elif value["protection"] != "owner-file":
                raise ValueError("Unknown credential protection")
            return DeviceSink(value["project_url"], token)
        url, key = os.environ.get("FXB_SUPABASE_URL"), os.environ.get("FXB_SUPABASE_SECRET_KEY")
        return SupabaseSink(project_url(url), key) if url and key else None

    def configure(self, value):
        if not isinstance(value, dict) or value.get("schema") != 1:
            raise ValueError("Unsupported connection file")
        sink = DeviceSink(value.get("project_url", ""), value.get("device_token"))
        receipt = sink([])  # Verify credential and deployed schema before saving.
        url = project_url(value["project_url"])
        token = value["device_token"]
        protected = "windows-dpapi" if os.name == "nt" else "owner-file"
        if os.name == "nt":
            token = base64.b64encode(protect(token.encode())).decode()
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump({"project_url": url, "protection": protected, "credential": token}, file)
            os.replace(tmp, self.path)
            self.state = {"status": "connected", "project_url": url, "device_id": receipt["device_id"],
                          "last_success": time.time(), "last_batch_events": 0}
        self.wake.set()
        return {"status": "connected", "project_url": url}

    def sync_once(self, max_batches=5):
        with self.lock:
            memory = None
            try:
                sink = self.sink()
                if sink is None:
                    self.state.update(status="not_configured")
                    return 0
                memory = Memory(self.database)
                self.state.update(status="syncing", error=None)
                total = 0
                for _ in range(max_batches):
                    count = memory.sync(sink)
                    total += count
                    if not count:
                        break
                if not total:
                    sink([])  # Empty outbox still verifies the connection.
                self.state.update(status="connected", last_success=time.time(), last_batch_events=total,
                                  pending=memory.stats()["pending_sync"])
                return total
            except Exception as error:
                # Never put a transport response or credential into UI/logs.
                self.state.update(status="retry_pending", error="Cloud sync failed; local events are retained")
                raise RuntimeError(self.state["error"]) from None
            finally:
                if memory:
                    memory.close()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        def run():
            while not self.stop_event.is_set():
                try:
                    self.sync_once()
                except RuntimeError:
                    pass
                self.wake.wait(30 if self.state["status"] == "connected" else 60)
                self.wake.clear()
        self.thread = threading.Thread(target=run, daemon=True, name="fxb-cloud-outbox")
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=1)
