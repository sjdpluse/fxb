"""Loopback-only operator interface. Preview mode has no broker/write authority."""
import argparse
from contextlib import asynccontextmanager
from functools import lru_cache
import hmac
import importlib.util
import json
import os
from pathlib import Path
import secrets
import threading
import webbrowser
from urllib.parse import urlparse
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .connectome import geometry
from .mt5 import Blocked
from .workstation import Workstation

STATIC = Path(__file__).parent / "static"


def create_app(root=".", preview=False, token=None):
    # Brian2 installs a SIGINT handler on its first import. Load it on the main
    # thread before HTTP/background workers; the 15M-edge graph remains lazy.
    if importlib.util.find_spec("brian2") is not None:
        import brian2  # noqa: F401
    controller = Workstation(root, preview)
    token_path = controller.state_dir / "owner.token"
    if token is None:
        if not token_path.exists():
            fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(secrets.token_urlsafe(32))
        token = token_path.read_text().strip()
    if not token or len(token) < 24:
        raise ValueError("Invalid workstation access token")

    @asynccontextmanager
    async def lifespan(app):
        yield
        controller.worker_stop.set()
        controller.cancel_job.set()
        if controller.worker:
            controller.worker.join(timeout=5)
        if controller.bridge and not (controller.worker and controller.worker.is_alive()):
            controller.bridge.close()

    app = FastAPI(title="FXB Workstation", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.controller, app.state.owner_token = controller, token
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if not preview:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if not preview:
                provided = request.headers.get("authorization", "").removeprefix("Bearer ")
                if not hmac.compare_digest(provided, token):
                    return JSONResponse({"error": "Open the workstation using its local access link"}, status_code=401)
                origin = request.headers.get("origin")
                if origin and urlparse(origin).netloc != request.url.netloc:
                    return JSONResponse({"error": "Cross-origin control is disabled"}, status_code=403)
            elif request.method != "GET" and request.url.path not in ("/api/brain/stimulate", "/api/job/cancel"):
                return JSONResponse({"error": "Preview permits only viewing and neural stimulation"}, status_code=403)
            if int(request.headers.get("content-length", "0")) > 2_000_000:
                return JSONResponse({"error": "Request too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        return response

    @app.exception_handler(Blocked)
    @app.exception_handler(ValueError)
    @app.exception_handler(FileNotFoundError)
    @app.exception_handler(KeyError)
    @app.exception_handler(TypeError)
    def invalid(request, error):
        return JSONResponse({"error": str(error)}, status_code=409)

    @app.get("/api/status")
    def status():
        return controller.status()

    @lru_cache(maxsize=1)
    def anatomical_geometry():
        return geometry(controller.brain_dir)

    @app.get("/api/brain/geometry")
    def brain_geometry():
        return anatomical_geometry()

    @app.get("/api/brain/neuron/{index}")
    def neuron(index: int):
        import numpy as np
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        with np.load(controller.brain_dir / "index.npz", allow_pickle=False) as f:
            ids, degree = f["ids"], f["degree"]
        if not 0 <= index < len(ids):
            raise ValueError("Unknown neuron index")
        t = pq.read_table(controller.brain_dir / "Connectivity_783.parquet",
            columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        t = t.filter(pc.or_(pc.equal(t["Presynaptic_Index"], index), pc.equal(t["Postsynaptic_Index"], index)))
        rows = sorted(t.to_pylist(), key=lambda x: abs(x["Excitatory x Connectivity"]), reverse=True)[:30]
        return {"index": index, "root_id": str(ids[index]), "degree": int(degree[index]),
                "connections": [{"pre": str(ids[r["Presynaptic_Index"]]), "post": str(ids[r["Postsynaptic_Index"]]),
                                 "weight": r["Excitatory x Connectivity"]} for r in rows], "shown": len(rows)}

    @app.get("/api/journal")
    def journal():
        return controller.journal()

    @app.post("/api/{action:path}")
    def command(action: str, payload: dict):
        # Serialize operator mutations; jobs own their own DB connections.
        with controller.lock:
            if action == "brain/install":
                anatomical_geometry.cache_clear()
                return controller.install_brain()
            if action == "brain/stimulate":
                return controller.stimulate(payload.get("channels", "all"), int(payload.get("frames", 8)), float(payload.get("rate", 150)))
            if action == "job/cancel":
                controller.cancel_job.set()
                return {"status": "cancellation_requested"}
            if action == "account/configure": return controller.configure(payload)
            if action == "account/connect": return controller.connect()
            if action == "account/refresh":
                if controller.worker and controller.worker.is_alive():
                    return controller.account_data
                controller.refresh_account()
                return controller.account_data
            if action == "data/export":
                return controller.export(str(payload["symbol"]), int(payload["bars"]), float(payload["commission"]),
                                         float(payload["slippage"]), float(payload["overnight"]))
            if action == "training/start":
                return controller.start_training(str(payload["dataset"]), int(payload.get("epochs", 10)), str(payload.get("encoder", "whole-brain")))
            if action == "scalper/prepare":
                return controller.prepare_scalper(str(payload["symbol"]), float(payload["commission"]),
                                                  float(payload["slippage"]), float(payload["overnight"]))
            if action == "execution/start":
                models = payload.get("models", [])
                if not isinstance(models, list) or any(not isinstance(p, str) for p in models):
                    raise ValueError("Invalid model selection")
                return controller.start(models, str(payload.get("mode", "dry-run")), payload.get("acknowledged_login"), float(payload.get("hours", 8)))
            if action == "execution/pause": return controller.pause()
            if action == "execution/stop": return controller.stop(False)
            if action == "execution/flatten": return controller.stop(True)
            if action == "execution/reconcile": return controller.reconcile_orders()
            if action == "calendar/save": return controller.save_calendar(payload)
            if action == "memory/sync": return controller.sync()
            return JSONResponse({"error": "Unknown operation"}, status_code=404)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(root=".", port=8765, open_browser=True, preview=False):
    import uvicorn
    app = create_app(root, preview)
    url = f"http://127.0.0.1:{port}/#token={app.state.owner_token}"
    if not preview:
        print("FXB local access link (keep private): " + url, flush=True)
    if open_browser and not preview:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="0.0.0.0" if preview else "127.0.0.1", port=port, access_log=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    p.add_argument("--host", default=None)  # Preview runner passes these; authority still comes from --preview.
    p.add_argument("--strictPort", action="store_true")
    a = p.parse_args()
    serve(port=a.port, preview=a.preview, open_browser=False)
