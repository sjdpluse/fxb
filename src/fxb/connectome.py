"""Pinned original Shiu/Spiller whole-brain model and real FlyWire v783 data.

Original model source is preserved in fxb.upstream under its MIT license.
Dataset licensing is separate; see the manifest and THIRD_PARTY.md.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
import threading
import time
import urllib.request
import numpy as np

SOURCE_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
SOURCE_URL = "https://github.com/philshiu/Drosophila_brain_model"
RAW_URL = f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SOURCE_COMMIT}/"
ASSETS = {
    "Completeness_783.csv": (3327347, "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
    "Connectivity_783.parquet": (100804642, "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
}
COORDINATE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783/coordinates.csv.gz"
_BRIAN_LOCK = threading.RLock()  # Brian's global code generation/RNG are not thread-safe.


def brain_directory():
    return Path(os.environ.get("FXB_BRAIN_DIRECTORY", "data/connectome")).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest, expected_size=None, expected_hash=None, progress=None):
    dest = Path(dest)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        if expected_hash is None or sha256(dest) == expected_hash:
            return
    temp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=40) as response, temp.open("wb") as out:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                out.write(block)
                if progress:
                    progress({"stage": "download", "file": dest.name, "bytes": total, "total": expected_size})
                if expected_size is not None and total > expected_size:
                    raise ValueError("Unexpected upstream file size")
        if expected_size is not None and temp.stat().st_size != expected_size:
            raise ValueError("Incomplete upstream download")
        if expected_hash is not None and sha256(temp) != expected_hash:
            raise ValueError("Upstream SHA-256 mismatch")
        temp.replace(dest)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def install(directory=None, progress=None):
    """Download exact published data, preserve all model neurons and connections."""
    import pandas as pd
    import pyarrow.parquet as pq
    path = Path(directory or brain_directory())
    path.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in ASSETS.items():
        download(RAW_URL + name, path / name, size, digest, progress)
    if progress:
        progress({"stage": "index", "message": "Indexing all published neurons and connections"})
    ids = pd.read_csv(path / "Completeness_783.csv", index_col=0).index.to_numpy(dtype=np.int64)
    table = pq.read_table(path / "Connectivity_783.parquet", columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory x Connectivity"])
    pre = table["Presynaptic_Index"].to_numpy().astype(np.int32)
    post = table["Postsynaptic_Index"].to_numpy().astype(np.int32)
    weights = table["Excitatory x Connectivity"].to_numpy().astype(np.float32)
    if (pre.min() < 0 or post.min() < 0 or max(pre.max(), post.max()) >= len(ids)
            or not np.isfinite(weights).all()):
        raise ValueError("Invalid biological graph indices or weights")
    edges = len(pre)
    synapses = int(table["Connectivity"].to_numpy().sum())
    del table
    # Small topology caches and stable engineering interfaces; no graph truncation.
    degree = np.bincount(pre, minlength=len(ids)) + np.bincount(post, minlength=len(ids))
    candidates = np.flatnonzero(degree > 0)
    rng = np.random.default_rng(783)
    inputs = rng.choice(candidates, 19 * 2 * 4, replace=False).astype(np.int32)
    pools = (ids % 64).astype(np.int32)
    np.savez_compressed(path / "index.npz", ids=ids, inputs=inputs, pools=pools, degree=degree.astype(np.int32))
    # Display a documented sample of links, while simulation retains every edge.
    selection = np.linspace(0, edges - 1, min(edges, 20000), dtype=np.int64)
    np.savez_compressed(path / "display_edges.npz", pre=pre[selection], post=post[selection], weights=weights[selection])
    del pre, post, weights
    geometry = {"available": False, "source": COORDINATE_URL}
    try:
        download(COORDINATE_URL, path / "coordinates.csv.gz", 5314546,
                 "14337121f451f98c2576cee72c24409ada5aaf7948b7c7ca8de9040296840e05", progress)
        lookup = {str(value): i for i, value in enumerate(ids)}
        coordinates = np.full((len(ids), 3), np.nan, dtype=np.float32)
        with gzip.open(path / "coordinates.csv.gz", "rt") as f:
            for row in csv.DictReader(f):
                index = lookup.get(row["root_id"])
                if index is not None and np.isnan(coordinates[index, 0]):
                    values = np.fromstring(row["position"].strip("[]"), sep=" ")
                    if len(values) == 3 and np.isfinite(values).all():
                        coordinates[index] = values
        valid = np.isfinite(coordinates).all(axis=1)
        if valid.sum() < 100:
            raise ValueError("No usable anatomical coordinates")
        center = np.median(coordinates[valid], axis=0)
        scale = max(float(np.ptp(coordinates[valid], axis=0).max()), 1)
        coordinates[valid] = (coordinates[valid] - center) / scale * 2
        np.savez_compressed(path / "geometry.npz", positions=coordinates, valid=valid)
        geometry.update(available=True, nodes=int(valid.sum()), sha256=sha256(path / "coordinates.csv.gz"),
                        transform="median-centered and uniformly scaled; missing positions are omitted")
    except Exception as e:
        geometry["error"] = str(e)[:160]
    manifest = {"schema": 1, "kind": "shiu_lif_v783", "source": SOURCE_URL, "source_commit": SOURCE_COMMIT,
        "release": "FlyWire v783", "neurons": len(ids), "connections": edges, "synapse_contacts": synapses,
        "graph_truncated": False, "model_license": "MIT", "data_license_notice": "FlyWire guidelines: CC BY-NC 4.0; dataset terms are separate from the MIT model code.",
        "assets": {name: {"sha256": digest, "bytes": size} for name, (size, digest) in ASSETS.items()},
        "geometry": geometry, "installed_at": datetime.now(timezone.utc).isoformat(),
        "adapter": "152 deterministic feature-input neurons; 64 root-ID pooling groups; engineering mapping, not biological market semantics"}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if progress:
        progress({"stage": "ready", "manifest": manifest})
    return manifest


def read_manifest(directory=None):
    return json.loads((Path(directory or brain_directory()) / "manifest.json").read_text(encoding="utf-8"))


def adapter_hash(path):
    digest = hashlib.sha256(b"fxb-feature-poisson-v1")
    with np.load(Path(path) / "index.npz", allow_pickle=False) as f:
        for name, dtype in (("ids", "<i8"), ("inputs", "<i4"), ("pools", "<i4")):
            digest.update(f[name].astype(dtype).tobytes())
    return digest.hexdigest()


class WholeBrain:
    """The original full Brian2 model, with a separate market input/readout adapter."""
    def __init__(self, directory=None, window_ms=20, seed=7, codegen="numpy"):
        self.path = Path(directory or brain_directory())
        self.manifest = read_manifest(self.path)
        if self.manifest["source_commit"] != SOURCE_COMMIT or self.manifest["graph_truncated"]:
            raise ValueError("Unsupported or truncated model pack")
        if not 5 <= window_ms <= 200:
            raise ValueError("Neural window must be 5..200 milliseconds")
        self.window_ms, self.seed, self.codegen = float(window_ms), int(seed), codegen
        self.provenance = {"kind": "shiu_lif_v783", "biological_connectome": True,
                           "source": SOURCE_URL, "source_commit": SOURCE_COMMIT,
                           "neurons": self.manifest["neurons"], "connections": self.manifest["connections"],
                           "window_ms": self.window_ms, "seed": self.seed,
                           "input_mapping_sha256": adapter_hash(self.path),
                           "input_mapping": self.manifest["adapter"], "assets": self.manifest["assets"]}
        self.output_dim = 19 + 64
        self.net = None
        self.last_time = None
        self.latest = {"status": "not_initialized", "active_neurons": 0, "spikes": 0, "simulation_ms": 0}
        self.initialization_seconds = None

    def initialize(self):
        if self.net is not None:
            return
        with _BRIAN_LOCK:
            import brian2 as b
            from .upstream import shiu_model
            started = time.perf_counter()
            for name, (size, digest) in ASSETS.items():
                if (self.path / name).stat().st_size != size or sha256(self.path / name) != digest:
                    raise ValueError("Biological model data changed; reinstall the pinned model pack")
            with np.load(self.path / "index.npz", allow_pickle=False) as f:
                self.ids, self.inputs, self.pools = f["ids"], f["inputs"], f["pools"]
            if (adapter_hash(self.path) != self.provenance["input_mapping_sha256"]
                    or self.ids.shape != (138639,) or self.inputs.shape != (152,)
                    or self.pools.shape != self.ids.shape or len(np.unique(self.inputs)) != 152
                    or self.inputs.min() < 0 or self.inputs.max() >= len(self.ids)
                    or self.pools.min() < 0 or self.pools.max() >= 64):
                raise ValueError("Invalid or changed neural input/readout mapping")
            b.prefs.codegen.target = self.codegen
            b.defaultclock.dt = 0.1 * b.ms
            self.params = dict(shiu_model.default_params)
            self.neurons, self.synapses, original_monitor = shiu_model.create_model(
                self.path / "Completeness_783.csv", self.path / "Connectivity_783.parquet", self.params)
            # Count-only monitor bounds memory during long trading sessions.
            self.monitor = b.SpikeMonitor(self.neurons, record=False)
            self.stimuli = b.PoissonGroup(len(self.inputs), rates=np.zeros(len(self.inputs)) * b.Hz)
            self.input_synapses = b.Synapses(self.stimuli, self.neurons,
                on_pre="v_post += stimulus_weight", namespace={"stimulus_weight": self.params["w_syn"] * self.params["f_poi"]})
            self.input_synapses.connect(i=np.arange(len(self.inputs)), j=self.inputs)
            self.neurons.rfc[self.inputs] = 0 * b.ms
            self.net = b.Network(self.neurons, self.synapses, self.monitor, self.stimuli, self.input_synapses)
            self.net.store("rest")
            del original_monitor
            self.initialization_seconds = time.perf_counter() - started
            self.latest = {"status": "ready", "active_neurons": 0, "spikes": 0, "simulation_ms": 0,
                           "initialization_seconds": self.initialization_seconds}

    def reset(self):
        self.initialize()
        with _BRIAN_LOCK:
            self.net.restore("rest")
        self.last_time = None
        self.latest = {"status": "ready", "active_neurons": 0, "spikes": 0, "simulation_ms": 0}

    def step(self, observation=None, stimulus_indices=None, rate_hz=150):
        self.initialize()
        import brian2 as b
        started = time.perf_counter()
        x = np.zeros(19) if observation is None else np.asarray(observation, dtype=float)
        if x.shape != (19,) or not np.isfinite(x).all():
            raise ValueError("Expected 19 finite market features")
        with _BRIAN_LOCK:
            # Feature input is explicitly engineered. Original connectome weights/dynamics are unchanged.
            positive = np.clip(x, 0, 8)
            negative = np.clip(-x, 0, 8)
            rates = np.repeat(np.r_[positive, negative] * (150 / 8), 4)
            if stimulus_indices is not None:
                rates[:] = 0
                chosen = np.asarray(stimulus_indices, dtype=int)
                if np.any(chosen < 0) or np.any(chosen >= len(rates)) or not 0 <= rate_hz <= 300:
                    raise ValueError("Invalid stimulation channel or rate")
                rates[chosen] = rate_hz
            self.stimuli.rates = rates * b.Hz
            # Deterministic per-observation randomness, independent of replay/partition traversal.
            key = hashlib.sha256(x.astype("<f8").tobytes() + rates.astype("<f8").tobytes()).digest()
            b.seed((self.seed + int.from_bytes(key[:4], "little")) % (2**32 - 1))
            before = np.asarray(self.monitor.count).copy()
            self.net.run(self.window_ms * b.ms, namespace={})
            counts = np.asarray(self.monitor.count).astype(np.int64) - before
            active = np.flatnonzero(counts)
            pooled = np.bincount(self.pools, weights=counts, minlength=64)
            sizes = np.bincount(self.pools, minlength=64)
            embedding = np.log1p(pooled / np.maximum(1, sizes) * (1000 / self.window_ms))
            strongest = active[np.argsort(counts[active])[-4000:]]
            self.latest = {"status": "running", "active_neurons": len(active), "spikes": int(counts.sum()),
                "simulation_ms": float(self.net.t / b.ms), "window_ms": self.window_ms,
                "wall_seconds": time.perf_counter() - started, "indices": strongest.tolist(),
                "counts": counts[strongest].tolist(), "pool_activity": embedding.tolist(),
                "input_rates": rates.tolist(), "mean_voltage_mv": float(np.mean(self.neurons.v / b.mV)),
                "source": "full_original_shiu_brian2_simulation", "display_activity_capped_at": 4000}
        return np.r_[x, embedding]

    def encode(self, observations):
        self.reset()
        return np.asarray([self.step(x) for x in observations])

    def observe(self, observations, timestamps):
        """Consume only newly closed bars in a persistent execution session."""
        if self.last_time is None:
            self.reset()
            start = max(0, len(observations) - 64)
        else:
            start = int(np.searchsorted(timestamps, self.last_time, side="right"))
            if start == 0:
                raise ValueError("Market history moved backwards")
        for i in range(start, len(observations)):
            self.last_embedding = self.step(observations[i])
            self.last_time = float(timestamps[i])
        return self.last_embedding


def geometry(directory=None):
    path = Path(directory or brain_directory())
    with np.load(path / "index.npz", allow_pickle=False) as f:
        ids = f["ids"]
    with np.load(path / "geometry.npz", allow_pickle=False) as f:
        valid, positions = f["valid"], f["positions"]
    indices = np.flatnonzero(valid)
    return {"indices": indices.tolist(), "ids": [str(x) for x in ids[valid]],
            "positions": positions[valid].round(6).ravel().tolist(), "source": "FlyWire v783 anatomical coordinates",
            "total_model_neurons": len(ids), "visible_neurons": len(indices)}
