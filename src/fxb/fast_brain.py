"""Compiled execution of the pinned Shiu LIF equations and full v783 graph.

Uses the same 0.1 ms schedule and per-observation NumPy Poisson stream as the
Brian2 NumPy reference. No pruning, reduced graph, timestep or window change.
The Brian2 implementation remains the reference and default backend.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import contextmanager
import numpy as np
from .connectome import WholeBrain, ASSETS, sha256, adapter_hash

IMPLEMENTATION = "numba-shiu-lif-v1"


@contextmanager
def cache_lock(path):
    """OS-owned lock releases on process exit, including a power interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as file:
        if file.tell() == 0:
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another process is encoding the same dataset") from error
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file, fcntl.LOCK_UN)


def _advance(v, g, last, refractory, queue, lengths, tick, offsets, targets, weights,
             inputs, uniforms, probabilities, decay_g, v_const, g_to_v, decay_v):
    n = len(v)
    counts = np.zeros(n, dtype=np.int64)
    available = np.empty(n, dtype=np.bool_)
    fired = np.empty(n, dtype=np.int32)
    for k in range(len(uniforms)):
        now = tick + k
        nfired = 0
        for i in range(n):
            allowed = now - last[i] >= refractory[i]
            available[i] = allowed
            if allowed:
                old_g = g[i]
                g[i] = decay_g * old_g
                v[i] = v_const + (g_to_v * old_g + decay_v * v[i])
                if v[i] > -0.045:
                    fired[nfired] = i
                    nfired += 1
                    counts[i] += 1
                    last[i] = now
                    available[i] = False
        # A shared 1.8 ms delay lets the queue store source IDs instead of
        # expanding millions of edge IDs. All outgoing edges are still applied
        # in original order when their source's delayed event is due.
        future = (now + 18) % 19
        lengths[future] = nfired
        queue[future, :nfired] = fired[:nfired]
        due = now % 19
        for ksource in range(lengths[due]):
            source = queue[due, ksource]
            for edge in range(offsets[source], offsets[source + 1]):
                dest = targets[edge]
                if available[dest]:
                    g[dest] += weights[edge]
        lengths[due] = 0
        # Brian2's default_synapses_pre precedes the zero-delay input pathway;
        # both reject writes to neurons that are refractory in this timestep.
        for channel in range(len(inputs)):
            dest = inputs[channel]
            if uniforms[k, channel] < probabilities[channel] and available[dest]:
                v[dest] += 0.000275 * 250
        for index in range(nfired):
            neuron = fired[index]
            v[neuron], g[neuron] = -0.052, 0.0
    return counts


class FastWholeBrain(WholeBrain):
    implementation = IMPLEMENTATION

    def __init__(self, directory=None, window_ms=20, seed=7):
        super().__init__(directory, window_ms, seed)
        if not np.isclose(self.window_ms / .1, round(self.window_ms / .1), atol=1e-10):
            raise ValueError("Compiled neural window must be aligned to 0.1 ms")
        self.provenance = self.provenance | {"implementation": IMPLEMENTATION,
                                           "reference": "Brian2 NumPy, pinned Shiu equations"}

    def initialize(self):
        if self.net is not None:
            return
        from numba import njit
        import pyarrow.parquet as pq
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
            raise ValueError("Invalid neural input/readout mapping")
        table = pq.read_table(self.path / "Connectivity_783.parquet",
                             columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        pre = table["Presynaptic_Index"].to_numpy().astype(np.int32)
        post = table["Postsynaptic_Index"].to_numpy().astype(np.int32)
        weights = table["Excitatory x Connectivity"].to_numpy().astype(np.float64) * .000275
        if len(pre) != 15091983 or pre.min() < 0 or pre.max() >= len(self.ids):
            raise ValueError("Unexpected connectome graph")
        # Stable sorting preserves individual edges and accumulation order;
        # duplicate targets are not merged into a different floating-point sum.
        order = np.argsort(pre, kind="stable")
        self.offsets = np.r_[0, np.cumsum(np.bincount(pre, minlength=len(self.ids)))].astype(np.int64)
        self.targets, self.weights = post[order], weights[order]
        self.sizes = np.bincount(self.pools, minlength=64)
        self.kernel = njit(cache=True, fastmath=False)(_advance)
        dt, tau, membrane, v0 = .0001, .005, .020, -.052
        self.coefficients = (np.exp(-dt / tau), v0 - v0 * np.exp(-dt / membrane),
                             ((tau / (membrane - tau) * (-np.exp(dt / membrane) + np.exp(dt / tau)))
                              * np.exp(-dt / membrane)) * np.exp(-dt / tau), np.exp(-dt / membrane))
        self.net = True  # initialized marker; this backend has no Brian2 Network
        self._clear()
        self.initialization_seconds = time.perf_counter() - started

    def _clear(self):
        n = len(self.ids)
        self.v, self.g = np.full(n, -.052), np.zeros(n)
        self.last = np.full(n, -10**12, dtype=np.int64)
        self.refractory = np.full(n, 22, dtype=np.int64)
        self.refractory[self.inputs] = 0
        self.queue = np.zeros((19, n), dtype=np.int32)
        self.lengths = np.zeros(19, dtype=np.int64)
        self.tick = 0
        self.last_time = None
        self.latest = {"status": "ready", "active_neurons": 0, "spikes": 0, "simulation_ms": 0,
                       "implementation": IMPLEMENTATION}

    def reset(self):
        self.initialize()
        self._clear()

    def step(self, observation=None, stimulus_indices=None, rate_hz=150):
        self.initialize()
        started = time.perf_counter()
        x = np.zeros(19) if observation is None else np.asarray(observation, dtype=float)
        if x.shape != (19,) or not np.isfinite(x).all():
            raise ValueError("Expected 19 finite market features")
        rates = np.repeat(np.r_[np.clip(x, 0, 8), np.clip(-x, 0, 8)] * (150 / 8), 4)
        if stimulus_indices is not None:
            rates[:] = 0
            chosen = np.asarray(stimulus_indices, dtype=int)
            if np.any(chosen < 0) or np.any(chosen >= len(rates)) or not 0 <= rate_hz <= 300:
                raise ValueError("Invalid stimulation channel or rate")
            rates[chosen] = rate_hz
        key = hashlib.sha256(x.astype("<f8").tobytes() + rates.astype("<f8").tobytes()).digest()
        seed = (self.seed + int.from_bytes(key[:4], "little")) % (2**32 - 1)
        steps = round(self.window_ms / .1)
        uniforms = np.random.RandomState(seed).random_sample((steps, len(self.inputs)))
        counts = self.kernel(self.v, self.g, self.last, self.refractory, self.queue, self.lengths,
                             self.tick, self.offsets, self.targets, self.weights, self.inputs,
                             uniforms, rates * .0001, *self.coefficients)
        self.tick += steps
        self.last_counts = counts
        active = np.flatnonzero(counts)
        pooled = np.bincount(self.pools, weights=counts, minlength=64)
        embedding = np.log1p(pooled / np.maximum(1, self.sizes) * (1000 / self.window_ms))
        strongest = active[np.argsort(counts[active])[-4000:]]
        self.latest = {"status": "running", "active_neurons": len(active), "spikes": int(counts.sum()),
                       "simulation_ms": self.tick * .1, "window_ms": self.window_ms,
                       "wall_seconds": time.perf_counter() - started, "indices": strongest.tolist(),
                       "counts": counts[strongest].tolist(), "pool_activity": embedding.tolist(),
                       "input_rates": rates.tolist(), "mean_voltage_mv": float(self.v.mean() * 1000),
                       "implementation": IMPLEMENTATION, "source": "compiled_full_shiu_lif_simulation",
                       "display_activity_capped_at": 4000}
        return np.r_[x, embedding]

    def encode_cached(self, observations, dataset_hash, progress=None, cache_directory=None,
                      checkpoint_every=200):
        """Resume encoding only from a complete atomic neural-state checkpoint.

        DQN epochs are deliberately separate; this is not an optimizer resume.
        Cache loading never unpickles Python objects.
        """
        x = np.ascontiguousarray(observations, dtype="<f8")
        if (x.ndim != 2 or x.shape[1] != 19 or not np.isfinite(x).all()
                or type(checkpoint_every) is not int or checkpoint_every < 1):
            raise ValueError("Invalid cached encoding inputs")
        from importlib.metadata import version
        identity = {"schema": 1, "dataset_sha256": dataset_hash, "topology": self.provenance,
                    "implementation_sha256": sha256(Path(__file__)), "numpy": np.__version__,
                    "numba": version("numba"), "observations_sha256": hashlib.sha256(x.tobytes()).hexdigest()}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = Path(cache_directory) if cache_directory is not None else self.path.parent / "encoding-cache"
        root.mkdir(parents=True, exist_ok=True)
        path = root / (key + ".npz")
        with cache_lock(root / (key + ".lock")):
            self.reset()
            encoded = np.empty((len(x), self.output_dim))
            start = 0
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    meta = json.loads(str(saved["metadata"]))
                    start = meta["completed"]
                    if (meta.get("identity") != identity or type(start) is not int or not 0 <= start <= len(x)
                            or saved["encoded"].shape != (start, self.output_dim)
                            or not np.isfinite(saved["encoded"]).all()):
                        raise ValueError("Invalid neural encoding cache; use a new cache directory")
                    state = {name: saved[name].copy() for name in ("v", "g", "last", "queue", "lengths")}
                    for name, value in state.items():
                        target = getattr(self, name)
                        if value.shape != target.shape or value.dtype != target.dtype or not np.isfinite(value).all():
                            raise ValueError("Invalid cached neural state")
                    if (np.any(state["lengths"] < 0) or np.any(state["lengths"] > len(self.ids))
                            or np.any(state["queue"] < 0) or np.any(state["queue"] >= len(self.ids))
                            or np.any(state["last"] >= start * round(self.window_ms / .1))):
                        raise ValueError("Invalid cached spike queue or refractory state")
                    for name, value in state.items():
                        setattr(self, name, value)
                    self.tick = start * round(self.window_ms / .1)
                    encoded[:start] = saved["encoded"]
            started = time.monotonic()
            if progress:
                progress({"stage": "brain_encoding", "completed": start, "total": len(x),
                          "resumed_bars": start, "elapsed_seconds": 0,
                          "remaining_seconds_estimate": None, "implementation": IMPLEMENTATION})
            for i in range(start, len(x)):
                encoded[i] = self.step(x[i])
                completed = i + 1
                if completed % checkpoint_every == 0 or completed == len(x):
                    temporary = path.with_suffix(".tmp.npz")
                    with temporary.open("wb") as file:
                        np.savez_compressed(file, metadata=np.array(json.dumps({"identity": identity, "completed": completed})),
                                            encoded=encoded[:completed], v=self.v, g=self.g, last=self.last,
                                            queue=self.queue, lengths=self.lengths)
                        file.flush()
                        os.fsync(file.fileno())
                    temporary.replace(path)
                if progress and (completed % 20 == 0 or completed == len(x)):
                    elapsed = time.monotonic() - started
                    progress({"stage": "brain_encoding", "completed": completed, "total": len(x),
                              "resumed_bars": start, "elapsed_seconds": elapsed,
                              "remaining_seconds_estimate": elapsed / (completed - start) * (len(x) - completed),
                              "implementation": IMPLEMENTATION,
                              "activity": {k: v for k, v in self.latest.items() if k not in ("indices", "counts", "input_rates")}})
            return encoded
