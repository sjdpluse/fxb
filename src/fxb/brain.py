"""Fixed recurrent topology + genuinely trained nonlinear Double-DQN readout.

This is a reservoir experiment, not a biophysical reconstruction of a fly brain.
No pickle deserialization is used for graphs or policy checkpoints.
"""
from pathlib import Path
from collections import deque
import csv
import hashlib
import json
import numpy as np


class Reservoir:
    def __init__(self, inputs, recurrent, provenance):
        self.inputs = np.asarray(inputs, dtype=float)
        self.recurrent = np.asarray(recurrent, dtype=float)
        self.provenance = provenance
        n = self.recurrent.shape[0]
        if (self.recurrent.shape != (n, n) or self.inputs.shape[0] != n or n > 2048
                or not np.isfinite(self.inputs).all() or not np.isfinite(self.recurrent).all()
                or np.any(np.abs(self.recurrent).sum(axis=1) > 0.850001)):
            raise ValueError("Invalid or unstable reservoir")

    @classmethod
    def synthetic(cls, input_dim, size=64, seed=7):
        rng = np.random.default_rng(seed)
        signs = np.where(rng.random(size) < 0.2, -1.0, 1.0)
        rec = (rng.random((size, size)) < 0.08) * rng.random((size, size)) * signs
        rec *= 0.85 / np.maximum(1, np.abs(rec).sum(axis=1, keepdims=True))
        return cls(rng.normal(0, 0.3 / np.sqrt(input_dim), (size, input_dim)), rec,
                   {"kind": "synthetic_sparse_control", "seed": seed,
                    "biological_connectome": False})

    def encode(self, features):
        state = np.zeros(len(self.recurrent))
        encoded = np.empty((len(features), features.shape[1] + len(state)))
        for i, x in enumerate(features):
            state = 0.5 * state + 0.5 * np.tanh(self.inputs @ x + self.recurrent @ state)
            encoded[i] = np.r_[x, state]
        return encoded

    def save(self, path):
        np.savez_compressed(path, inputs=self.inputs, recurrent=self.recurrent,
                            provenance=np.array(json.dumps(self.provenance)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            return cls(f["inputs"], f["recurrent"], json.loads(str(f["provenance"])))


def import_connectome(edge_csv, provenance_json, output, input_dim, seed=7):
    """Import an explicitly selected real subgraph, retaining source/checksum.

    CSV: pre,post,weight,sign. sign must be +1/-1, explicitly curated by the
    caller, never guessed from transmitter names. Nodes remain string IDs.
    A 2048-node ceiling prevents silently calling a small model a whole brain.
    """
    meta = json.loads(Path(provenance_json).read_text(encoding="utf-8"))
    for k in ("source_url", "release", "license", "selection", "sign_method"):
        if not isinstance(meta.get(k), str) or not meta[k].strip():
            raise ValueError(f"Missing provenance: {k}")
    ids, edges = {}, []
    with open(edge_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            a, b = row["pre"], row["post"]
            weight, sign = float(row["weight"]), float(row["sign"])
            if not a or not b or not np.isfinite(weight) or weight <= 0 or sign not in (-1, 1):
                raise ValueError("Edges require node IDs, positive weights and explicit signs")
            for node in (a, b):
                if node not in ids:
                    ids[node] = len(ids)
            if len(ids) > 2048 or len(edges) >= 2_000_000:
                raise ValueError("Select a documented subgraph of <=2048 nodes and <=2M edges first")
            edges.append((ids[a], ids[b], weight * sign))
    if len(ids) < 2 or not edges:
        raise ValueError("Empty or trivial graph")
    rec = np.zeros((len(ids), len(ids)))
    for pre, post, weight in edges:
        rec[post, pre] += weight
    rec *= 0.85 / np.maximum(1, np.abs(rec).sum(axis=1, keepdims=True))
    rng = np.random.default_rng(seed)
    meta.update(kind="user_supplied_connectome_subgraph", biological_connectome="user_attested",
                sha256=hashlib.sha256(Path(edge_csv).read_bytes()).hexdigest(),
                nodes=len(ids), edges=len(edges), node_order=list(ids),
                transform="incoming absolute row normalization to <=0.85; fixed tanh reservoir")
    Reservoir(rng.normal(0, 0.3 / np.sqrt(input_dim), (len(ids), input_dim)), rec, meta).save(output)
    return {k: v for k, v in meta.items() if k != "node_order"}


class Network:
    def __init__(self, input_dim, hidden=64, seed=7):
        rng = np.random.default_rng(seed)
        self.p = {"w1": rng.normal(0, 1 / np.sqrt(input_dim), (input_dim, hidden)),
                  "b1": np.zeros(hidden), "w2": rng.normal(0, 0.02, (hidden, 4)), "b2": np.zeros(4)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.steps = 0

    def predict(self, x):
        return np.tanh(x @ self.p["w1"] + self.p["b1"]) @ self.p["w2"] + self.p["b2"]

    def gradients(self, x, actions, target):
        hidden = np.tanh(x @ self.p["w1"] + self.p["b1"])
        q = hidden @ self.p["w2"] + self.p["b2"]
        error = q[np.arange(len(x)), actions] - target
        loss = np.where(abs(error) <= 1, 0.5 * error ** 2, abs(error) - 0.5).mean()
        dq = np.zeros_like(q)
        dq[np.arange(len(x)), actions] = np.clip(error, -1, 1) / len(x)
        dh = (dq @ self.p["w2"].T) * (1 - hidden ** 2)
        grads = {"w1": x.T @ dh, "b1": dh.sum(axis=0), "w2": hidden.T @ dq, "b2": dq.sum(axis=0)}
        return float(loss), grads

    def train(self, x, actions, target, learning_rate=0.0003):
        loss, grads = self.gradients(x, actions, target)
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in grads.values()):
            raise FloatingPointError("Nonfinite training step")
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        self.steps += 1
        for k, g in grads.items():
            g = g * min(1, 5 / max(norm, 1e-12))
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            m = self.m[k] / (1 - 0.9 ** self.steps)
            v = self.v[k] / (1 - 0.999 ** self.steps)
            self.p[k] -= learning_rate * m / (np.sqrt(v) + 1e-8)
        return loss


class DQN:
    def __init__(self, dim, seed=7, capacity=20_000, batch_size=64):
        self.online, self.target = Network(dim, seed=seed), Network(dim, seed=seed)
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.updates = 0

    def act(self, state, mask, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return int(np.argmax(np.where(mask, self.online.predict(state), -np.inf)))

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None
        chosen = self.rng.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[int(i)] for i in chosen]
        s, a, r, nxt, done, masks = [np.asarray(v) for v in zip(*batch)]
        best = np.argmax(np.where(masks, self.online.predict(nxt), -np.inf), axis=1)
        values = self.target.predict(nxt)[np.arange(len(batch)), best]
        target = r + 0.99 * (1 - done) * values
        loss = self.online.train(s, a, target)
        self.updates += 1
        if self.updates % 100 == 0:
            self.target.p = {k: v.copy() for k, v in self.online.p.items()}
        return loss
