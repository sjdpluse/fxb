"""Chronological model selection; one untouched final holdout per training run."""
from dataclasses import asdict
from pathlib import Path
import json
import uuid
import numpy as np
from .brain import DQN, Reservoir
from .data import features, Scaler, WARMUP, FEATURE_VERSION, FEATURE_NAMES, iso
from .memory import Memory
from .risk import Risk, Spec
from .simulator import Simulator, PORTFOLIO_DIM


def save_checkpoint(path, agent, reservoir, scaler, meta):
    path = Path(path)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **agent.online.p, inputs=reservoir.inputs, recurrent=reservoir.recurrent,
                        mean=scaler.mean, scale=scaler.scale, meta=np.array(json.dumps(meta, allow_nan=False)))
    temp.replace(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        if meta.get("feature_version") != FEATURE_VERSION or meta.get("format") != 1:
            raise ValueError("Incompatible checkpoint")
        Risk(**meta["risk"])
        Spec(**meta["spec"])
        scaler = Scaler(f["mean"], f["scale"])
        reservoir = Reservoir(f["inputs"], f["recurrent"], meta["topology"])
        if (scaler.mean.shape != (len(FEATURE_NAMES),) or scaler.scale.shape != scaler.mean.shape
                or np.any(scaler.scale <= 0) or not np.isfinite(scaler.mean).all()
                or not np.isfinite(scaler.scale).all() or reservoir.inputs.shape[1] != len(FEATURE_NAMES)):
            raise ValueError("Invalid feature transform")
        agent = DQN(len(FEATURE_NAMES) + len(reservoir.recurrent) + PORTFOLIO_DIM)
        for k in agent.online.p:
            v = f[k]
            if v.shape != agent.online.p[k].shape or not np.isfinite(v).all():
                raise ValueError("Invalid model weights")
            agent.online.p[k] = v.copy()
        agent.target.p = {k: v.copy() for k, v in agent.online.p.items()}
    return agent, reservoir, scaler, meta


def evaluate(env, agent, baseline=None):
    s, done = env.reset(), False
    while not done:
        mask = env.mask()
        if baseline == "flat":
            action = 0
        elif baseline == "trend":
            # Same risk engine, costs and max holding time as the learned policy.
            wanted = 1 if env.bars.close[env.i] > env.bars.close[env.i - 12] else 2
            action = wanted if mask[wanted] else 0
        else:
            action = agent.act(s, mask)
        s, _, done, _ = env.step(action)
    return env.report()


def train(bars, output, epochs=6, seed=7, graph=None, risk=None, plain=False):
    if type(epochs) is not int or not 1 <= epochs <= 200:
        raise ValueError("epochs must be 1..200")
    risk = risk or Risk()
    n = len(bars.time)
    if n < 800:
        raise ValueError("At least 800 M5 bars required for a smoke run; real research needs much more")
    embargo = risk.max_hold_bars + 1
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    val_start, test_start = train_end + embargo, val_end + embargo
    if min(val_end - val_start, n - test_start) < 80:
        raise ValueError("Insufficient validation/holdout after embargo")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    raw, atr = features(bars)
    scaler = Scaler.fit(raw[WARMUP:train_end])
    if graph and plain:
        raise ValueError("Choose a graph or a plain control, not both")
    reservoir = (Reservoir(np.empty((0, len(FEATURE_NAMES))), np.empty((0, 0)),
                           {"kind": "plain_dqn_control", "biological_connectome": False}) if plain else
                 Reservoir.load(graph) if graph else Reservoir.synthetic(len(FEATURE_NAMES), seed=seed))
    if reservoir.inputs.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Graph input feature count mismatch")
    encoded = reservoir.encode(scaler.transform(raw))
    agent = DQN(encoded.shape[1] + PORTFOLIO_DIM, seed=seed)
    run_id = str(uuid.uuid4())
    meta = {"format": 1, "feature_version": FEATURE_VERSION, "features": FEATURE_NAMES,
            "run_id": run_id, "seed": seed, "synthetic": bars.synthetic,
            "data_source": bars.source, "dataset_sha256": bars.digest(), "spec": asdict(bars.spec),
            "risk": asdict(risk), "topology": reservoir.provenance,
            "train_end": train_end, "validation_start": val_start, "validation_end": val_end,
            "test_start": test_start, "data_end": iso(bars.time[-1] + 300),
            "train_end_time": iso(bars.time[train_end - 1] + 300), "embargo_bars": embargo,
            "live_approved": False, "learning": "Double-DQN nonlinear readout; fixed recurrent reservoir"}
    memory = Memory(out / "memory.sqlite")
    history, best_score, best_epoch = [], -np.inf, 0
    try:
        for epoch in range(epochs):
            env = Simulator(bars, encoded, atr, WARMUP, train_end, risk)
            state, done, losses, steps = env.reset(), False, [], 0
            while not done:
                at = env.i
                epsilon = max(0.08, 0.9 * (1 - epoch / epochs))
                action = agent.act(state, env.mask(), epsilon)
                nxt, reward, done, detail = env.step(action)
                agent.buffer.append((state.copy(), action, reward, nxt.copy(), float(done), env.mask()))
                if steps % 4 == 0:
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
                # Store actual observed transitions once; replay is separate from the audit journal.
                if epoch == epochs - 1:
                    memory.record(run_id, bars.spec.symbol, "experience", bars.time[at] + 300,
                        bars.time[env.i] + 300, {"phase": "train", "state": state.tolist(),
                            "next_state": nxt.tolist(), "action": action, "reward": reward,
                            "done": done, **detail})
                state, steps = nxt, steps + 1
            validation = evaluate(Simulator(bars, encoded, atr, val_start, val_end, risk), agent)
            score = validation["return_pct"] - 2 * validation["max_drawdown_pct"]
            row = {"epoch": epoch + 1, "train": env.report(), "validation": validation,
                   "loss": float(np.mean(losses)) if losses else None, "updates": agent.updates}
            history.append(row)
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                save_checkpoint(out / "model.npz", agent, reservoir, scaler,
                                meta | {"selected_epoch": best_epoch, "updates": agent.updates})
            print(json.dumps({"epoch": epoch + 1, "updates": agent.updates,
                              "validation_return_pct": validation["return_pct"], "synthetic": bars.synthetic}), flush=True)
    finally:
        memory.close()
    chosen, _, _, chosen_meta = load_checkpoint(out / "model.npz")
    test_env = Simulator(bars, encoded, atr, test_start, n, risk)
    holdout = evaluate(test_env, chosen)
    baselines = {name: evaluate(Simulator(bars, encoded, atr, test_start, n, risk), chosen, name)
                 for name in ("flat", "trend")}
    # Stress exactly the frozen policy, with doubled costs. No model selection on this result.
    stress_bars = type(bars)(bars.time, bars.open, bars.high, bars.low, bars.close, bars.volume,
                             bars.spread * 2, Spec(**(asdict(bars.spec) | {
                                 "commission_per_lot_side": 2 * bars.spec.commission_per_lot_side,
                                 "slippage_points": 2 * bars.spec.slippage_points})), bars.source, bars.synthetic)
    stress_raw, stress_atr = features(stress_bars)
    stress = evaluate(Simulator(stress_bars, reservoir.encode(scaler.transform(stress_raw)), stress_atr,
                                 test_start, n, risk), chosen)
    report = {"model": chosen_meta, "selected_epoch": best_epoch, "history": history,
              "holdout": holdout, "baselines": baselines, "double_cost_stress": stress,
              "profitability_established": False, "real_money_supported": False,
              "limitations": ["bar-level execution approximation", "fixed contract conversion and margin snapshots",
                "independent single-symbol research; not a joint portfolio backtest",
                "reward is not a correctness or causal explanation label",
                "one holdout is insufficient for deployment; conduct rolling out-of-sample studies and demo forward tests"]}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "holdout_trades.json").write_text(json.dumps(test_env.trades, indent=2, allow_nan=False), encoding="utf-8")
    return report


def evaluate_file(bars, checkpoint, forward=False):
    agent, reservoir, scaler, meta = load_checkpoint(checkpoint)
    if asdict(bars.spec) != meta["spec"]:
        raise ValueError("Contract/cost metadata differs; perform a new documented experiment")
    if forward:
        from .data import timestamp
        if bars.time[0] < timestamp(meta["data_end"]):
            raise ValueError("Forward evaluation must start after the entire original dataset")
        start = WARMUP
    else:
        if bars.digest() != meta["dataset_sha256"]:
            raise ValueError("Dataset changed; use a strictly later --forward dataset")
        start = meta["test_start"]
    raw, atr = features(bars)
    env = Simulator(bars, reservoir.encode(scaler.transform(raw)), atr, start, len(bars.time), Risk(**meta["risk"]))
    return evaluate(env, agent)


def walk_forward(bars, output, folds=3, epochs=6, seed=7, graph=None, plain=False):
    if not 2 <= folds <= 5:
        raise ValueError("Use 2..5 chronological folds")
    ends = [int(len(bars.time) * 0.8 ** k) for k in reversed(range(folds))]
    if ends[0] < 800:
        raise ValueError("More data required for the requested walk-forward folds")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    reports = []
    for i, end in enumerate(ends):
        part = type(bars)(**{k: getattr(bars, k)[:end] for k in
                            ("time", "open", "high", "low", "close", "volume", "spread")},
                          spec=bars.spec, source=bars.source, synthetic=bars.synthetic)
        result = train(part, root / f"fold-{i + 1}", epochs, seed, graph, plain=plain)
        reports.append({"fold": i + 1, "test_start": result["model"]["test_start"],
                        "test_end": end, "holdout": result["holdout"], "baselines": result["baselines"]})
    summary = {"folds": reports, "synthetic": bars.synthetic, "profitability_established": False,
               "note": "Expanding histories with separate nonoverlapping test windows; each fold starts with 10,000 units."}
    (root / "walk_forward.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
