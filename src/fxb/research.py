"""Chronological model selection; one untouched final holdout per training run."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import uuid
import numpy as np
from .brain import DQN, Reservoir
from .data import features, Scaler, WARMUP, FEATURE_VERSION, FEATURE_NAMES, iso
from .memory import Memory
from .risk import Risk, Spec
from .simulator import Simulator, PORTFOLIO_DIM, EXECUTION_VERSION
from .rollover import Rollover


def save_checkpoint(path, agent, reservoir, scaler, meta):
    path = Path(path)
    temp = path.with_suffix(".tmp.npz")
    if hasattr(reservoir, "recurrent"):
        encoder_arrays = {"inputs": reservoir.inputs, "recurrent": reservoir.recurrent}
    else:
        meta = meta | {"format": 2, "encoder": {"kind": "shiu_lif_v783", "window_ms": reservoir.window_ms,
                                                "seed": reservoir.seed, "output_dim": reservoir.output_dim,
                                                "implementation": getattr(reservoir, "implementation", "brian2")}}
        encoder_arrays = {}
    np.savez_compressed(temp, **agent.online.p, **encoder_arrays,
                        mean=scaler.mean, scale=scaler.scale, meta=np.array(json.dumps(meta, allow_nan=False)))
    temp.replace(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        if meta.get("feature_version") != FEATURE_VERSION or meta.get("format") not in (1, 2):
            raise ValueError("Incompatible checkpoint")
        Risk(**meta["risk"])
        Spec(**meta["spec"])
        scaler = Scaler(f["mean"], f["scale"])
        if meta["format"] == 2:
            from .connectome import WholeBrain
            implementation = meta["encoder"].get("implementation", "brian2")
            if implementation == "brian2":
                encoder_class = WholeBrain
            else:
                from .fast_brain import FastWholeBrain, IMPLEMENTATION
                if implementation != IMPLEMENTATION or meta["topology"].get("implementation") != IMPLEMENTATION:
                    raise ValueError("Unknown or inconsistent whole-brain implementation")
                encoder_class = FastWholeBrain
            reservoir = encoder_class(window_ms=meta["encoder"]["window_ms"], seed=meta["encoder"]["seed"])
            if (reservoir.provenance["assets"] != meta["topology"]["assets"]
                    or reservoir.provenance["input_mapping_sha256"] != meta["topology"].get("input_mapping_sha256")):
                raise ValueError("Brain dataset does not match the frozen policy")
            input_dim = reservoir.output_dim + PORTFOLIO_DIM
        else:
            reservoir = Reservoir(f["inputs"], f["recurrent"], meta["topology"])
            input_dim = len(FEATURE_NAMES) + len(reservoir.recurrent) + PORTFOLIO_DIM
        if (scaler.mean.shape != (len(FEATURE_NAMES),) or scaler.scale.shape != scaler.mean.shape
                or np.any(scaler.scale <= 0) or not np.isfinite(scaler.mean).all()
                or not np.isfinite(scaler.scale).all()
                or (meta["format"] == 1 and reservoir.inputs.shape[1] != len(FEATURE_NAMES))):
            raise ValueError("Invalid feature transform")
        agent = DQN(input_dim)
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


def train(bars, output, epochs=6, seed=7, graph=None, risk=None, plain=False, full_brain=False,
          progress=None, initial=10_000, fast_brain=False, encoding_cache=None):
    if type(epochs) is not int or not 1 <= epochs <= 200:
        raise ValueError("epochs must be 1..200")
    risk = risk or Risk()
    bars.validate()
    if bars.spec.timeframe_seconds != 300:
        raise ValueError("Historical DQN training remains M5; use scalp-prepare for the M1 online scalper")
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("Initial balance must be positive")
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
    if sum(bool(v) for v in (graph, plain, full_brain, fast_brain)) > 1:
        raise ValueError("Choose one encoder: whole brain, supplied subgraph or plain control")
    full_brain = full_brain or fast_brain
    if full_brain:
        if fast_brain:
            from .fast_brain import FastWholeBrain
            reservoir = FastWholeBrain(seed=seed)
        else:
            from .connectome import WholeBrain
            reservoir = WholeBrain(seed=seed)
    else:
        reservoir = (Reservoir(np.empty((0, len(FEATURE_NAMES))), np.empty((0, 0)),
                           {"kind": "plain_dqn_control", "biological_connectome": False}) if plain else
                     Reservoir.load(graph) if graph else Reservoir.synthetic(len(FEATURE_NAMES), seed=seed))
    if not full_brain and reservoir.inputs.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Graph input feature count mismatch")
    if full_brain:
        normalized = scaler.transform(raw)
        if fast_brain:
            encoded = reservoir.encode_cached(normalized, bars.digest(), progress, encoding_cache)
        else:
            started = time.monotonic()
            reservoir.reset()
            encoded = np.empty((len(raw), reservoir.output_dim))
            for i, x in enumerate(normalized):
                encoded[i] = reservoir.step(x)
                if (i % 20 == 0 or i == len(raw) - 1) and progress:
                    elapsed = time.monotonic() - started
                    progress({"stage": "brain_encoding", "completed": i + 1, "total": len(raw),
                              "elapsed_seconds": elapsed,
                              "remaining_seconds_estimate": elapsed / (i + 1) * (len(raw) - i - 1),
                              "activity": {k: v for k, v in reservoir.latest.items() if k not in ("indices", "counts", "input_rates")}})
        np.savez_compressed(out / "brain_embeddings.npz", encoded=encoded, scaler_mean=scaler.mean,
                            scaler_scale=scaler.scale, dataset_sha256=np.array(bars.digest()))
    else:
        encoded = reservoir.encode(scaler.transform(raw))
    agent = DQN(encoded.shape[1] + PORTFOLIO_DIM, seed=seed)
    run_id = str(uuid.uuid4())
    meta = {"format": 1, "feature_version": FEATURE_VERSION, "features": FEATURE_NAMES,
            "run_id": run_id, "seed": seed, "synthetic": bars.synthetic,
            "data_source": bars.source, "time_basis": bars.time_basis,
            "assumptions": bars.assumptions, "rollover": Rollover.from_config(bars.rollover).config(),
            "execution_version": EXECUTION_VERSION, "initial_balance": initial,
            "dataset_sha256": bars.digest(), "spec": asdict(bars.spec),
            "risk": asdict(risk), "topology": reservoir.provenance,
            "train_end": train_end, "validation_start": val_start, "validation_end": val_end,
            "test_start": test_start, "data_end": iso(bars.time[-1] + 300),
            "train_end_time": iso(bars.time[train_end - 1] + 300), "embargo_bars": embargo,
            "deployment_status": "requires_operator_validation", "learning": "Double-DQN nonlinear readout; frozen neural dynamics",
            "encoder_type": "whole_brain_lif" if full_brain else "control"}
    memory = Memory(out / "memory.sqlite")
    history, best_score, best_epoch = [], -np.inf, 0
    try:
        for epoch in range(epochs):
            env = Simulator(bars, encoded, atr, WARMUP, train_end, risk, initial)
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
                        bars.time[env.i] + 300, {"phase": "train", "epoch": epoch + 1, "state": state.tolist(),
                            "next_state": nxt.tolist(), "action": action, "reward": reward,
                            "done": done, **detail})
                state, steps = nxt, steps + 1
            validation = evaluate(Simulator(bars, encoded, atr, val_start, val_end, risk, initial), agent)
            score = validation["return_pct"] - 2 * validation["max_drawdown_pct"]
            row = {"epoch": epoch + 1, "train": env.report(), "validation": validation,
                   "loss": float(np.mean(losses)) if losses else None, "updates": agent.updates}
            history.append(row)
            if progress:
                progress({"stage": "learning", "epoch": epoch + 1, "epochs": epochs, "validation": validation,
                          "loss": row["loss"], "updates": agent.updates})
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                save_checkpoint(out / "model.npz", agent, reservoir, scaler,
                                meta | {"selected_epoch": best_epoch, "updates": agent.updates})
            print(json.dumps({"epoch": epoch + 1, "updates": agent.updates,
                              "validation_return_pct": validation["return_pct"], "synthetic": bars.synthetic}), flush=True)
    finally:
        memory.close()
    chosen, _, _, chosen_meta = load_checkpoint(out / "model.npz")
    test_env = Simulator(bars, encoded, atr, test_start, n, risk, initial)
    holdout = evaluate(test_env, chosen)
    baselines = {name: evaluate(Simulator(bars, encoded, atr, test_start, n, risk, initial), chosen, name)
                 for name in ("flat", "trend")}
    # Stress exactly the frozen policy, with doubled costs. No model selection on this result.
    stress_bars = type(bars)(bars.time, bars.open, bars.high, bars.low, bars.close, bars.volume,
                             bars.spread * 2, Spec(**(asdict(bars.spec) | {
                                 "commission_per_lot_side": 2 * bars.spec.commission_per_lot_side,
                                 "slippage_points": 2 * bars.spec.slippage_points,
                                 "overnight_cost_per_lot": 2 * bars.spec.overnight_cost_per_lot})),
                             bars.source, bars.synthetic, bars.time_basis, bars.rollover, bars.assumptions)
    # Cost shock is applied at execution while keeping observed market features frozen.
    # This avoids claiming that a second graph traversal was performed when it was not.
    stress = evaluate(Simulator(stress_bars, encoded, atr, test_start, n, risk, initial), chosen)
    report = {"model": chosen_meta, "selected_epoch": best_epoch, "history": history,
              "holdout": holdout, "baselines": baselines, "double_cost_stress": stress,
              "profitability_established": False, "real_money_supported": True,
              "stress_method": "double execution costs with frozen observations",
              "limitations": ["bar-level execution approximation", "fixed contract conversion and margin snapshots",
                "funding is an explicit cost scenario, not a reconstructed historical broker tariff",
                "independent single-symbol research; not a joint portfolio backtest",
                "reward is not a correctness or causal explanation label",
                "one holdout is insufficient for deployment; conduct rolling out-of-sample studies and demo forward tests"]}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (out / "holdout_trades.json").write_text(json.dumps(test_env.trades, indent=2, allow_nan=False), encoding="utf-8")
    return report


def evaluate_file(bars, checkpoint, forward=False):
    agent, reservoir, scaler, meta = load_checkpoint(checkpoint)
    if meta.get("execution_version") != EXECUTION_VERSION:
        raise ValueError("Checkpoint uses an older execution simulator; retrain before comparing reports")
    if Rollover.from_config(bars.rollover).config() != meta.get("rollover"):
        raise ValueError("Rollover assumptions changed; perform a new documented experiment")
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
    env = Simulator(bars, reservoir.encode(scaler.transform(raw)), atr, start, len(bars.time),
                    Risk(**meta["risk"]), meta["initial_balance"])
    return evaluate(env, agent)


def walk_forward(bars, output, folds=3, epochs=6, seed=7, graph=None, plain=False, full_brain=False,
                 initial=10_000, progress=None, fast_brain=False, encoding_cache=None):
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
                          spec=bars.spec, source=bars.source, synthetic=bars.synthetic, time_basis=bars.time_basis,
                          rollover=bars.rollover, assumptions=bars.assumptions)
        result = train(part, root / f"fold-{i + 1}", epochs, seed, graph, plain=plain, full_brain=full_brain,
                       initial=initial, progress=progress, fast_brain=fast_brain, encoding_cache=encoding_cache)
        reports.append({"fold": i + 1, "test_start": result["model"]["test_start"],
                        "test_end": end, "holdout": result["holdout"], "baselines": result["baselines"]})
    summary = {"folds": reports, "synthetic": bars.synthetic, "profitability_established": False,
               "initial_balance": initial,
               "note": "Expanding histories with separate nonoverlapping test windows; each fold starts with the declared initial balance."}
    (root / "walk_forward.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
