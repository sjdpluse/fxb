# Research protocol and acceptance boundaries

## Hypotheses

1. A frozen recurrent graph can encode useful temporal market information.
2. A documented FlyWire subgraph may improve an otherwise identical Double-DQN readout.
3. Any improvement survives transaction costs, new dates, new seeds and demo execution.

None of these hypotheses is an established property of this repository. The default synthetic graph is a computational control, not biological evidence. A connectome maps anatomical connections; it does not encode financial knowledge or establish general reasoning ability.

## Data and chronology

- Obtain the exact broker's bid M5 bars independently for gold and the broker's Nasdaq CFD. Preserve immutable CSV + metadata, checksums, account currency, units and data source.
- Validate data before training. The feature scaler is fit only on the training partition. Feature windows use at most the current closed bar and preceding observations. Reservoir state is causal; at inference, 512 closed bars provide a contracting-state warmup.
- The policy sees bar `t` after its close. Its action first executes at bar `t+1` open. No same-close fill or future stop-hit information is fed into the decision.
- Positions are liquidated at partition ends. Keep a holding-window embargo between train, validation and test.
- Select epochs using validation return minus twice validation maximum drawdown (both percentage points). This is a declared engineering criterion, not a statistically justified optimal objective.
- Evaluate the frozen winner once on the final holdout. The `evaluate` command can reproduce that report, but repeatedly choosing models from it destroys its status as untouched evidence.
- Walk-forward uses expanding histories and successive nonoverlapping holdout windows. Later training can include earlier historical outcomes. Each fold restarts its capital and must not be presented as a compounded portfolio curve.

## Learning and reward

The fixed reservoir feeds a trainable tanh MLP with four Q outputs. Adam, Huber loss, gradient clipping, bounded replay, legal-action masks and a periodically copied target network train the readout. Double-DQN selects the next action using the online network and values it using the target network. Terminal transitions do not bootstrap. There is no supervised label claiming to know the uniquely correct trade.

At each step:

`reward = log(equity_after/equity_before) / risk_fraction`

`         - 0.5 * increase_in_drawdown / risk_fraction`

`         - 0.005 * opened_position`

Equity already includes charged spread, commission, slippage and applicable configured overnight cost. Costs are not subtracted a second time in reward. Open positions are marked to executable closing quotes. Risk exits and rejected entries remain under the independent engine. A rational trained policy can choose WAIT for every observation.

Weights are learned by the readout, not by the graph. Replaying old transitions is different from an episodic archive or recalling a causal explanation. The recall API restricts both experiment ID and label availability time and returns similar past observations; it is diagnostic, not a policy feature or proof of causation.

## Comparison

Compare the plain DQN control, the synthetic reservoir, and each documented real subgraph using the same histories, seed list, training budget, costs, risk rules and evaluation dates. Keep the flat and simple trend controls. Report all trials, including failures, and include doubled-cost stress. Avoid selecting a particular subgraph after inspecting test outcomes without allocating a fresh test set.

Before considering a future live-capable release, obtain sufficient independent trades across volatile, trending and ranging periods; inspect return, drawdown, net expectancy, tail losses and profit factor jointly. Use block-bootstrap uncertainty intervals appropriate to temporal dependence and account for model-selection trials. No hardcoded count of trades or fixed profit factor certifies safety. This release reports descriptive metrics and does not implement statistical promotion approval.

## Execution reconciliation

Real account submissions are rejected. Demo submissions require the exact configured login, a current quote, known contract properties, broker SL/TP support, an approved order_check and adequate risk/margin. Both models share one worker, journal, daily guard and entry slot; the model order in the CLI provides deterministic priority, not portfolio optimization.

Persist the decision and order reservation before sending. Unknown outcomes stay blocked across restarts. Partial fills are accepted without topping up; the resulting position is read from the broker. Matching broker deal comments can reconcile uncertain requests. A missing match remains unresolved. Independent state directories or external writers can defeat single-worker assumptions and are unsupported.

News blocking uses a separately maintained calendar with explicit coverage. The software validates the schema, freshness and timing, not the factual completeness of the supplied schedule. Network failure may prevent market exits; attached stops are still subject to broker execution, gaps and slippage.

## Next evidence required

- Actual broker-specific history and transaction-cost estimates.
- A selected and licensed biological dataset/subgraph with explicit sign assumptions.
- Multiple controlled research runs and preserved evaluation reports.
- Windows MT5 demo forward execution, including disconnects, rollover, rejection and partial-fill scenarios.
- Applying the journal schema to the intended Supabase project and verifying role access, insert/upsert and advisors there.
- Tick-level and correlated multi-asset portfolio simulation before treating individual-symbol reports as portfolio performance.

Until that evidence exists, the appropriate status is **research prototype with tested software paths**, not an autonomous profitable trader.
