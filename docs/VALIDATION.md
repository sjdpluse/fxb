# Validation record

## v0.2.4 — compiled full-brain execution and encoding resume — 2026-11-11 (Asia/Kabul)

- Added an opt-in Numba implementation of the pinned Shiu LIF equations with the entire 138,639-neuron / 15,091,983-edge graph, unchanged 0.1 ms timestep, 20 ms market window, float64 state, per-observation NumPy Poisson stream and input/readout mapping. Brian2 NumPy remains the default reference backend. Compiled checkpoints explicitly record and reload their backend.
- The fixed-delay event queue stores source neurons and expands their original ordered outgoing edges at delivery. It does not prune neurons/edges or merge repeated targets. Refractory writes, thresholding, delayed recurrent input, immediate stimulus and reset ordering follow the reference schedule.
- Added atomic neural-state/embedding checkpoints every 200 observations, keyed by data, normalized observations, topology, implementation source, seed and numeric-library versions. OS locks release on process exit; interrupted writes preserve the previous completed checkpoint. Cache loading uses no pickle. This resumes encoding only; interrupted DQN learning restarts its epochs. The pre-0.2.4 running process has no transferable encoding checkpoint.
- Added `--fast-brain`, PowerShell `-FastBrain` and a workstation encoder choice. Existing prepared data remains usable. Two pinned wheel dependencies (`numba==0.64.0`, `llvmlite==0.46.0`) are included with the brain extras.
- Local standard suite: **97 tests run, 90 passed, 7 skipped** (5 opt-in whole-brain checks and 2 Windows SDK checks). The compiled-brain suite was then run with opt-in enabled: **5 tests passed in 93.280 seconds**, including the actual full graph, strong stimulation / variable input / silence comparison to Brian2, exact spike-count and embedding comparisons, voltage/conductance tolerances, interrupted cache restore in a fresh object, completed-cache reuse, invalidation on changed observations, and an 800-bar synthetic training/evaluation/checkpoint round-trip.
- Short steady-state benchmark, same host / identical observations: reference **0.5658 seconds/window**, compiled **0.04694 seconds/window** (about **12.05x**, excluding first compilation). A separate 100-real-observation compiled timing sample took **3.943 seconds**. These are short timing samples, not completed 20,000-bar training or a guaranteed Windows-device speedup.
- Python compilation, dependency consistency and whitespace checks passed. Hosted CI now runs the full original and compiled graph tests on both Windows and Linux, in addition to the existing Python 3.11/3.12 core matrix. Refer to the exact commit's Actions result for hosted outcomes.
- No MT5 orders were submitted and no trading/account/clock/risk configuration was changed. Faster mathematical execution does not establish a profitable policy; numerical agreement on tested inputs does not prove bitwise identity for all possible trajectories.

## v0.2.3 — prepared market data and funding schedule — 2026-11-11 (Asia/Kabul)

- Added offline raw-archive preparation with SHA-256 verification, explicit historical clock/cost evidence, exact contract units, UTC timestamps, preserved gaps, round-trip validation and atomic non-overwriting output. Assumptions participate in dataset hashes and are retained in model metadata. Imported nested datasets are listed by the workstation.
- Separated simulator funding cutoffs from the Kabul risk day. Explicit time-zone schedules support weekday weights and the previous-day convention for midnight funding. Legacy datasets use a documented UTC-midnight daily scenario, not an inferred broker tariff. Reports record the execution version; historical checkpoints cannot silently be re-evaluated under the new simulator.
- Added an explicit initial balance to training, evaluation and walk-forward folds; every partition and control uses the same declared value. Cost stress now doubles overnight charges as well as spread, commission and slippage.
- Added whole-brain progress/ETA output and a Windows launcher that verifies prepared dataset hashes, trains symbols sequentially and packages reports without changing MT5 account settings or sending orders. Interrupted encoding is not resumable.
- Local suite: **92 tests run, 88 passed, 4 skipped, no failures**. Skips are the two Windows SDK checks and two opt-in full-brain cases. New regression coverage includes summer/winter archive conversion, corrupted CSV rejection, quote/clock inconsistency, assumption hashing, Kabul risk midnight without an unrelated funding charge, triple rollover, weekend gaps and ambiguous DST cutoffs.
- Processed the user's supplied real M5 archive and completed ten-epoch plain-DQN control runs with chronological train/validation/holdout partitions. Those control runs are not whole-connectome training, and their results do not establish profitability. Private account inputs and data are delivered separately rather than committed to the repository.
- A 20-observation full-connectome timing probe with real market features averaged about 0.46 seconds per observation on this host using the default NumPy backend. This is a short hardware-dependent timing sample, not completed market training or an estimate guaranteed for the user's Windows device.
- The broker's historical DST calendar, exact funding cutoff, historical tariffs and realized slippage remain unconfirmed. Supplied preparation assumptions are explicitly labelled; current quote conversion alone does not prove the annual rule. The full market-history connectome training must still be run.

## Raw market-data export — 2026-11-11 (Asia/Kabul)

- The user reported three successful Windows MT5 clock checks for `XAUUSD.m` and `US100.std`, with zero orders sent, and then both native-machine full-connectome integration tests passing in **55.228 seconds**. The reported Pyparsing deprecation warning did not fail those tests. These are user-provided runtime results, distinct from hosted CI.
- Added a standalone exporter compatible with the installed 0.2.2 package. It captures closed M5 bars, contract fields and a labelled margin snapshot, validates the configured account/server and current quote/bar times, and emits a unique ZIP with per-file SHA-256 values.
- Historical timestamps are preserved raw, even outside the current clock policy's validity interval. The archive cannot be mistaken for the existing UTC training CSV format. Unknown commission, slippage and overnight costs stay null; no historical clock rule is inferred from today's offset.
- Local suite: **85 tests run, 81 passed, 4 skipped, no failures**. New coverage verifies exact raw values/hashes, private-config exclusion, account mismatch, stale quotes/bars, malformed history, non-overwriting output and execution from a temporary path containing spaces. Compilation and whitespace checks passed. Hosted Windows/Linux results are attached to the commit's Actions run.
- The exporter sends no orders and does not alter the account configuration. It does not verify a broker's historical DST policy or establish a trained strategy's profitability. Before operational training/deployment, the simulator's existing flat overnight charge at a Kabul-date boundary must also be distinguished from the broker's actual rollover schedule and tariff; those are not interchangeable.

## v0.2.2 — native MT5 record compatibility — 2026-11-11 (Asia/Kabul)

- Reproduced the reported `vars() argument must have __dict__ attribute` failure for tick, position and deal records before changing the adapter. The previous portable tests used Python namespaces and did not cover the extension record protocol.
- The adapter now reads native records through their documented `_asdict()` method, copies all fields and changes only the requested timestamp fields. Existing Python namedtuple/namespace inputs remain supported; unknown record layouts are rejected explicitly.
- Local suite: **80 tests run, 76 passed, 4 skipped, no failures**. The skipped cases are two Windows SDK checks and two opt-in full-connectome integration cases. Python compilation and Git whitespace checks passed.
- Regression coverage checks `clock-check` with records lacking both `__dict__` and `_replace`, millisecond precision, stale/future-quote rejection, position age, history filtering, and preservation of prices, costs and integer identifiers above 2^53. The fixtures send no orders.
- Windows CI additionally constructs the actual SDK `Tick`, `TradePosition` and `TradeDeal` extension types and exercises their timestamp conversion without initializing a terminal. Refer to this commit's Actions results for the native test outcome; a connected broker session still requires the user's read-only `clock-check`.
- No dependency, trading limit, clock policy, account setting or persisted model/data format changed in this patch.

## v0.2.1 — broker timestamp conversion — 2026-11-11 (Asia/Kabul)

- Local suite: **76 tests run, 73 passed, 3 skipped, no failures**. The skipped cases are the Windows SDK surface check and two opt-in full-connectome integration cases; the previous full-connectome result below is not represented as a new run.
- New regression cases cover the observed UTC+2 transport shape, rejection of stale quotes after conversion, bounded fixed offsets, different US/European DST dates, ambiguous/nonexistent civil times, UTC CSV round-trip and provenance, Kabul-midnight history filtering, one-hour position exits, incompatible legacy-model rejection, and admission checks before expensive training.
- The read-only `clock-check` path is exercised with a fake MT5 API, including the remaining 2.4-second future-clock rejection. It sends no orders.
- Python compilation, JavaScript syntax and Git whitespace checks passed. CI parses both Windows launch/update scripts and runs the SDK checks on Windows; refer to the commit's Actions run for hosted results.
- No local Windows/MT5 integration, real broker order, broker annual DST rule, live slippage or market-trained profitability has been verified by these software tests. A configured fixed offset is an explicit bounded assumption, not an inferred annual broker calendar.
- The updater pins an exact commit. Git installations require a clean tracked working tree and fast-forward; ZIP installations back up program files and preserve `data`, `runs`, `state`, `diagnostics` and `.venv`.

## v0.2.0 — 2026-11-10 (Asia/Kabul)

## Completed in the development environment

- Linux, CPython 3.12.14, NumPy 2.3.5 and Brian2 2.10.1.
- The declared workstation/brain dependencies installed in the project virtual environment. `pip check` reported no broken requirements.
- `PYTHONPATH=src FXB_FULL_BRAIN_TEST=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v`: **63 tests run, 62 passed, one Windows-only SDK check skipped**, in 8.058 seconds on the final local run.
- The opt-in biological integration tests load **138,639 actual neurons and 15,091,983 actual weighted connections**, verify nonzero spikes, exact reset reproducibility, learned readout weight changes and pickle-free policy save/load. All 138,639 anatomical coordinates are available; root IDs remain strings in JSON.
- A separate cold-process test starts the workstation and initializes the entire network in a background job. This catches Brian2's requirement to install its signal handler on the main thread before background use.
- Existing numerical-gradient, causal-feature, chronological-evaluation, risk/margin, gap/spread/cost and persistent-memory tests remain passing.
- Live execution tests use a fake broker API: exact account acknowledgement, login/server/model/risk scope, expiry before entry and immediately before send, permitted owned-position closure after expiry, closure with exhausted margin, read-only real-account dry run, duplicate prevention and unresolved-order persistence across restarts. No test contacts a brokerage account.
- Local HTTP tests cover private API access, token omission from public HTML, Host/Origin controls, preview mutation denial, file-boundary checks, synthetic-data rejection, background failure reporting and calendar saving during its own news blackout.
- CLI compatibility for the original synthetic fixtures and chronological evaluation remains covered by the tests. Earlier XAUUSD/US100 manufactured-data verification runs are retained locally and clearly labelled synthetic.
- JavaScript syntax, Python compilation and Git whitespace checks on FXB-owned files passed. The unmodified upstream files retain their original trailing whitespace and are excluded from the whitespace check so their source hashes remain exact. Windows launcher parsing and actual SDK surface checks are included in CI; see the exact workflow run for their hosted result.

## Browser verification

The actual FastAPI backend and frontend ran through the managed preview. This was not a static mockup and used no fabricated account telemetry.

- Rendered the anatomical point cloud, rotated/reset its view and inspected a real neuron through the UI.
- Clicked the real stimulation control for 8 windows at 200 Hz across 152 engineered inputs. The displayed last window showed **10,268 spikes, 6,616 active neurons and 160 ms total simulated time**. CPU wall time depends on hardware; the screenshot shows 0.57 seconds for that final window.
- Selected root `720575940612058543` (model index 12973), displaying degree 231 and the strongest real directed signed connections.
- Verified model registry, training form and full-connectome default; synthetic control results remained explicitly labelled.
- Verified Live selection reveals exact-account acknowledgement and expiry fields while activation stays disabled without a connected account. No live activation was attempted.
- Checked captured browser console output: no FXB application exception was observed; unrelated browser-extension metadata errors were present.
- Screenshot: [`images/fxb-neural-workstation.png`](images/fxb-neural-workstation.png). The preview has broker mutations disabled. The normal installed workstation implements operator-controlled broker access.

## Supabase journal verification

- The sole active connected project was inspected. A new, independent `public.fxb_events` table and recall index were created through migrations `fxb_backend_experience_journal` and `fxb_journal_explicit_backend_privileges`.
- Verified RLS enabled; `anon` and `authenticated` have no read/insert access. The backend role has explicit SELECT/INSERT/UPDATE privileges, with DELETE/TRUNCATE revoked.
- Executed insert, duplicate-ID upsert and readback as `service_role`, with an assertion on the updated payload, inside a rolled-back transaction. Verified no test rows remained.
- Security Advisors reported the informational `rls_enabled_no_policy` finding for this backend-only table. This is intentional: no browser role is granted access, no user-facing RLS policies are needed, and the trusted backend role bypasses RLS. [Advisor explanation](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy).
- Other pre-existing findings concerned shared-project functions and Auth settings outside FXB; no changes were made to those objects. The relevant explanations are [mutable search paths](https://supabase.com/docs/guides/database/database-linter?lint=0011_function_search_path_mutable), [public SECURITY DEFINER execution](https://supabase.com/docs/guides/database/database-linter?lint=0028_anon_security_definer_function_executable), [signed-in SECURITY DEFINER execution](https://supabase.com/docs/guides/database/database-linter?lint=0029_authenticated_security_definer_function_executable), and [password protection](https://supabase.com/docs/guides/auth/password-security#password-strength-and-leaked-password-protection). This work is not a security audit of the other applications sharing the project.

## Not established by these checks

- No real MT5 terminal or broker account was connected. Neither a real nor a demo order was submitted by this development session. The account execution path is implemented but still needs a Windows/broker integration test.
- No genuine broker-history dataset was supplied. No full-connectome market training, market win rate or profitability has been established. A successful neural stimulation test is not financial evidence.
- Readout weights learn; the original biological synaptic weights do not. General intelligence, causal market understanding and learned financial semantics are not claimed.
- The full-neural encoder has a finite live restart warmup. Matching indefinitely continuous state, especially after history gaps, is not established.
- The Windows runtime needs its backend environment variables and an actual REST outbox sync test. SQL-level verification does not establish end-to-end client connectivity.
- Tick-level fills, broker-specific rollover/triple swap, portfolio correlation and automated calendar collection are not implemented. The current bar simulator and supplied calendar have documented limits.
- GitHub Actions runs the core/workstation suite on Linux and Windows Python 3.11/3.12 and separately installs/downloads/tests the complete connectome on Linux Python 3.12. Hosted results are visible in the repository Actions tab; a local pass is not substituted for a hosted result.
