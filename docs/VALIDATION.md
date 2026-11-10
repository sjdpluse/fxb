# Validation record — v0.2.0 — 2026-11-10 (Asia/Kabul)

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
