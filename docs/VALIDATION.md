# Validation record — 2026-11-09

## Completed in the development environment

- Linux, CPython 3.12.14, NumPy 2.3.5.
- Editable package build/install succeeded with the declared build backend.
- `python -m unittest discover -s tests -q`: **45 core tests passed**. An additional native-SDK contract test is skipped on Linux and runs on Windows with the actual pinned MetaTrader5 wheel, without connecting to a terminal.
- `python -m compileall -q src tests`: passed.
- `git diff --cached --check`: passed before publication.
- Installed CLI: synthetic generation, three training epochs, saved-model evaluation and memory inspection completed for both XAUUSD and US100 fixtures (2,400 manufactured M5 bars each). All eight command invocations exited 0. Reloaded-policy evaluation reproduced the saved holdout metrics.
- The learning tests verify gradients numerically and demonstrate weight changes and decreasing loss on a controlled target. This establishes that learning code executes, not that it learns profitable trading.
- Expanding walk-forward and the plain-DQN control run in the test suite. The final holdout is excluded from scaler fitting, optimizer updates and replay.
- Tests cover future-data invariance, invalid data, lot rounding, distinct contract values, risk and margin budgets, latched drawdown stops, next-open execution, conservative ambiguous-bar exits, gap losses, short ask exits and single daily overnight charges.
- Fake MT5 tests cover real-account rejection, wrong login, terminal restrictions, invalid/stale quotes, missing SL/TP capability, contract changes, dry-run behavior, duplicate/ambiguous/partial responses, existing exposure, manual-position ownership, calendar coverage, single-worker locks, account entry slots and synthetic-model rejection.
- Memory tests cover persistent reservations, causal run-scoped recall, UUID reuse after a failed remote batch and backend-secret destination validation.

## Supabase journal verification

- The sole active connected project was inspected. A new, independent `public.fxb_events` table and recall index were created through migrations `fxb_backend_experience_journal` and `fxb_journal_explicit_backend_privileges`.
- Verified RLS enabled; `anon` and `authenticated` have no read/insert access. The backend role has explicit SELECT/INSERT/UPDATE privileges, with DELETE/TRUNCATE revoked.
- Executed insert, duplicate-ID upsert and readback as `service_role`, with an assertion on the updated payload, inside a rolled-back transaction. Verified no test rows remained.
- Security Advisors reported the informational `rls_enabled_no_policy` finding for this backend-only table. This is intentional: no browser role is granted access, no user-facing RLS policies are needed, and the trusted backend role bypasses RLS. [Advisor explanation](https://supabase.com/docs/guides/database/database-linter?lint=0008_rls_enabled_no_policy).
- Other pre-existing findings concerned shared-project functions and Auth settings outside FXB; no changes were made to those objects. The relevant explanations are [mutable search paths](https://supabase.com/docs/guides/database/database-linter?lint=0011_function_search_path_mutable), [public SECURITY DEFINER execution](https://supabase.com/docs/guides/database/database-linter?lint=0028_anon_security_definer_function_executable), [signed-in SECURITY DEFINER execution](https://supabase.com/docs/guides/database/database-linter?lint=0029_authenticated_security_definer_function_executable), and [password protection](https://supabase.com/docs/guides/auth/password-security#password-strength-and-leaked-password-protection). This work is not a security audit of the other applications sharing the project.

## Not established by these checks

- No real MT5 terminal or broker account was connected. MT5 transport tests use a fake API. No real or demo order was submitted by this development session.
- No market-history dataset was supplied or used to establish returns. The manufactured data are labelled synthetic and cannot authorize execution.
- No FlyWire dataset was downloaded, trained or represented as a biological simulation.
- The Windows runtime still needs a locally configured backend secret and a real HTTP outbox sync test. SQL-level verification does not establish end-to-end client authentication or connectivity.
- No profitability, forecast accuracy, live readiness, general intelligence or causal market understanding is claimed.
- GitHub Actions repeats tests and CLI smoke checks on Python 3.11/3.12 for Linux/Windows, including the installed MT5 SDK surface on Windows. The initial core suite passed in all four jobs; hosted results for subsequent commits are recorded in the repository's Actions tab.
