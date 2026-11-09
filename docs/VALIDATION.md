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

## Not established by these checks

- No real MT5 terminal or broker account was connected. MT5 transport tests use a fake API. No real or demo order was submitted by this development session.
- No market-history dataset was supplied or used to establish returns. The manufactured data are labelled synthetic and cannot authorize execution.
- No FlyWire dataset was downloaded, trained or represented as a biological simulation.
- No Supabase project was modified. The SQL bootstrap and HTTP integration still need a test against the intended database, including role/grant verification and security advisors.
- No profitability, forecast accuracy, live readiness, general intelligence or causal market understanding is claimed.
- GitHub Actions repeats tests and CLI smoke checks on Python 3.11/3.12 for Linux/Windows, including the installed MT5 SDK surface on Windows. The initial core suite passed in all four jobs; hosted results for subsequent commits are recorded in the repository's Actions tab.
