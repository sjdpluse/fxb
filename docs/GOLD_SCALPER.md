# Gold M1 scalper — 0.8.1

## Recorded entry diagnostics

In 0.8.1, Execution distinguishes a passing quote-momentum filter from entry
permission. The structure panel shows each alternative's actual measurements,
thresholds and failed conditions. One matching alternative is sufficient;
later alternatives are labelled not evaluated after a match. Zone validity,
net reward/risk, learned-entry, calendar and account checks still follow.

When momentum passes but structural setup, stop geometry or target discovery
rejects the entry, the worker automatically captures the **first** rejection
per account/profile/closed M1 bar/direction/reason. The JSON contains the exact
500 closed bars used by that assessment, up to 64 sampled bid/ask quotes,
profile settings, ATR, computed structure and measured rejection. Repeating
the same rejection within that bar does not overwrite the first sample.
At most twelve snapshots are retained atomically in local state, separately
from the event/cloud outbox, learning weights, orders and account risk state.
Failure to save a diagnostic cannot authorize a rejected order.

Select the recorded Kabul timestamp in Execution and click
**دانلود گزارش تصمیم و ۵۰۰ کندل**. The authenticated, read-only endpoint is
`GET /api/diagnostics/entry/{id}` (`/api/diagnostics/entry` returns the latest).
The download is a frozen observation, not a new MT5 data request. It remains
available after restart; expired report IDs do not silently return a different
snapshot. Credentials, account identifiers/balances and learned weights are
excluded. The user's earlier screenshot cannot be reconstructed retroactively;
the next qualifying decision will supply the missing candles.

For a read-only reproduction with no broker connection or model training:

```powershell
.\.venv\Scripts\python.exe -m fxb.entry_diagnostics --report 'C:\path\FXB-entry-ID.json'
```

This recomputes momentum and structural selection from the captured arrays,
with an integrity check on captured bars. `reason_matches` means the decision
is reproducible; it does not establish that the rejection was economically
optimal, validate profitability, or authorize trading. An update can be checked
against the same report without placing a trade. Existing profiles need no
upgrade for diagnostics; trading rules, neural dynamics and risk remain intact.

## Structural entries in 0.7.0

New or explicitly upgraded Micro profiles require quote momentum PLUS a closed
M1 structure setup: confirmed liquidity sweep / existing ICT gap retest,
breakout retest, or trend-aligned pullback. M5 bias uses only closed blocks.
Stops are beyond the invalidation extreme with a 0.10 ATR / two-tick buffer.
Targets are before the nearest observed opposing pivot/range bound, never
pushed beyond it just to manufacture a reward ratio. SELL bid-bar zones are
converted to the closing ask side. Every child is checked against the full
configured net reward/risk ratio after commission and adverse slippage.

A hypothetical $5 account is not automatically executable: the broker minimum
lot, stop distance, free margin and risk budget must permit it. Automatic sizing
uses account-currency `order_calc_profit` and `order_calc_margin`, without moving
the structural stop to fit the account or silently increasing risk. This is
not evidence that JustMarkets' standard gold contract is feasible at $5.

The engine shows SUBMITTING before contacting MT5 and MANAGING after confirmed
fills. Manual requests are explicitly labelled and do not become learning
samples. WAITING for a tick is distinct from managing an existing basket.

The raster contains real kernel spike indices and 0.1 ms timestamps from the
last simulation window, capped at 4096 events. It does not animate fabricated
spikes between M1 observations. The 19 market features, input rates, learning
revision, contribution of the 64 neural outputs and last entry plan are visible.
Reward = settled net cash / original filled stop risk; the bounded readout
updates once per fully closed autonomous basket. Prediction error, weight delta
and resulting weights are persisted together. Biological synapses are fixed;
"dopamine" is an analogy for the reward signal, not a biological implementation.

These new setup rules have software regression coverage, not verified market
profitability. Low win rate alone and high win rate alone do not determine net
expectancy; actual costs and average win/loss matter.

## Micro baskets and manual orders

The new UI default is **Micro**, saved as a separate `M1-micro` profile. Old
Active/Structure profiles keep their previous behavior and learning scopes.
No long historical training is required: the full connectome still encodes
features calculated from 500 recent closed M1 bars (64 neural warmup windows,
then newly closed bars), and the bounded readout learns from completed trades.
Biological weights are fixed. This release adds an explicit quote trigger around
that context, not pretrained general market intelligence.

Micro polls at 0.5 seconds **plus** MT5/API/computation latency. It keeps up to
eight seconds of distinct bid quotes, requires at least four samples over one
second, 60% directional efficiency and favorable recent velocity. As of 0.8.0,
new or explicitly upgraded profiles use `direction_policy=price_path`: signed
net bid displacement divided by the sum of absolute bid changes. The sign is
aligned with the proposed trade. Existing profiles without this field retain
`tick_count`, the number-based imbalance (up moves minus down moves divided by
nonzero moves), so installing the update alone does not change their entries.
Equal counts can still represent a directional path if the price changes in
one direction are larger. Equal net movement with excessive back-and-forth
travel still fails. Neither measure is exchange volume or order flow.

Movement must
exceed the greater of 0.05 ATR, three ticks, or 1.25 spreads plus configured
round-trip commission/slippage, and must not already exceed one ATR. Duplicate,
stale or disconnected quotes cannot build a signal. A signal expires after
three seconds; graph warmup clears tick history and waits for a fresh signal.
This measures quote momentum, not exchange volume/order-flow, and does not
promise to capture every fluctuation or provide low-latency HFT execution.

Execution shows each measured momentum condition and its threshold: minimum
cost-covering movement, maximum one-ATR movement, directional efficiency/count
imbalance, and recent velocity. The audit includes all failed conditions and is
also returned in `scalper.momentum` and `last_decision.details.momentum` when
evaluated. During a quote gap or active-basket management it is not presented
as a fresh entry assessment. Passing this filter is not an order authorization;
structural, learned-entry, calendar, exposure, cost, risk and margin checks still
apply. Count imbalance remains unchanged in the learned readout's input; adding
price-path gating does not reinterpret existing learned weights.

New Micro profiles default to **capacity sizing** and **no daily count cap**
(`max_trades=0`). A positive count still selects an optional daily quota. Capacity
is the integer minimum of remaining aggregate stop-risk allowance, free-margin
budget and the broker's declared directional volume allowance, using the actual broker-calculated per-order volume. New profiles default to
**risk-based volume**, floored to the broker step; optional fixed volume keeps
the explicit 0.01/0.02 choice. There is no fixed ten-order cap in capacity mode. Fixed mode
retains an explicit 5..10 count; manual checks/sends retain their explicit 1..10
count. More orders are not additional independent opportunities or learning samples.

Multiple tickets require a hedging account; one basket at a time is allowed.
Default aggregate stop risk is 2.5% of equity, per-order risk 0.5%, daily loss 3%,
persistent maximum drawdown 8%, and aggregate margin budget 20% of free margin.
Capacity mode uses the largest count fitting **all** these limits, not all raw
free margin. Per-order volume is never rounded upward. Fixed/manual counts are
not silently reduced. The proposal must fit before the first send; each later
child is checked against current equity, free margin, directional volume and
original aggregate budgets. A sibling that closes cancels remaining entries.
`ACCOUNT_LIMIT_ORDERS` limits pending orders, not market-position counts.
Undeclared broker limits and changing requirements are still enforced by actual
`order_check` / order-send retcodes. See the official [symbol volume rules](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants)
and [margin-estimation semantics](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercalcmargin_py).
Stops, gaps and delayed replies can produce losses beyond estimates.

Children are submitted sequentially, with a durable unique order claim each.
Every position receives broker SL/TP. Only FOK/IOC filling is supported for
baskets; RETURN is rejected because an unfilled remainder could stay pending.
Legacy momentum profiles use stops of at least 0.45 M1 ATR, 4.1
spreads and the broker minimum distance; net planned reward must cover risk
including configured costs. In 0.5.1 the initial target is priced for the
configured **net** reward/risk ratio using `order_calc_profit`. Configured
round-trip commission and slippage are included; volume and SL are unchanged.
The target cannot exceed 2.5 ATR times the configured reward ratio. Incompatible
costs remain blocked with the required target, cash risk and cost inputs shown.
Every repriced child now requires the full configured net reward/risk ratio.
New structural profiles use the zones described above instead of the legacy
ATR target extension.
A rejection, partial fill, expired signal, operator
stop or unknown response cancels remaining children. Already-confirmed positions
continue being managed. Unknown entries are never automatically retried; they
do not prevent closing a separately proven position. Unknown closes cannot be
repeated. A restart resumes management with the original profile, never the
unsubmitted portion of a basket. IDs are linked through deal history/comments,
not by assuming order and position ticket numbers are equal.

New/upgraded profiles use **confirmed** profit protection:

- Default minimum hold is 20 seconds from the first fill. Protection arms only
  after net basket profit reaches the larger of 0.5 times original filled stop
  risk (0.5R), three cash ticks or the cash noise allowance.
- A giveback must exceed **all** of 35% of peak net profit, 0.15R and the noise
  allowance (the larger of two current spreads and 0.10 entry ATR, converted to
  cash for the open volume). Favorable velocity must also fall below 30% of peak.
- Both conditions must persist on fresh, distinct samples for three seconds;
  gaps reset confirmation. Barely positive profit or a small price wiggle does
  not qualify. The basket must still have positive estimated liquidation net.
- Defaults are configurable hold/confirmation times, with a five-minute maximum
  holding time (1..15 selectable). Attached SL/TP, missing-protection exits,
  account risk halts and operator flatten remain immediate and independent.
  Stops are never widened by this policy. Exit reasons identify time, risk,
  operator stop or profit protection separately.

Net estimation includes closed siblings, booked fees, open profit/swap and
estimated exit commission/slippage. History is required for profit protection,
but never for emergency exits. Existing profiles/baskets without the new fields
retain the old one-second fade policy and their original maximum holding time.

A completely closed autonomous basket supplies **one** reward: actual linked
`profit + commission + swap + fee`, divided by the original filled stop risk.
Eight entry-time tick features now accompany the existing 19 market + 64 neural
features and direction/setup context. Positive/negative net outcomes update a
bounded prediction of R; this changes future entry scores and can block losing
contexts. The reward event, prediction before/after, weight-change magnitude,
revision and weights commit in one SQLite transaction. Replays cannot train twice;
Supabase's outbox, when configured, receives the committed reward audit. Training
uses neither floating outcomes nor manual/external interventions. Biological
synapses remain fixed; this is an adaptive readout, not a pretrained trading brain.

## Upgrading a Micro profile without losing learning

Stop the worker and select the **same Micro profile that placed the trades** in
Execution. Click **ارتقای Micro انتخاب‌شده با حفظ یادگیری**. The result is a new
profile with unlimited daily count, capacity sizing, price-path direction gating,
structural entries, risk-based volume and the patient exit defaults;
the UI selects it, but does not start execution. The old file stays untouched.
The new profile retains its parent's learning lineage; server, login and actual
DEMO/LIVE mode still scope the memory separately. Existing 86 weights keep their
meaning and the added eight weights begin at zero. Repeated upgrades keep the
same lineage. Guards, history, entry counts and STOP are never cleared. Existing
open baskets continue under their frozen original policy and can settle into the
same learning lineage. Live activation must be renewed for the new file hash.
Creating a brand-new profile through the preparation form intentionally starts a
separate learner. No long historical training is required for an upgrade.

CLI alternative (FXB stopped):

```powershell
.\.venv\Scripts\python.exe -m fxb scalp-upgrade --root . --model 'runs\YOUR-EXISTING-MICRO\model.npz'
```

In **Execution**, use the new manual panel with the selected M1 worker running:

1. Select BUY/SELL, 0.01/0.02 lot and count (default one; multiple requires Micro).
2. **Check broker / no send** returns `order_check` plus total lots, planned stop
   loss and margin. It sends zero orders and can diagnose without a news calendar.
   Successful checking does not guarantee a subsequent order will be filled.
3. **Send manual order** sends to the prominently displayed DEMO/LIVE account,
   with the same account, costs, news, risk and protective-stop gates. Dry-run
   cannot send. Live requires the existing exact-account/profile activation.
4. The panel displays actual retcodes/comments, orders/deals, filled volumes and
   partial/unknown/blocked results. A rejected request is not shown as a trade.

Manual intents run in the same locked worker, expire after ten seconds, and are
idempotent across repeated requests. They can be sent while automatic entries
are paused; stop/flatten still cancels them. An in-flight operation cannot be
recalled after the broker accepts it. Switching accounts/profiles or restarting
never replays an unconsumed intent. Manual positions get the same exit management
but cannot be used as an autonomous learning sample.

Official protocol references: [account modes](https://www.mql5.com/en/docs/constants/environment_state/accountinformation),
[order check](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py),
[trade return codes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes).

## Starting a new demo risk period

Equity guards cover the entire account, including manual trading. A persistent
drawdown halt is not evidence that FXB placed the losing trades. Re-funding the
same demo account, changing its profile or restarting FXB does not clear it.

After re-funding a **demo** account, stop the worker, finish basket settlement
and resolve any uncertain orders. The account must have no positions or pending
orders. In Execution, choose **بررسی شروع دورهٔ جدید دمو**, review the actual
account number, server, equity and previous/new baselines, then explicitly record
the new period. The review expires after two minutes; changes to the account,
equity, balance, current-day entries or guard require another review.

The backend rechecks the actual MT5 demo mode and configured login/server while
holding the worker lease. It atomically records an audit event and sets day-start
and peak equity to current equity, clearing only that account's two risk halts.
Current-day entry counts, model learning, order receipts, other account guards
and the STOP file remain intact. The worker stays off; restarting execution is
a separate action. Retrying an already recorded request does not rebase again.
There is no live-account reset and no automatic adjustment for deposits.

A fresh risk period **does not waive position-sizing limits**. For example, a
reported 0.01-lot gold order with 1.96 USD estimated stop risk cannot fit a
100 USD account's 0.5% allowance (0.50 USD). Five such orders need 9.80 USD;
their 2.5% basket allowance is only 2.50 USD. With the same quote and costs,
both selected risk limits imply at least 392 USD equity. This is a calculation,
not a funding recommendation or a guarantee of margin/news/broker acceptance.
The UI and `fixed_lot_budget` receipt display these numbers; recalculate using
the latest quote after connecting the intended demo balance. Limits are not
silently raised and SL is not narrowed to force an order through.

## Worker recovery and observable neural activity

Version 0.5.1 acquires the worker lease before acknowledging startup or clearing
STOP. A second active worker is refused with its owner PID and lock path. The
OS releases the lease when a process exits, including after a crash. The
`.worker.guard` file stays on disk intentionally; its existence is not evidence
of a running worker. Do not delete lock files while FXB is running.

A legacy PID-only `.worker.lock` is migrated automatically only after the owner
is proven dead. A living PID, denied process inspection or unreadable legacy
marker remains blocked. Close the previous FXB process before restarting; the
application never kills processes or cancels broker positions to recover a lock.

The engine card distinguishes STARTING, INITIALIZING, WARMUP, WAITING, BLOCKED,
RUNNING, PAUSED and FAILED. Micro warmup advances at most four neural windows
per worker cycle, so queued manual intents and stop requests are checked between
batches. Initial graph loading/JIT and an individual neural window are still
synchronous; they cannot be interrupted by this change. Manual controls are
disabled during initial graph loading, then become available between batches.

Neural counts describe the **last actual simulation window**, not continuously
active neurons. The panel shows the last processing time, remaining warmup bars,
or why inference is suspended (paused entries, entry gate, basket management).
Quote pressure and open-position exits do not need a new neural simulation on
every tick. Manual orders do not need a neural signal and are not training data.
A second UI opening cannot mark the active worker's pending intent interrupted.

## Active deployment profile

The UI also offers **Active**. Existing profiles and the CLI default
retain **Structure** behavior. Active adds three independently identified setups:

- Closed M1 momentum: body at least 0.45 ATR and 60% of range, close in the outer
  quarter, and a break of the prior three candles by more than 0.05 ATR and no
  more than one ATR. Candle range must not exceed 2.5 ATR.
- Trend pullback: aligned rising/falling 5- and 13-close averages, a preceding
  countertrend candle touching the fast average, then a directional close past
  that candle's extremum. Body at least 0.25 ATR; range at most two ATR.
- Live momentum: fresh bid quotes cross the preceding five closed candles'
  extremum, followed by two distinct confirming quotes in an eight-second
  history. Movement must exceed 0.15 ATR and twice the observed spread; extension
  cannot exceed 0.7 ATR. Starting beyond a level, duplicate/stale ticks, and
  a disconnected quote history cannot manufacture a crossing. The quote-based
  signal expires after eight seconds or the next M1 boundary, whichever is first.

The worker polls roughly every two seconds plus API/computation time; this is not
a submillisecond execution engine. Neural context still advances on closed M1
candles. Live quote checks supplement that context without treating a forming
candle as completed. Closed-candle/FVG setups have a 50-second entry window in
Active. An initial WAIT may be reconsidered within that minute; durable intent,
order and account-minute claims prevent duplicate orders and uncertain retries.

Active defaults: 0.5% planned equity risk per entry, 3% daily loss, 8% maximum
drawdown, 250 maximum daily entries, one account position, 1.5 gross reward/risk,
two-minute maximum hold, 15 seconds after an exit, and a three-minute pause after
five consecutive losing closes. Daily counts use Asia/Kabul. The operator can
choose 0.1..1% per-entry risk, 1..5% daily loss and 1..250 maximum entries during
profile creation. These are limits, not minimum activity targets. Stops can slip;
planned risk is not a guaranteed maximum realized loss. No position-size rounding
up, martingale or artificial trades/rewards are used to meet a quota.

The broker still prices stop loss and margin. An unaffordable minimum lot now
reports its estimated loss including configured costs, its percentage of equity,
and the chosen cash risk/margin budgets. Spread, fresh data, valid news coverage,
net reward/risk and live activation checks remain effective. Account drawdown
halts are visible even without a signal and are not reset by a new profile.
The dashboard counts distinct evaluated bars, signal-bearing bars and blocked
reasons per bar; repeated two-second polls do not inflate those counts.

Select **Active** in the preparation form, review its risk limits and actual cost
assumptions, create once, then select only the `M1-active` profile for execution.
Each profile retains a separate learning scope; these new setup rules do not
silently reuse an older profile's learned weights. CLI preparation also accepts
`--style active --risk-percent 0.5 --daily-loss-percent 3 --max-trades 250` in
addition to the symbol and mandatory cost arguments below. No claim of 100–250
daily executed trades, measured slippage or profitability has been established.

## Shared model and original Structure profile

This deployment path reads **500 recent, closed M1 bid candles** from the exact
MT5 gold symbol. It does not require the historical 20,000-bar DQN run. The
existing M5 research and deployment paths remain available separately.

## What makes decisions

1. Causal, explicit price-structure rules propose an entry.
2. Nineteen M1 structure features stimulate the complete pinned Shiu/FlyWire
   graph (138,639 neurons, 15,091,983 edges) using the compiled LIF engine.
3. An online reward readout uses those features and 64 neural activity pools to
   adjust the entry score within a fixed +/-0.20 bound. The initial readout has
   zero weights and zero learned trades: initial entries therefore follow the
   explicit rules, not a pretrained neural trading skill.
4. Independent execution/risk controls may veto an entry. The readout cannot
   change position size, SL/TP, trade limits or the account authorization.

The original biological connections are fixed. This is a purpose-built market
input/readout around a biological simulation, not a newly reconstructed fly
brain and not evidence of general reasoning or profitable trading. Five hundred
candles supply current context; they are not sufficient performance validation.

## Exact implemented setups

These are **ICT-inspired operational definitions**, not all ICT concepts or a
claim to infer institutional intent:

- A swing high/low is a unique extremum with two bars on either side. It is
  available only after both right-side candles close.
- A sweep trades beyond a confirmed swing and closes back inside that level.
- A structure break closes through a confirmed swing after the preceding close
  was on the other side. A same-direction sweep within eight bars marks the
  reversal setup; otherwise the setup is a continuation break/retest.
- A three-candle FVG requires a middle body greater than 0.8 ATR and a gap at
  least 0.08 ATR, with a matching structure break within three bars.
- Only a **later** first retest can enter: the candle touches the gap and closes
  out of it in the intended direction with an aligned body. Expired gaps
  (12 bars), invalidated gaps and failed first retests are consumed.
- M5 confirmation uses complete five-minute groups assembled from M1; a forming
  M5 candle is never used. Its last close versus the mean of four closed M5
  candles must not oppose the entry. Recent missing M1 candles block entry.

ATR is the trailing 14-candle mean true range. Features also include body and
wicks, distances to swings/FVG, tick volume, spread, and position within the last
50-candle range. All calculations use only information available at the close.

## Execution and learning

Default limits: 0.1% equity risk per entry, 1% daily loss, 5% maximum drawdown,
40 entries per Kabul day, one account position at a time, 1.5 nominal reward/risk,
five-minute maximum holding time. These are engineering defaults, not optimized
or validated strategy parameters. A daily maximum is not a target trade count.

SL is beyond the structural invalidation point. Short stops account for the ask
side of a bid chart. The entry is refused if broker stop distance, spread,
volatility, minimum lot, margin or estimated net reward/risk is unsuitable.
Lot risk is priced with `order_calc_profit`/`order_calc_margin`; contract values
for MNQ are never substituted. SL and TP must be accepted at the broker.

Entry must occur within 20 seconds after the signal candle closes, including
initial neural loading time. A moved quote or a price far from the FVG is refused.
The worker polls every two seconds, manages time/risk exits before neural
processing and closes an owned position on an opposing confirmed M1 break.
Broker SL/TP continue to operate between polls. This is M1 scalping, not
exchange-colocated high-frequency trading or tick-by-tick neural processing.

There is a 60-second post-trade cooldown and a 15-minute pause after three
consecutive losing learned trades. There is no random action exploration,
averaging down or increase in risk to recover losses.

After a position fully closes, its opening order is matched to the broker
position ID. Reward is `(profit + commission + swap + fee) / initial risk cash`,
including both entry and exit deal charges. Attached deal charges are used;
unattributed later account-level fees cannot be assigned automatically. Open or
partially closed positions, reversals, external additions and manual exits are
excluded from learning. Filled volume scales the risk denominator for partial
entry fills. A transaction commits reward and weights together, once per
position; restarts cannot reward the same trade twice.

Learning is isolated by server, account, account mode and profile hash. Creating
a different profile starts a different learning history. Demo results are not
silently merged into a live account. State is stored durably in the existing
SQLite journal; its reward events can use the existing optional journal sync.
Dry run observes and explains setups but sends no orders and creates no simulated
broker rewards. The displayed entry score is **not** a win probability.

## Run on the existing Windows installation

Stop an old training process before updating the source. Keep MT5 logged in.
Open `Start-FXB.ps1`, connect the configured account, and choose **اسکلپر و آموزش**.
In **اسکلپر طلا · M1**, use the exact symbol (for example `XAUUSD.m`) and review
commission, adverse-slippage and overnight-cost assumptions. When available,
the form copies these assumptions from previous metadata for that exact symbol.
Click **آماده‌سازی اسکلپر طلا از بازار اخیر**. This only reads data and creates a
profile in `runs/*-M1-scalper-*/model.npz`; it sends no orders.

Then select that single profile on **اجرای معاملات**, choose the intended mode,
and start the worker. Live mode retains explicit account/server/profile/risk
activation. Do not mix the M1 scalper with an M5 model in the same worker.

CLI alternative for preparation (provide your own verified cost assumptions):

```powershell
.\.venv\Scripts\python.exe -m fxb scalp-prepare --symbol XAUUSD.m --commission 0 --slippage-points 26 --overnight-cost 85
```

Those example cost values are inherited scenarios from this project's November
11 JustMarkets preparation, not a current broker tariff or measured slippage.
The 500 candles come fresh from MT5; M5 candles are never upsampled to fake M1.
Clock validity bounds are optional as of 0.3.1. If explicitly configured, they
must cover the candles and daily/settlement history. A fixed +02:00 rule does
not infer the server's seasonal DST rule. In 0.3.2, freshness is checked after
the API read completes, including profile creation. The execution panel shows
the latest close and age, and `market-check --symbols XAUUSD.m` reports raw and
converted M1/quote timestamps without changing clock settings or sending orders.

Fresh economic-calendar coverage remains mandatory. With FXB closed, run
`python -m fxb calendar-update --auto`, or use the automatic-calendar button in
the settings panel. Forex Factory's own weekly export is refreshed hourly in a
separate thread while the workstation is open. Failed or invalid downloads
preserve the prior file and its original expiry; scheduled high-impact USD
blackouts remain enforced. Manual uploads disable automatic refresh. This is a
scheduled-event filter, not a detector for unscheduled news.

The workstation shows current swings, M5 bias, last decision, number of learned
closed trades and their realized net result. The brain view uses genuine neural
activity from this adapter. Keep the workstation process and MT5 running to
manage positions and learn from closes. No external language-model API is used
in the trading loop.

## Primary implementation references

- [MetaQuotes: closed-bar indexing](https://www.mql5.com/en/docs/python_metatrader5/mt5copyratesfrompos_py)
- [MetaQuotes: deal IDs, direction, commissions, swap and fees](https://www.mql5.com/en/docs/constants/tradingconstants/dealproperties)
- [MetaQuotes: order checking](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py)
- [Published Shiu model and its sensorimotor scope](https://pmc.ncbi.nlm.nih.gov/articles/PMC11446845/)
