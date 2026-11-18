# Gold M1 scalper — 0.5.0

## Micro baskets and manual orders

The new UI default is **Micro**, saved as a separate `M1-micro` profile. Old
Active/Structure profiles keep their previous behavior and learning scopes.
No long historical training is required: the full connectome still encodes
500 recent closed M1 bars, and the bounded readout learns from completed trades.
Biological weights are fixed. This release adds an explicit quote trigger around
that context, not pretrained general market intelligence.

Micro polls at 0.5 seconds **plus** MT5/API/computation latency. It keeps up to
eight seconds of distinct bid quotes, requires at least four samples over one
second, 60% directional imbalance and favorable recent velocity. Movement must
exceed the greater of 0.05 ATR, three ticks, or 1.25 spreads plus configured
round-trip commission/slippage, and must not already exceed one ATR. Duplicate,
stale or disconnected quotes cannot build a signal. A signal expires after
three seconds; graph warmup clears tick history and waits for a fresh signal.
This measures quote momentum, not exchange volume/order-flow, and does not
promise to capture every fluctuation or provide low-latency HFT execution.

Choose **5..10 orders**, each **0.01 or 0.02 lot**. Multiple tickets require a
hedging account (`ACCOUNT_MARGIN_MODE_RETAIL_HEDGING`); runtime rechecks this.
One basket at a time is allowed. Default aggregate stop-risk allowance is 2.5%
of equity, subject to the selected per-order limit (default 0.5%), remaining
daily loss allowance (default 3%), persistent 8% drawdown floor, 20% margin
budget and daily entry quota (default 250 individual orders, not 250 baskets).
There is no silent reduction of the requested count or rounding lots upward.
The full proposed basket must fit before the first send; each later child is
repriced and checked against the remaining and original aggregate budgets.
These allowances are configurable caps, not a suitability recommendation.
Stops, slippage, gaps and delayed broker replies can still produce larger losses.

Children are submitted sequentially, with a durable unique order claim each.
Every position receives broker SL/TP. Only FOK/IOC filling is supported for
baskets; RETURN is rejected because an unfilled remainder could stay pending. Stops use at least 0.45 M1 ATR, 4.1
spreads and the broker minimum distance; net planned reward must cover risk
including configured costs. A rejection, partial fill, expired signal, operator
stop or unknown response cancels remaining children. Already-confirmed positions
continue being managed. Unknown entries are never automatically retried; they
do not prevent closing a separately proven position. Unknown closes cannot be
repeated. A restart resumes management with the original profile, never the
unsubmitted portion of a basket. IDs are linked through deal history/comments,
not by assuming order and position ticket numbers are equal.

The exit latches when estimated basket liquidation profit remains positive and
favorable quote velocity fades below 30% of its observed peak, or more than 40%
of protected peak profit is given back, sustained for a second on fresh samples.
Net estimation includes closed siblings, broker-booked fees, open floating
profit/swap, and estimated exit commission/slippage. Broker history is required
for that profit estimate. Attached stops, maximum two-minute hold, equity halts
and operator flatten work independently of the neural/news/history entry path.
A basket contributes **one** learning reward after all confirmed children close,
using actual linked net cash flow divided by recorded stop risk. Manually chosen
orders and externally modified positions are excluded from policy learning.

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
