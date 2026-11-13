# Gold M1 scalper — 0.3.0

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
The existing bounded broker-clock policy must cover both these candles and the
history needed for daily risk/settled outcomes. An expired clock interval must
be verified and renewed explicitly; this release never silently extends it.
Fresh economic-calendar coverage remains mandatory for new orders.

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
