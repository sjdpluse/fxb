"""Short-window quote momentum, not exchange order-flow or HFT.

The connectome still observes causal closed M1 features. This adapter supplies
an independently auditable tick trigger and a profit-protection exit.
"""
from dataclasses import dataclass
import math

VERSION = "gold-m1-micro-v1"


@dataclass(frozen=True)
class MicroConfig:
    orders: int = 5
    lot: float = .01
    basket_risk: float = .025
    # Missing fields retain v1 behavior for existing, already authorized profiles.
    sizing_mode: str = "fixed"
    exit_policy: str = "legacy"
    min_hold_seconds: int = 20
    confirmation_seconds: int = 3
    profit_arm_r: float = .5
    giveback_fraction: float = .35
    entry_policy: str = "momentum"
    lot_mode: str = "fixed"

    def __post_init__(self):
        if type(self.orders) is not int or not 5 <= self.orders <= 10:
            raise ValueError("Micro baskets require 5..10 orders")
        if type(self.lot) not in (int, float) or self.lot not in (.01, .02):
            raise ValueError("Choose exactly 0.01 or 0.02 lot per order")
        if (type(self.basket_risk) not in (int, float) or not math.isfinite(self.basket_risk)
                or not .001 <= self.basket_risk <= .05):
            raise ValueError("Basket risk must be 0.1..5% of equity")
        if self.sizing_mode not in ("fixed", "capacity") or self.exit_policy not in ("legacy", "confirmed"):
            raise ValueError("Unknown basket sizing or exit policy")
        if self.entry_policy not in ("momentum", "structure") or self.lot_mode not in ("fixed", "risk"):
            raise ValueError("Unknown entry or lot policy")
        if (type(self.min_hold_seconds) is not int or not 5 <= self.min_hold_seconds <= 120
                or type(self.confirmation_seconds) is not int or not 2 <= self.confirmation_seconds <= 15
                or type(self.profit_arm_r) not in (int, float) or not .25 <= self.profit_arm_r <= 2
                or type(self.giveback_fraction) not in (int, float) or not .2 <= self.giveback_fraction <= .75):
            raise ValueError("Invalid profit protection timing or thresholds")


def quote_pressure(history, tick, now):
    """Only fresh, unique quotes count. Gaps reset the observation window."""
    t = tick.time_msc / 1000
    bid, ask = float(tick.bid), float(tick.ask)
    if not all(math.isfinite(x) for x in (t, bid, ask, now)) or not -2 <= now-t <= 2 or not 0 < bid <= ask:
        history.clear()
        return None
    if history and (t < history[-1][0] or t-history[-1][0] > 2):
        history.clear()
    if history and t == history[-1][0]:
        return None
    history.append((t, bid, ask))
    history[:] = [x for x in history[-64:] if t-x[0] <= 8]
    if len(history) < 4 or t-history[0][0] < 1:
        return None
    # Bid changes avoid false bullish momentum caused only by widening ask.
    delta = [b[1]-a[1] for a, b in zip(history, history[1:])]
    moves = [d for d in delta if d]
    if not moves:
        return None
    short = [x for x in history if t-x[0] <= 2]
    velocity = ((bid-short[0][1])/(t-short[0][0])
                if len(short) >= 2 and t-short[0][0] >= .3 else 0.)
    movement = bid-history[0][1]
    return {"time": t, "bid": bid, "ask": ask, "spread": ask-bid,
            "movement": movement, "velocity": velocity,
            "imbalance": sum(1 if d > 0 else -1 for d in moves)/len(moves),
            "duration": t-history[0][0], "samples": len(history)}


def micro_signal(pressure, atr, tick_size, cost_price):
    if pressure is None or not math.isfinite(atr) or atr <= 0:
        return None
    p = pressure
    # Movement must overcome current spread and configured friction; no quota
    # forces a trade when quotes have no economically meaningful movement.
    threshold = max(.05*atr, 1.25*p["spread"]+cost_price, 3*tick_size)
    side = 1 if p["movement"] > 0 else -1
    if (abs(p["movement"]) < threshold or abs(p["movement"]) > atr
            or side*p["imbalance"] < .60 or side*p["velocity"] <= tick_size/2):
        return None
    return {"side": side, "kind": "micro_momentum", "score": .60,
            "setup_id": f"micro:{int(p['time']*1000)}:{side}",
            "expires_at": p["time"]+3, "pressure": p,
            "entry_reference": p["ask"] if side > 0 else p["bid"]}


def weakening(group, pressure, estimated_net, now, tick_value_cash):
    """Latch a close after sustained fading or profit giveback, never on stale quotes."""
    cfg = MicroConfig(**group.get("micro_config", {}))
    if cfg.exit_policy == "confirmed":
        return confirmed_weakening(group, pressure, estimated_net, now, tick_value_cash, cfg)
    if pressure is None:
        return False
    velocity = group["side"]*pressure["velocity"]
    group["peak_velocity"] = max(group.get("peak_velocity", 0.), velocity)
    group["peak_net"] = max(group.get("peak_net", 0.), estimated_net)
    protected = estimated_net > max(tick_value_cash, 0.)
    faded = group["peak_velocity"] > 0 and velocity <= .30*group["peak_velocity"]
    giveback = group["peak_net"] > 3*tick_value_cash and estimated_net <= .60*group["peak_net"]
    if protected and (faded or giveback):
        group.setdefault("weak_since", now)
        return now-group["weak_since"] >= 1
    group.pop("weak_since", None)
    return False


def confirmed_weakening(group, pressure, net, now, tick_cash, cfg):
    """Protect meaningful net profit, with noise hysteresis and fresh-quote persistence.

    This never moves SL/TP and never delays a separate emergency/risk exit.
    R is the original filled basket risk, not floating profit or available margin.
    """
    if pressure is None or not -2 <= now-pressure["time"] <= 2:
        group.pop("weak_since", None)
        group.pop("weak_last_tick", None)
        return False
    t = pressure["time"]
    if t <= group.get("exit_last_tick", -1):
        return False
    group["exit_last_tick"] = t
    velocity = group["side"]*pressure["velocity"]
    group["peak_velocity"] = max(group.get("peak_velocity", 0.), velocity)
    group["peak_net"] = max(group.get("peak_net", 0.), net)
    risk = sum(c.get("risk_cash", 0.) for c in group["children"])
    noise = max(2*pressure["spread"], .10*group["entry_atr"]) / group["spec"]["tick_size"] * tick_cash
    arm = max(cfg.profit_arm_r*risk, noise, 3*tick_cash)
    giveback = max(cfg.giveback_fraction*group["peak_net"], .15*risk, noise)
    age = now-group.get("first_fill_at", group["created_at"])
    group["exit_diagnostics"] = {"policy": "confirmed", "age_seconds": age,
        "minimum_hold_seconds": cfg.min_hold_seconds, "risk_cash": risk,
        "arm_net_cash": arm, "peak_net_cash": group["peak_net"], "noise_cash": noise,
        "required_giveback_cash": giveback, "armed": risk > 0 and group["peak_net"] >= arm}
    faded = velocity <= .30*group["peak_velocity"]
    eligible = (risk > 0 and age >= cfg.min_hold_seconds and net > tick_cash
                and group["peak_net"] >= arm and group["peak_net"]-net >= giveback and faded)
    if eligible:
        if t-group.get("weak_last_tick", -1e30) > 2:
            group["weak_since"] = t
        group.setdefault("weak_since", t)
        group["weak_last_tick"] = t
        return t-group["weak_since"] >= cfg.confirmation_seconds
    group.pop("weak_since", None)
    group.pop("weak_last_tick", None)
    return False
