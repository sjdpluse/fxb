"""Capture closed broker bars for review without guessing history time or costs.

Works with an installed FXB 0.2.2 environment, including when this script is
downloaded to a temporary directory. No terminal credentials or orders are
written to the archive. The raw CSVs are intentionally not FXB training CSVs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import uuid
import zipfile

import numpy as np

from fxb import __version__
from fxb.mt5 import MT5, Blocked


RATE_FIELDS = ("time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume")
CSV_FIELDS = ("mt5_time_raw", "bid_open", "bid_high", "bid_low", "bid_close",
              "tick_volume", "spread_points", "real_volume")
SPEC_FIELDS = (
    "name", "description", "custom", "currency_base", "currency_profit", "currency_margin",
    "digits", "point", "trade_tick_size", "trade_tick_value_profit", "trade_tick_value_loss",
    "trade_contract_size", "trade_calc_mode", "chart_mode", "trade_mode", "trade_exemode",
    "volume_min", "volume_max", "volume_step", "volume_limit", "trade_stops_level",
    "trade_freeze_level", "filling_mode", "order_mode", "swap_mode", "swap_long", "swap_short",
    "swap_rollover3days", "swap_sunday", "swap_monday", "swap_tuesday", "swap_wednesday",
    "swap_thursday", "swap_friday", "swap_saturday",
)


def utc_iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def validate_rates(rates, requested):
    if rates is None or len(rates) < min(requested, 800):
        raise Blocked("Insufficient closed M5 history; open the symbol's M5 chart and load history in MT5")
    if rates.dtype.names is None or not set(RATE_FIELDS).issubset(rates.dtype.names):
        raise Blocked("Unexpected MT5 bar fields")
    if any(not np.isfinite(rates[field]).all() for field in RATE_FIELDS):
        raise Blocked("Non-finite values in broker history")
    if (np.any(rates["time"] <= 0) or np.any(np.diff(rates["time"]) <= 0)
            or np.any(rates["time"] % 300 != 0)):
        raise Blocked("Unordered, duplicate or unaligned M5 bar times; history was not changed")
    if (np.any(rates["low"] <= 0)
            or np.any(rates["low"] > np.minimum(rates["open"], rates["close"]))
            or np.any(rates["high"] < np.maximum(rates["open"], rates["close"]))
            or any(np.any(rates[f] < 0) for f in ("tick_volume", "spread", "real_volume"))):
        raise Blocked("Invalid broker OHLC, spread or volume")


def export_market_data(root, symbols, bars=20000, bridge_factory=MT5):
    root = Path(root).resolve()
    if type(bars) is not int or not 800 <= bars <= 100000:
        raise ValueError("Choose between 800 and 100000 closed bars per symbol")
    if not 1 <= len(symbols) <= 8 or len(set(symbols)) != len(symbols):
        raise ValueError("Choose one to eight distinct exact broker symbols")
    if any(not isinstance(s, str) or not s.strip() for s in symbols):
        raise ValueError("Empty broker symbol")
    config = json.loads((root / "state" / "workstation.json").read_text(encoding="utf-8-sig"))
    if not config.get("login") or not config.get("server"):
        raise ValueError("Save the exact account and server in FXB first")
    bridge = bridge_factory(terminal=config.get("terminal") or None, clock=config.get("broker_clock"))
    artifacts = {}
    manifest = {
        "schema": "fxb-raw-market-data-v1", "fxb_version": __version__,
        "capture_started_utc": utc_iso(time.time()), "timeframe": "M5",
        "history_time_encoding": "raw MT5 transport values; not relabelled as UTC",
        "history_normalized": False, "current_clock_policy": bridge.clock.config(),
        "training_csv": False, "symbols": [], "orders_sent": 0,
        "required_before_training": ["validated history clock rule and date coverage",
                                     "explicit commission, slippage and rollover cost assumptions"],
    }
    try:
        bridge.connect()
        account = bridge.account(config["login"], expected_server=config["server"])
        manifest["account"] = {"server": account.server, "currency": account.currency,
                               "trade_mode": account.trade_mode, "configured_account_matches": True}
        manifest["mt5_package"] = getattr(bridge.api, "__version__", None)
        for index, symbol in enumerate(symbols, 1):
            bridge.account(config["login"], expected_server=config["server"])
            info = bridge.symbol(symbol)
            quote = bridge.clock_snapshot(symbol)
            if not quote["ready"]:
                raise Blocked(f"{symbol}: {quote.get('reason', 'Quote time check failed')}")
            rates = bridge.api.copy_rates_from_pos(symbol, bridge.api.TIMEFRAME_M5, 1, bars)
            validate_rates(rates, bars)
            # Only the latest closed bar is checked against the current clock
            # policy. Older raw timestamps never receive today's fixed offset.
            closed_at = bridge.utc(float(rates["time"][-1])) + 300
            if not 0 <= time.time() - closed_at < 300:
                raise Blocked(f"{symbol}: closed M5 history has not caught up with the current quote")
            bridge.account(config["login"], expected_server=config["server"])
            tick = bridge.tick(symbol)
            ref_lots = max(info.volume_min, min(1.0, info.volume_max))
            margin = bridge.api.order_calc_margin(bridge.api.ORDER_TYPE_BUY, symbol, ref_lots,
                                                  tick.ask)
            if margin is not None and (not math.isfinite(margin) or margin <= 0):
                margin = None
            text = io.StringIO(newline="")
            writer = csv.writer(text, lineterminator="\n")
            writer.writerow(CSV_FIELDS)
            writer.writerows(tuple(row[field].item() for field in RATE_FIELDS) for row in rates)
            payload = text.getvalue().encode("utf-8")
            name = f"{index:02d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', symbol)}_M5_raw.csv"
            artifacts[name] = payload
            gaps = np.diff(rates["time"])
            manifest["symbols"].append({
                "symbol": symbol, "file": name, "sha256": hashlib.sha256(payload).hexdigest(),
                "requested_closed_bars": bars, "returned_closed_bars": len(rates),
                "first_time_raw": int(rates["time"][0]), "last_time_raw": int(rates["time"][-1]),
                "last_closed_at_utc": utc_iso(closed_at), "quote_clock_check": quote,
                "bid_snapshot": tick.bid, "ask_snapshot": tick.ask,
                "snapshot_quote_utc": utc_iso(tick.time_msc / 1000),
                "specification": {k: getattr(info, k, None) for k in SPEC_FIELDS},
                "margin_snapshot": {"direction": "BUY", "volume": ref_lots,
                                    "price": tick.ask, "amount_in_account_currency": margin,
                                    "status": "available" if margin is not None else "unavailable"},
                "gaps_over_five_minutes": int(np.count_nonzero(gaps > 300)),
                "largest_raw_gap_seconds": int(gaps.max()),
                "cost_inputs": {"commission_per_lot_side": None, "slippage_points": None,
                                "overnight_cost_per_lot": None, "status": "not_assumed"},
            })
        bridge.account(config["login"], expected_server=config["server"])
        manifest["capture_finished_utc"] = utc_iso(time.time())
    finally:
        bridge.close()

    # Serialize fully before writing. Failed reads leave no partial ZIP that
    # could be mistaken for a successful capture; existing files stay intact.
    manifest_bytes = json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8")
    destination = root / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    name = "fxb-market-data-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = destination / f"{name}-{uuid.uuid4().hex[:8]}.zip"
    temporary = target.with_suffix(".zip.part")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest_bytes)
            for name, payload in artifacts.items():
                archive.writestr(name, payload)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"status": "saved", "archive": str(target), "orders_sent": 0,
            "symbols": [{"symbol": s["symbol"], "closed_bars": s["returned_closed_bars"]}
                        for s in manifest["symbols"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--bars", type=int, default=20000)
    args = parser.parse_args()
    try:
        result = export_market_data(args.root, args.symbols, args.bars)
    except (Blocked, ValueError, OSError) as error:
        parser.exit(1, f"Market export stopped: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
