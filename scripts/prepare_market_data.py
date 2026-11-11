"""Validate a raw FXB archive and prepare UTC training data with explicit assumptions.

Offline only: no MetaTrader import, connection, account changes or orders.
Clock and cost evidence are carried into dataset hashes and model reports.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np

from fxb.clock import BrokerClock, instant
from fxb.data import Bars, KABUL, iso, load_bars, save_bars
from fxb.risk import Spec
from fxb.rollover import Rollover


FIELDS = ("mt5_time_raw", "bid_open", "bid_high", "bid_low", "bid_close",
          "tick_volume", "spread_points", "real_volume")


def prepare_market_data(archive_path, profile, output):
    archive_path, output = Path(archive_path), Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if profile.get("schema") != "fxb-market-preparation-v1":
        raise ValueError("Unsupported preparation profile")
    clock = BrokerClock.from_config(profile["broker_clock"])
    evidence = profile.get("clock_evidence")
    if not isinstance(evidence, dict) or not evidence.get("status") or not evidence.get("description"):
        raise ValueError("Document clock evidence or the explicit historical assumption")
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    prepared, audit = [], []
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        if (len(entries) > 40 or len({x.filename for x in entries}) != len(entries)
                or sum(x.file_size for x in entries) > 50 * 1024 * 1024):
            raise ValueError("Duplicate members or oversized market archive")
        manifest = json.loads(archive.read("manifest.json"))
        if (manifest.get("schema") != "fxb-raw-market-data-v1" or manifest.get("timeframe") != "M5"
                or manifest.get("history_normalized") is not False
                or manifest.get("training_csv") is not False):
            raise ValueError("Expected an unmodified raw M5 market archive")
        symbols = [row["symbol"] for row in manifest["symbols"]]
        if not symbols or len(set(symbols)) != len(symbols) or set(profile["costs"]) != set(symbols):
            raise ValueError("Provide exactly one cost profile for every archive symbol")
        for number, item in enumerate(manifest["symbols"], 1):
            symbol = item["symbol"]
            payload = archive.read(item["file"])
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"{symbol}: CSV SHA-256 mismatch")
            reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
            if tuple(reader.fieldnames or []) != FIELDS:
                raise ValueError(f"{symbol}: unexpected CSV fields")
            rows = list(reader)
            values = np.asarray([[float(row[key]) for key in FIELDS] for row in rows])
            if (len(rows) < 800 or len(rows) != item["returned_closed_bars"]
                    or values.shape != (len(rows), len(FIELDS)) or not np.isfinite(values).all()):
                raise ValueError(f"{symbol}: insufficient, malformed or nonfinite history")
            raw = values[:, 0]
            if (np.any(np.diff(raw) <= 0) or np.any(raw % 300) or raw[0] != item["first_time_raw"]
                    or raw[-1] != item["last_time_raw"] or np.any(values[:, 5:] < 0)):
                raise ValueError(f"{symbol}: invalid raw chronology or volumes")
            converted = np.asarray([clock.to_utc(value) for value in raw])
            quote = item["quote_clock_check"]
            if (not quote.get("ready") or abs(clock.to_utc(quote["raw_time_msc"] / 1000)
                    - instant(quote["quote_utc"])) > 0.001
                    or converted[-1] + 300 != instant(item["last_closed_at_utc"])):
                raise ValueError(f"{symbol}: history clock disagrees with the captured UTC quote")
            source = item["specification"]
            if source["chart_mode"] != 0 or source["custom"]:
                raise ValueError(f"{symbol}: expected broker bid OHLC, not a custom/last-price chart")
            if not np.isclose(source["trade_tick_value_profit"], source["trade_tick_value_loss"], rtol=1e-6):
                raise ValueError(f"{symbol}: asymmetric tick values need a different P&L model")
            margin = item["margin_snapshot"]
            if margin["status"] != "available" or margin["volume"] <= 0:
                raise ValueError(f"{symbol}: no usable margin snapshot")
            costs = profile["costs"][symbol]
            if not costs.get("evidence"):
                raise ValueError(f"{symbol}: document cost assumptions")
            rollover = Rollover.from_config(costs["rollover"]).config()
            spec = Spec(symbol, manifest["account"]["currency"], 300,
                        source["point"], source["trade_tick_size"], source["trade_tick_value_loss"],
                        source["volume_min"], source["volume_max"], source["volume_step"],
                        margin["amount_in_account_currency"] / margin["volume"],
                        costs["commission_per_lot_side"], costs["slippage_points"],
                        costs["overnight_cost_per_lot"], source["trade_stops_level"] * source["point"])
            assumptions = {"archive_sha256": archive_hash, "raw_csv_sha256": item["sha256"],
                           "clock_evidence": evidence, "cost_evidence": costs["evidence"],
                           "contract_snapshot_utc": item["snapshot_quote_utc"],
                           "margin": "fixed BUY snapshot in account currency; not historical or direction-specific margin"}
            bars = Bars(converted, *[values[:, i] for i in range(1, 6)], values[:, 6] * spec.point,
                        spec, f"MT5/{manifest['account']['server']}/{symbol}/raw-archive", False,
                        {"storage": "UTC", "broker_clock": clock.config()}, rollover, assumptions).validate()
            gaps = np.diff(converted)
            gap_list = [{"after_utc": iso(converted[i]), "next_utc": iso(converted[i + 1]),
                         "seconds": int(gaps[i])} for i in np.flatnonzero(gaps > 300)]
            name = f"{number:02d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', symbol)}_M5.csv"
            prepared.append((name, bars))
            audit.append({"symbol": symbol, "file": name, "closed_bars": len(rows),
                          "first_open_utc": iso(converted[0]), "last_close_utc": iso(converted[-1] + 300),
                          "first_open_kabul": datetime.fromtimestamp(converted[0], KABUL).isoformat(),
                          "last_close_kabul": datetime.fromtimestamp(converted[-1] + 300, KABUL).isoformat(),
                          "spread_price_min_median_p95_max": np.quantile(bars.spread, [0, .5, .95, 1]).tolist(),
                          "gap_count": len(gap_list), "gaps": gap_list,
                          "dataset_sha256": bars.digest(), "raw_sha256_verified": True,
                          "rollover": rollover, "assumptions": assumptions})
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "fxb-market-audit-v1", "archive_sha256": archive_hash,
              "clock_evidence": evidence, "profile": profile, "symbols": audit,
              "orders_sent": 0, "training_started": False, "profitability_established": False,
              "limitations": ["M5 spread observations are not historical intrabar Ask quotes",
                              "Clock and cost assumptions are not independently confirmed broker policies",
                              "Gaps are preserved; no missing prices are invented"]}
    with tempfile.TemporaryDirectory(prefix=".fxb-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        for name, bars in prepared:
            save_bars(bars, stage / name)
            if load_bars(stage / name).digest() != bars.digest():
                raise ValueError("Prepared dataset round-trip mismatch")
        (stage / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        (stage / "profile.json").write_text(json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
        stage.rename(output)
    return {"status": "prepared", "output": str(output), "orders_sent": 0,
            "symbols": [{"symbol": row["symbol"], "file": row["file"], "closed_bars": row["closed_bars"]}
                        for row in audit]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        profile = json.loads(Path(args.profile).read_text(encoding="utf-8-sig"))
        result = prepare_market_data(args.archive, profile, args.out)
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Market preparation stopped: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
