"""Read-only MT5 environment preflight. Never submits, modifies, or closes trades."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _safe(value):
    if value is None:
        return None
    if hasattr(value, "_asdict"):
        return {k: _safe(v) for k, v in value._asdict().items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    symbols = cfg.get("research", {}).get("symbols", [])
    report = {
        "tool": "Quant-RL MT5 read-only preflight",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "trade_operations_performed": False,
        "configured_server_timezone": cfg.get("broker_connection", {}).get("mt5_server_timezone"),
        "symbols": {},
    }

    if not mt5.initialize():
        report["initialize_error"] = _safe(mt5.last_error())
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        report["terminal"] = {
            "connected": getattr(terminal, "connected", None),
            "trade_allowed": getattr(terminal, "trade_allowed", None),
            "build": getattr(terminal, "build", None),
        } if terminal else None
        if account:
            server = str(getattr(account, "server", ""))
            report["account"] = {
                "login_masked": ("***" + str(account.login)[-3:]) if getattr(account, "login", None) else None,
                "server": server,
                "currency": getattr(account, "currency", None),
                "trade_allowed": getattr(account, "trade_allowed", None),
                "trade_expert": getattr(account, "trade_expert", None),
                "trade_mode": getattr(account, "trade_mode", None),
                "account_is_demo": (
                    int(getattr(account, "trade_mode", -1)) == int(mt5.ACCOUNT_TRADE_MODE_DEMO)
                    if getattr(account, "trade_mode", None) is not None else False
                ),
                "demo_server_name_hint": any(x in server.lower() for x in ("demo", "practice", "trial")),
            }
        else:
            report["account_error"] = _safe(mt5.last_error())

        for symbol in symbols:
            item = {"selected": False}
            if not mt5.symbol_select(symbol, True):
                item["symbol_select_error"] = _safe(mt5.last_error())
                report["symbols"][symbol] = item
                continue
            item["selected"] = True
            info = mt5.symbol_info(symbol)
            tick = mt5.symbol_info_tick(symbol)
            if info:
                item["spec"] = {
                    "visible": getattr(info, "visible", None),
                    "digits": getattr(info, "digits", None),
                    "point": getattr(info, "point", None),
                    "trade_tick_size": getattr(info, "trade_tick_size", None),
                    "trade_tick_value": getattr(info, "trade_tick_value", None),
                    "volume_min": getattr(info, "volume_min", None),
                    "volume_max": getattr(info, "volume_max", None),
                    "volume_step": getattr(info, "volume_step", None),
                    "trade_mode": getattr(info, "trade_mode", None),
                    "filling_mode": getattr(info, "filling_mode", None),
                }
            if tick:
                tick_epoch = getattr(tick, "time", None)
                item["tick"] = {
                    "bid": getattr(tick, "bid", None),
                    "ask": getattr(tick, "ask", None),
                    "time_utc": datetime.fromtimestamp(tick_epoch, timezone.utc).isoformat() if tick_epoch else None,
                    "age_seconds_approx": max(0, int(datetime.now(timezone.utc).timestamp() - tick_epoch)) if tick_epoch else None,
                }
            else:
                item["tick_error"] = _safe(mt5.last_error())

            item["history_smoke"] = {}
            for label, timeframe, count in (("H1", mt5.TIMEFRAME_H1, 100), ("M1", mt5.TIMEFRAME_M1, 1000)):
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, count)  # completed bars only
                item["history_smoke"][label] = {
                    "requested_completed_bars": count,
                    "returned_bars": len(rates) if rates is not None else 0,
                    "error": _safe(mt5.last_error()) if rates is None else None,
                }
            report["symbols"][symbol] = item

        report["notes"] = [
            "Read-only checks only; no order_check/order_send/position modification/close was called.",
            "Account mode is read from MT5 trade_mode; a server-name Demo/Practice/Trial match is only a hint.",
            "The configured IANA server timezone must be independently verified before LIVE; null blocks LIVE.",
            "This smoke test does not certify four-year history coverage, DST/session rules, execution costs, or order recovery.",
        ]
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if account and terminal and terminal.connected else 3
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
