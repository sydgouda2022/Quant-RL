"""Phase-1 safety regressions; all MT5 access is mocked or absent."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from quant_rl.config.schema import QuantRLConfig
from quant_rl.data.mt5_source import account_is_demo, broker_tick_age
from quant_rl.research.evolution import is_qualified


def test_data_and_config_modules_import_independently_in_fresh_processes():
    root = Path(__file__).resolve().parents[1]
    for module in (
        "quant_rl.data.mt5_source",
        "quant_rl.data.time_alignment",
        "quant_rl.config.logger",
        "quant_rl.config.schema",
    ):
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}; print('ok')"],
            cwd=root, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, f"{module}: {result.stderr}"
        assert result.stdout.strip() == "ok"


def test_weekly_schedule_reason_is_forwarded_to_research(monkeypatch, tmp_path):
    from quant_rl.routing import RoutingDecision, run_automatic_session
    import quant_rl.routing as routing
    import quant_rl.research.evolution as evolution

    cfg = QuantRLConfig.model_validate({
        "project_root": str(tmp_path),
        "research": {"symbols": ["EURUSD"]},
        "broker_connection": {"mt5_server_timezone": "UTC"},
    })
    decision = RoutingDecision("RESEARCH", "WEEKLY_SCHEDULE", ("EURUSD",), False)
    monkeypatch.setattr(routing, "choose_mode", lambda _: (decision, {}, {}))
    received = {}
    def fake_research(config, *, targets, preloaded, reason):
        received.update(targets=targets, preloaded=preloaded, reason=reason)
        return {"status": "RESEARCH_STARTED"}
    monkeypatch.setattr(evolution, "run_research_session", fake_research)
    assert run_automatic_session(cfg) == {"status": "RESEARCH_STARTED"}
    assert received == {"targets": ("EURUSD",), "preloaded": {}, "reason": "WEEKLY_SCHEDULE"}


def test_demo_detection_fails_closed_for_unknown_account_mode():
    assert account_is_demo(NS(trade_mode=0))
    assert not account_is_demo(NS(trade_mode=2))
    assert not account_is_demo(NS())


def test_mt5_market_timestamps_are_not_shifted_by_broker_timezone():
    now = 1_800_000_000
    assert broker_tick_age(now, now - 37) == 37
    assert broker_tick_age(now, now + 37) == -37


def test_incumbent_and_challenger_share_m1_evidence_qualification():
    cfg = QuantRLConfig()
    sim = NS(
        time_s_curve=[0, 100 * 86400], executed_trades=[object()],
        trades_per_year=20, net_return_pct=2., max_drawdown_pct=5.,
    )
    stress = NS(net_return_pct=1., max_drawdown_pct=6.)
    ok, reasons = is_qualified(
        sim, stress, cfg, "MT5_TERMINAL",
        m1_price_coverage_fraction=.90, m1_spread_coverage_fraction=.90,
    )
    assert ok and not reasons
    ok, reasons = is_qualified(
        sim, stress, cfg, "MT5_TERMINAL",
        m1_price_coverage_fraction=.90, m1_spread_coverage_fraction=.50,
    )
    assert not ok and "INSUFFICIENT_M1_SPREAD_EVIDENCE" in reasons
    ok, reasons = is_qualified(
        sim, stress, cfg, "MT5_TERMINAL",
        m1_price_coverage_fraction=.50, m1_spread_coverage_fraction=.90,
    )
    assert not ok and "INSUFFICIENT_M1_PRICE_PATH_EVIDENCE" in reasons


def test_unknown_or_funded_account_cannot_enter_live_by_default():
    cfg = QuantRLConfig()
    assert cfg.live.allow_non_demo_account is False


def test_mt5_broker_adapter_blocks_funded_account_without_explicit_opt_in():
    from quant_rl.execution.mt5_broker import MT5Broker

    cfg = QuantRLConfig()
    terminal = NS(connected=True, trade_allowed=True)
    funded = NS(login=42, server='Broker-Live', currency='USD', trade_mode=2,
                trade_allowed=True, equity=10000.)
    broker = MT5Broker(cfg.broker_connection)
    broker._mt5 = NS(terminal_info=lambda: terminal, account_info=lambda: funded)
    with pytest.raises(PermissionError, match='Demo-only'):
        broker.account()

    opted_in = MT5Broker(cfg.broker_connection, allow_non_demo_account=True)
    opted_in._mt5 = NS(terminal_info=lambda: terminal, account_info=lambda: funded)
    assert opted_in.account() is funded


def test_mt5_broker_adapter_blocks_unknown_account_mode_even_with_opt_in():
    from quant_rl.execution.mt5_broker import MT5Broker

    cfg = QuantRLConfig()
    terminal = NS(connected=True, trade_allowed=True)
    unknown = NS(login=42, server='Unknown', currency='USD',
                 trade_allowed=True, equity=10000.)
    broker = MT5Broker(cfg.broker_connection, allow_non_demo_account=True)
    broker._mt5 = NS(terminal_info=lambda: terminal, account_info=lambda: unknown)
    with pytest.raises(PermissionError, match='mode unavailable'):
        broker.account()


def test_mt5_h1_and_m1_history_requests_use_utc_bounds_and_raw_utc_rows(monkeypatch):
    from datetime import datetime, timezone
    import numpy as np
    from quant_rl.data.mt5_source import MT5DataSource

    start = 1_704_067_200
    now = start + 10 * 3600
    monkeypatch.setattr('quant_rl.data.mt5_source.time.time', lambda: now)
    calls = []
    rates = np.array([
        (start, 1.1, 1.2, 1.0, 1.1, 100, 3),
        (start + 3600, 1.1, 1.2, 1.0, 1.1, 110, 4),
    ], dtype=[('time','i8'),('open','f8'),('high','f8'),('low','f8'),
              ('close','f8'),('tick_volume','i8'),('spread','i8')])

    class API:
        TIMEFRAME_H1 = 1
        TIMEFRAME_M1 = 2
        def symbol_select(self, symbol, visible): return True
        def copy_rates_range(self, symbol, timeframe, start_dt, end_dt):
            calls.append((timeframe, start_dt, end_dt))
            return rates if timeframe == self.TIMEFRAME_H1 else rates[:1]
        def last_error(self): return (0, 'ok')

    source = MT5DataSource()
    source._mt5 = API()
    bars = source.fetch_bars_since('EURUSD', 'H1', start)
    assert [bar.time_s for bar in bars] == [start, start + 3600]
    assert calls[0][1] == datetime.fromtimestamp(start, timezone.utc)
    assert calls[0][2] == datetime.fromtimestamp(now, timezone.utc)

    m1 = source.fetch_m1_range('EURUSD', start, start + 120)
    assert [bar.time_s for bar in m1] == [start]
    assert calls[1][1] == datetime.fromtimestamp(start, timezone.utc)
    assert calls[1][2] == datetime.fromtimestamp(start + 119, timezone.utc)


def test_mt5_deal_history_requests_use_utc_datetimes(monkeypatch):
    from datetime import datetime, timezone
    from quant_rl.execution.mt5_broker import MT5Broker

    start = 1_704_067_200
    now = start + 86_400
    monkeypatch.setattr('quant_rl.execution.mt5_broker.time.time', lambda: now)
    seen = []
    class API:
        def history_deals_get(self, start_dt, end_dt):
            seen.append((start_dt, end_dt))
            return []
    broker = MT5Broker(QuantRLConfig().broker_connection)
    broker._mt5 = API()
    assert broker.recent_deals(start) == []
    assert seen == [(datetime.fromtimestamp(start, timezone.utc),
                     datetime.fromtimestamp(now, timezone.utc))]
