from __future__ import annotations

import time

import numpy as np
import pytest

from quant_rl.data.mt5_history import MT5History
from quant_rl.data.mt5_source import OHLCVBar
from tests._specs import default_symbol_spec


def test_freeze_consumes_m1_generator_and_preserves_evidence_arrays():
    bars = [OHLCVBar(1704067200 + 3600 * i, 1.1, 1.1002, 1.0998, 1.1, 100, 2.0)
            for i in range(8)]
    yielded = []

    def stream():
        for i in range(120):
            yielded.append(i)
            ts = 1704067200 + i * 60
            yield OHLCVBar(ts, 1.1, 1.1002, 1.0998, 1.1, 10, 1.5)

    frozen = MT5History().freeze(
        'EURUSD', 'H1', bars, default_symbol_spec('EURUSD'), m1_bars=stream(),
        fallback_spread_points=2.0,
    )
    assert len(yielded) == 120
    assert frozen.m1_time_s.dtype == np.int64
    assert frozen.m1_high.dtype == np.float64
    assert frozen.m1_spread_points.dtype == np.float64
    assert frozen.m1_time_s[0] == 1704067200
    assert frozen.m1_time_s[-1] == 1704067200 + 119 * 60


def test_freeze_checks_session_deadline_while_consuming_stream():
    bars = [OHLCVBar(1704067200 + 3600 * i, 1.1, 1.1002, 1.0998, 1.1, 100, 2.0)
            for i in range(8)]

    def stream():
        for i in range(100_001):
            ts = 1704067200 + i * 60
            yield OHLCVBar(ts, 1.1, 1.1002, 1.0998, 1.1, 10, 1.5)

    with pytest.raises(TimeoutError, match='deadline'):
        MT5History().freeze(
            'EURUSD', 'H1', bars, default_symbol_spec('EURUSD'), m1_bars=stream(),
            deadline_monotonic=time.monotonic() - 1,
        )



def test_freeze_rechecks_deadline_every_4096_rows(monkeypatch):
    import quant_rl.data.mt5_history as history_module

    bars = [OHLCVBar(1704067200 + 3600 * i, 1.1, 1.1002, 1.0998, 1.1, 100, 2.0)
            for i in range(8)]
    yielded = []

    def stream():
        for i in range(10_000):
            yielded.append(i)
            ts = 1704067200 + i * 60
            yield OHLCVBar(ts, 1.1, 1.1002, 1.0998, 1.1, 10, 1.5)

    # First check (row 0) is in time; the second check (row 4096) is late.
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(history_module.time, "monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="deadline"):
        MT5History().freeze(
            "EURUSD", "H1", bars, default_symbol_spec("EURUSD"),
            m1_bars=stream(), deadline_monotonic=1.0,
        )
    assert len(yielded) == 4097


def test_iter_m1_range_checks_deadline_after_blocking_mt5_request(monkeypatch):
    import quant_rl.data.mt5_source as source_module

    class Rates:
        iterated = False

        def __iter__(self):
            self.iterated = True
            return iter(())

    rates = Rates()

    class FakeMT5:
        TIMEFRAME_M1 = 1

        def symbol_select(self, symbol, enabled):
            return True

        def copy_rates_range(self, *args):
            return rates

        def last_error(self):
            return (0, "ok")

    from quant_rl.data.mt5_source import MT5DataSource

    source = MT5DataSource.__new__(MT5DataSource)
    source._mt5 = FakeMT5()

    # Before the blocking request the deadline is valid; immediately after it expires.
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(source_module.time, "monotonic", lambda: next(clock))

    stream = source.iter_m1_range(
        "EURUSD", 1704067200, 1704067260, deadline_monotonic=1.0
    )
    with pytest.raises(TimeoutError, match="deadline"):
        next(stream)
    assert rates.iterated is False


def test_fetch_preserves_timeout_error_and_closes_mt5(monkeypatch):
    from types import SimpleNamespace
    import quant_rl.data.mt5_history as history_module

    closed = []

    class FakeMT5:
        def account_info(self):
            return SimpleNamespace(currency="USD", server="Demo")

        def symbol_info(self, symbol):
            return None

        def symbol_info_tick(self, symbol):
            return None

    class FakeSource:
        def __init__(self, **kwargs):
            self._mt5 = FakeMT5()

        def connect(self):
            pass

        def get_symbol_spec(self, symbol, currency):
            return default_symbol_spec(symbol)

        def fetch_bars_since(self, symbol, timeframe, start):
            raise TimeoutError("synthetic session deadline")

        def close(self):
            closed.append(True)

    config = SimpleNamespace(
        broker_connection=SimpleNamespace(
            mt5_login=0, mt5_password="", mt5_server="", mt5_path=""
        ),
        research=SimpleNamespace(lookback_years=2),
    )
    monkeypatch.setattr(history_module, "MT5DataSource", FakeSource)

    with pytest.raises(TimeoutError, match="synthetic session deadline"):
        MT5History(config).fetch("EURUSD", deadline_monotonic=time.monotonic() + 60)
    assert closed == [True]



def test_ppo_fit_passes_deadline_to_training_and_validation_replays(monkeypatch):
    from types import SimpleNamespace
    from quant_rl.intelligence.policy import PPOTrader

    calls = []
    simulation = SimpleNamespace(decision_events=(), learning_reward_account_ccy=1.0)

    def replay(self, *args, **kwargs):
        calls.append(kwargs.get("deadline_monotonic"))
        return simulation, []

    monkeypatch.setattr(PPOTrader, "replay", replay)
    monkeypatch.setattr(PPOTrader, "snapshot", lambda self: {})
    monkeypatch.setattr(PPOTrader, "restore", lambda self, snapshot: None)

    trader = object.__new__(PPOTrader)
    trader.settings = SimpleNamespace(episodes=1, validation_interval_episodes=1, ppo_epochs=1)
    deadline = time.monotonic() + 60

    trader.fit(
        obs=None, spec=SimpleNamespace(name="EURUSD"), train=np.array([0]), val=np.array([0]),
        cfg=None, seed=7, deadline_monotonic=deadline,
    )

    assert calls == [deadline, deadline]



def test_freeze_checks_deadline_after_short_stream_finishes(monkeypatch):
    import quant_rl.data.mt5_history as history_module

    bars = [OHLCVBar(1704067200 + 3600 * i, 1.1, 1.1002, 1.0998, 1.1, 100, 2.0)
            for i in range(2)]

    def stream():
        for i in range(10):
            ts = 1704067200 + i * 60
            yield OHLCVBar(ts, 1.1, 1.1002, 1.0998, 1.1, 10, 1.5)

    clock = iter((0.0, 2.0))
    monkeypatch.setattr(history_module.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="deadline"):
        MT5History().freeze(
            "EURUSD", "H1", bars, default_symbol_spec("EURUSD"),
            m1_bars=stream(), deadline_monotonic=1.0,
        )


def test_iter_m1_range_checks_deadline_during_large_response_chunk(monkeypatch):
    import quant_rl.data.mt5_source as source_module
    from quant_rl.data.mt5_source import MT5DataSource

    start = 1704067200
    yielded = []

    def rates():
        for i in range(4097):
            yielded.append(i)
            yield {
                "time": start + i * 60, "open": 1.1, "high": 1.1002,
                "low": 1.0998, "close": 1.1, "tick_volume": 10, "spread": 2,
            }

    class FakeMT5:
        TIMEFRAME_M1 = 1

        def symbol_select(self, symbol, enabled):
            return True

        def copy_rates_range(self, *args):
            return rates()

        def last_error(self):
            return (0, "ok")

    source = MT5DataSource.__new__(MT5DataSource)
    source._mt5 = FakeMT5()
    # Before request, after request, first row, then row 4096 (expired).
    clock = iter((0.0, 0.0, 0.0, 2.0))
    monkeypatch.setattr(source_module.time, "monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="deadline"):
        list(source.iter_m1_range(
            "EURUSD", start, start + 4097 * 60, deadline_monotonic=1.0
        ))
    assert len(yielded) == 4097
