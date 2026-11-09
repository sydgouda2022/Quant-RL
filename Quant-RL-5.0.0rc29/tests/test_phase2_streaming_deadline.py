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
