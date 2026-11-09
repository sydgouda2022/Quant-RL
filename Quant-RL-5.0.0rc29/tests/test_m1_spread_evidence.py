"""rc8: chronological minute-spread evidence in SL/TP sequencing."""
from dataclasses import replace
import numpy as np
import pytest

from quant_rl.data.mt5_history import MT5History
from quant_rl.data.mt5_source import OHLCVBar
from tests._specs import default_symbol_spec
from quant_rl.execution.accounting import PortfolioAccountingSimulator
from quant_rl.contracts import TradeDirection

T=1704067200

def sell_dual_touch(**kw):
    return PortfolioAccountingSimulator._resolve_dual_touch_m1(
        direction=TradeDirection.SELL, stop_price=1.1006, target_price=1.0996,
        half_spread=0.00001, bar_price_side='BID',
        bar_start_s=T, bar_end_s=T+120,
        m1_time_s=np.array([T,T+60],dtype=np.int64),
        m1_high=np.array([1.1001,1.1006]),
        m1_low=np.array([1.0995,1.1000]),
        **kw)

def test_sells_use_actual_minute_spread_not_last_h4_spread():
    # With narrow H4 spread we would incorrectly record TP at minute zero.
    assert sell_dual_touch() == 'TP'
    # M1 spread is wide at minute zero (no TP) and narrow later (SL hit).
    assert sell_dual_touch(m1_spread_points=np.array([20.,2.]),point=0.00001) == 'SL'

def test_spread_stress_affects_minute_ask_path():
    # With low spread, minute zero reaches SELL TP. Stressed costs push ask up.
    spread=np.array([2.,2.])
    assert sell_dual_touch(m1_spread_points=spread,point=0.00001) == 'TP'
    assert sell_dual_touch(m1_spread_points=spread,point=0.00001,cost_multiplier=10) == 'SL'

def test_missing_or_corrupt_minute_spread_cannot_create_favorable_exit():
    for spreads in [np.array([np.nan,2.]),np.array([0.,2.]),np.array([-1.,2.]),np.array([2.])]:
        assert sell_dual_touch(m1_spread_points=spreads,point=0.00001) == 'AMBIGUOUS_SL'

def test_missing_preceding_minute_never_inferrs_tp():
    assert PortfolioAccountingSimulator._resolve_dual_touch_m1(
        direction=TradeDirection.SELL, stop_price=1.1006, target_price=1.0996,
        half_spread=.00001, bar_price_side='BID',bar_start_s=T,bar_end_s=T+120,
        m1_time_s=np.array([T+60],dtype=np.int64),
        m1_high=np.array([1.1001]),m1_low=np.array([1.0995]),
        m1_spread_points=np.array([2.]),point=.00001) == 'AMBIGUOUS_SL'

def test_m1_spread_changes_content_fingerprint_and_manifest(tmp_path):
    store=MT5History()
    spec=default_symbol_spec('EURUSD')
    h1=[OHLCVBar(T+j*3600,1.1,1.1008,1.0994,1.1,100,2.) for j in range(4)]
    m1=[OHLCVBar(T+j*60,1.1,1.1002,1.0998,1.1,100,2.) for j in range(4)]
    a=store.freeze('EURUSD','H1',h1,spec,m1_bars=m1)
    b=store.freeze('EURUSD','H1',h1,spec,m1_bars=[*m1[:2], replace(m1[2],spread_points=5.),m1[3]])
    assert a.fingerprint != b.fingerprint
    assert list(b.m1_spread_points) == [2.,2.,5.,2.]
    assert a.manifest.m1_spread_coverage_fraction == 1.0
    c=store.freeze('EURUSD','H1',h1,spec,m1_bars=[replace(m1[0],spread_points=None),*m1[1:]])
    assert c.manifest.m1_spread_coverage_fraction == 0.75

def test_invalid_m1_price_and_timestamp_rejected(tmp_path):
    store=MT5History()
    spec=default_symbol_spec('EURUSD')
    h1=[OHLCVBar(T+j*3600,1.1,1.1005,1.0995,1.1,100,2.) for j in range(3)]
    minute=OHLCVBar(T,1.1,1.1002,1.0998,1.1,10,2.)
    for data in [[minute,minute],[replace(minute,high=1.099)],[replace(minute,spread_points=-1.)]]:
        with pytest.raises(ValueError,match='Invalid M1'):
            store.freeze('EURUSD','H1',h1,spec,m1_bars=data)

def test_m1_covered_bar_count_is_not_minute_count(tmp_path):
    store=MT5History()
    spec=default_symbol_spec('EURUSD')
    h1=[OHLCVBar(T+j*3600,1.1,1.1005,1.0995,1.1,100,2.) for j in range(2)]
    m1=[OHLCVBar(T+j*60,1.1,1.1002,1.0998,1.1,10,2.) for j in range(60)]
    result=store.freeze('EURUSD','H1',h1,spec,m1_bars=m1)
    assert result.manifest.m1_intrabar_verified_bars == 1
    assert result.manifest.m1_window_coverage_fraction == 0.5
