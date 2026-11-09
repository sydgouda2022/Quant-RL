"""Regression tests for rc21 streaming M1, coverage, FX and ledger anchoring."""
from __future__ import annotations
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest
import time

from quant_rl.data.mt5_source import OHLCVBar, MT5DataSource
from quant_rl.data.quality import DataQualityAuditor
from quant_rl.execution.loss_history import consecutive_loss_streak
from quant_rl.execution.live import LiveSession
from quant_rl.config.schema import QuantRLConfig
from tests._specs import default_symbol_spec


def bar(ts):
    return OHLCVBar(ts,1.20,1.21,1.19,1.205,3,1)


def broker_bar(ts):
    return dict(time=ts,open=1.20,high=1.21,low=1.19,close=1.205,tick_volume=3,spread=1)


class FakeM1:
    TIMEFRAME_M1=1
    def __init__(self, times):
        self.times=times
        self.windows=[]
    def symbol_select(self,symbol,enable):
        return True
    def copy_rates_range(self,symbol,tf,start,end):
        left=int(start.timestamp());right=int(end.timestamp())
        self.windows.append((left,right))
        return [broker_bar(t) for t in self.times if left<=t<=right]
    def last_error(self):return 'fake error'


def source(times):
    obj=MT5DataSource(); obj._mt5=FakeM1(times)
    return obj


def test_m1_boundary_windows_preserve_order_without_duplicates():
    fortnight=14*86400
    times=[fortnight-60,fortnight,fortnight+60,2*fortnight-60,2*fortnight]
    src=source(times)
    result=src.fetch_m1_range('EURUSD',0,2*fortnight+60)
    assert [r.time_s for r in result]==times
    assert len(src._mt5.windows)==3
    assert src._mt5.windows[0][1] == fortnight-1
    assert src._mt5.windows[1][0] == fortnight


def test_m1_rejects_duplicate_unordered_and_out_of_window():
    fortnight=14*86400
    for rows in ([fortnight,fortnight], [fortnight+60,fortnight], [59]):
        src=source(rows)
        with pytest.raises(ValueError,match='misaligned|repeated|unordered'):
            src.fetch_m1_range('EURUSD',0,fortnight+120)
    src=source([])
    src._mt5.copy_rates_range=lambda *args:[broker_bar(5*fortnight)]
    with pytest.raises(ValueError,match='out-of-window'):
        src.fetch_m1_range('EURUSD',0,fortnight)


def test_m1_auditor_exact_coverage_handles_missing_and_trailing_minutes():
    bars=[bar(0),bar(3600)]
    complete=np.arange(0,7200,60,dtype=np.int64)
    params=dict(symbol='EURUSD',timeframe='H1',bars=bars,bar_price_side='BID')
    full=DataQualityAuditor.audit_bars(**params,m1_time_s=complete)
    assert full.m1_intrabar_verified_bars==2
    missing=complete[complete!=1800]
    result=DataQualityAuditor.audit_bars(**params,m1_time_s=missing)
    assert result.m1_intrabar_verified_bars==1
    with pytest.raises(ValueError,match='Unsorted|duplicated'):
        DataQualityAuditor.audit_bars(**params,m1_time_s=np.array([60,0,120],dtype=np.int64))


def test_m1_auditor_large_index_requires_no_synthetic_minutes():
    # Sparse minute evidence spanning years must never count as a full H1.
    times=np.arange(0,4*365*86400,3600,dtype=np.int64)
    result=DataQualityAuditor.audit_bars(
        'EURUSD','H1',[bar(0),bar(3600)],m1_time_s=times,
        bar_price_side='BID')
    assert result.m1_intrabar_verified_bars==0


def test_fx_conversion_selects_real_tradable_symbol_and_rejects_stale(monkeypatch):
    now=1_900_000_000
    monkeypatch.setattr(time,'time',lambda: now)
    calls=[]
    class Terminal:
        def symbol_select(self,s,enabled):
            calls.append(s)
            return s=='USDCAD'
        def symbol_info_tick(self,s):
            return NS(bid=1.33,ask=1.34,time=now)
    session=LiveSession.__new__(LiveSession)
    session.cfg=QuantRLConfig()
    session.broker=NS(_mt5=Terminal())
    spec=default_symbol_spec('EURUSD')
    rates=session._fx_rates(spec,NS(currency='CAD'),NS(bid=1.12,ask=1.13))
    assert rates['USDCAD']==pytest.approx(1.335)
    assert rates['EURCAD']==pytest.approx(1.125*1.335)
    assert 'USDCAD' in calls
    session.broker._mt5.symbol_info_tick=lambda s:NS(bid=1.33,ask=1.34,time=now-1000)
    with pytest.raises(ValueError,match='Missing broker conversion'):
        session._fx_rates(spec,NS(currency='CAD'),NS(bid=1.12,ask=1.13))


def test_live_rereads_conversion_for_final_safety_check(tmp_path,monkeypatch):
    from tests.test_live_mt5_unified import FakeBroker,setup_session
    from tests.test_rc20_loss_blackout import _freeze_open_broker_candle
    _freeze_open_broker_candle(monkeypatch)
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    calls=[]
    original=session._fx_rates
    def spy(spec,account,tick):
        calls.append(float(tick.ask))
        return original(spec,account,tick)
    monkeypatch.setattr(session,'_fx_rates',spy)
    outcome=session.step()
    assert outcome['status'] in ('FILLED','VETOED','VETOED_REFRESH','VETOED_BROKER_CALC','QUOTE_MOVED')
    assert len(calls)==2


def test_loss_anchor_detects_corrected_broker_cash_on_restart(tmp_path):
    from tests.test_rc20_loss_blackout import Broker,close,streak
    broker=Broker();broker.deals=[close(11,1700000000,-10)]
    assert streak(broker,tmp_path)==1
    broker.deals=[close(11,1700000000,-20)]
    with pytest.raises(RuntimeError,match='exit cash changed'):
        streak(broker,tmp_path)


def test_old_loss_certificate_requires_reverification_not_implicit_migration(tmp_path):
    from tests.test_rc20_loss_blackout import Broker,close,streak
    broker=Broker();broker.deals=[close(11,1700000000,-10)]
    assert streak(broker,tmp_path)==1
    state=next((tmp_path/'live_state').glob('*loss_history.json'))
    record=json.loads(state.read_text());record['version']=1
    state.write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='identity mismatch'):
        streak(broker,tmp_path)
