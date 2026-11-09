"""Offline broker mocks only. Live trading stays disallowed without quote freshness."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta, time as wall_time
from types import SimpleNamespace as NS

import pytest

from quant_rl.config.schema import QuantRLConfig
from quant_rl.data.mt5_history import MT5History
from quant_rl.data.mt5_source import OHLCVBar, broker_tick_age
from quant_rl.routing import choose_mode
from tests.test_auto_mt5_routing import FakeAPI, FakeSource, FakeStore, FakePopulation, make_cfg

NOW = int(datetime(2026,10,30,8,40,tzinfo=timezone.utc).timestamp())
OFFSET = 7200

@pytest.fixture(autouse=True)
def stable_decision_clock(monkeypatch):
    # LIVE fixtures need the synthetic last H4 candle to have closed just now.
    # Importing setup_session from another test module does not import its
    # autouse fixture, so freeze this module's clock independently.
    import time
    now=datetime.now(timezone.utc)
    friday=now.date()-timedelta(days=(now.weekday()-4)%7)
    close=datetime.combine(friday,wall_time(20),timezone.utc)
    if close>now:close-=timedelta(days=7)
    monkeypatch.setattr(time,'time',lambda:close.timestamp()+30)


def run_with_quote(tmp_path, *, tick_time, models=('EURUSD','GBPUSD')):
    cfg=make_cfg(tmp_path)
    api=FakeAPI(NOW,supported={'EURUSD','GBPUSD'})
    api.symbol_info_tick=lambda s:NS(bid=1.1,ask=1.1001,time=tick_time)
    source=FakeSource(api)
    population=FakePopulation(models,NOW,cfg=cfg)
    return choose_mode(cfg,source_factory=lambda:source,history=FakeStore(),
                       population=population,now_s=NOW)


def test_mt5_tick_timestamp_is_already_utc_and_never_offset():
    assert broker_tick_age(NOW,NOW-8)==8
    assert broker_tick_age(NOW,NOW+8)==-8


def test_router_rejects_future_mt5_utc_tick_without_offset_guessing(tmp_path):
    with pytest.raises(ConnectionError,match='market closure unconfirmed'):
        run_with_quote(tmp_path,tick_time=NOW+OFFSET-8)


def test_historical_spread_uses_only_training_half_not_final_quarter(tmp_path,monkeypatch):
    import quant_rl.data.mt5_history as h
    start=1640995200
    bars=[OHLCVBar(start+i*3600,1.1,1.1005,1.0995,1.1,120,
                   2.0 if i<550 else (100.0 if i>=900 else None)) for i in range(1100)]
    assert MT5History.historical_spread_estimate(bars)==pytest.approx(2.0)
    api=NS(account_info=lambda:NS(currency='USD'),
           symbol_info=lambda s:NS(spread=0),
           symbol_info_tick=lambda s:None)
    class Broker:
        def __init__(self,**kw):self._mt5=api
        def connect(self):pass
        def close(self):pass
        def get_symbol_spec(self,s,c):
            from quant_rl.data.mt5_source import SymbolSpec
            from tests._specs import default_symbol_spec
            return default_symbol_spec(s)
        def fetch_bars_since(self,s,tf,start):return bars
        def fetch_m1_range(self,s,start,end):return []
    monkeypatch.setattr(h,'MT5DataSource',Broker)
    # This test targets spread provenance; its 45-day data is not a four-year fixture.
    monkeypatch.setattr(h,'select_rolling_bars',lambda bars,tf,years,**kw:
                        (bars,bars[0].time_s,bars[-1].time_s+3600))
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path),'research':{'symbols':['USDCAD'],'lookback_years':2}})
    frozen=MT5History(cfg).fetch('USDCAD')
    assert frozen.fallback_spread_points==2.0
    assert frozen.manifest.fallback_spread_source=='MT5_TRAINING_SPREAD_P95_ESTIMATE'
    assert frozen.manifest.verified_spread_fraction<1
    assert frozen.bars[550].spread_points is None
    assert frozen.bars[-1].spread_points==100.


def test_no_spread_basis_when_training_half_has_too_few_observations():
    base=1640995200
    bars=[OHLCVBar(base+i*3600,1.1,1.1001,1.0999,1.1,200,
                   None if i<900 else 30.) for i in range(1500)]
    assert MT5History.historical_spread_estimate(bars) is None


def test_live_runner_uses_raw_mt5_utc_for_order_quote(tmp_path,monkeypatch):
    import time
    from tests.test_live_mt5_unified import FakeBroker, setup_session
    broker=FakeBroker(account_id=9900)
    session=setup_session(tmp_path,broker,'BUY')
    broker.tick_time=int(time.time())-4
    result=session.step()
    assert result['status']=='FILLED'
    assert broker.calls==1


def test_live_runner_rejects_future_mt5_utc_tick(tmp_path):
    import time
    from tests.test_live_mt5_unified import FakeBroker, setup_session
    broker=FakeBroker(account_id=9901)
    session=setup_session(tmp_path,broker,'BUY')
    broker.tick_time=int(time.time())+OFFSET-2
    with pytest.raises(ConnectionError,match='Stale live quote'):
        session.step()
    assert broker.calls==0


def test_spread_snapshot_cannot_understate_training_history(tmp_path,monkeypatch):
    import quant_rl.data.mt5_history as h
    from quant_rl.data.mt5_source import SymbolSpec
    from tests._specs import default_symbol_spec
    start=1640995200
    bars=[OHLCVBar(start+i*3600,1.1,1.1004,1.0996,1.1,150,
                   14. if i<600 else None) for i in range(1200)]
    api=NS(account_info=lambda:NS(currency='USD'),
           symbol_info=lambda s:NS(spread=1),
           symbol_info_tick=lambda s:None)
    class Broker:
        def __init__(self,**kw):self._mt5=api
        def connect(self):pass
        def close(self):pass
        def get_symbol_spec(self,s,c):return default_symbol_spec(s)
        def fetch_bars_since(self,s,tf,start):return bars
        def fetch_m1_range(self,s,start,end):return []
    monkeypatch.setattr(h,'MT5DataSource',Broker)
    # This test targets spread provenance; its 45-day data is not a four-year fixture.
    monkeypatch.setattr(h,'select_rolling_bars',lambda bars,tf,years,**kw:
                        (bars,bars[0].time_s,bars[-1].time_s+3600))
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path),'research':{'lookback_years':2}})
    frozen=MT5History(cfg).fetch('EURUSD')
    assert frozen.fallback_spread_points==pytest.approx(14.)
    assert frozen.manifest.fallback_spread_source=='MT5_BROKER_SNAPSHOT_AND_TRAINING_P95_MAX'
