"""Real calendar rolling-window tests; generated bars are test fixtures only."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from quant_rl.config.schema import QuantRLConfig, ResearchConfig
from quant_rl.data.mt5_source import OHLCVBar
from tests._specs import default_symbol_spec
from quant_rl.data.rolling_window import (
    rolling_start_utc, rolling_request_start_utc, select_rolling_bars,
)
from quant_rl.data.mt5_history import MT5History

UTC = timezone.utc


def utc(y, m, d, h=0):
    return int(datetime(y,m,d,h,tzinfo=UTC).timestamp())


def bar(ts):
    return OHLCVBar(ts,1.1,1.1005,1.0995,1.1,150,3.)


def test_training_budget_is_one_thousand_each_and_deprecated_start_year_is_rejected():
    cfg=QuantRLConfig()
    assert (cfg.research.lookback_years,cfg.ppo.episodes,cfg.ppo.ppo_epochs,
            cfg.optimization.trials)==(4,1000,1000,1000)
    assert cfg.ppo.model_copy(update={'ppo_epochs':1500}).ppo_epochs==1500
    with pytest.raises(ValidationError):
        ResearchConfig(start_year=2022)
    with pytest.raises(ValidationError):
        ResearchConfig(lookback_years=0)


def test_four_year_calendar_subtraction_is_not_fixed_days_and_handles_feb29():
    assert rolling_start_utc(utc(2026,10,30,12),4)==utc(2022,10,30,12)
    assert rolling_start_utc(utc(2024,2,29,12),1)==utc(2023,2,28,12)
    assert rolling_start_utc(utc(2025,2,28,12),1)==utc(2024,2,28,12)
    assert rolling_request_start_utc(utc(2026,10,30,12),4)<utc(2022,10,30,12)


def test_rolling_selection_moves_with_latest_completed_bar_and_excludes_forming_candle():
    # Weekly bars in a broker mock, including data beyond the rolling period.
    first=datetime(2021,1,1,tzinfo=UTC)
    rows=[bar(int((first+timedelta(days=7*i)).timestamp())) for i in range(320)]
    close=rows[-2].time_s+3600
    execution_time=close+30
    selected,start,end=select_rolling_bars(rows,'H1',4,now_utc_s=execution_time)
    assert end==execution_time
    assert start==rolling_start_utc(execution_time,4)
    assert selected[0].time_s>=start and selected[-1].time_s+3600<=end
    assert rows[-1] not in selected
    close_next=rows[-1].time_s+3600
    selected2,start2,end2=select_rolling_bars(rows,'H1',4,now_utc_s=close_next+30)
    assert end2==close_next+30 and end2>end and start2>start
    assert selected2[-1]==rows[-1]


def test_rolling_selection_rejects_short_history_and_stale_broker_history():
    now=utc(2026,10,30,12)
    with pytest.raises(ValueError,match='Insufficient MT5 rolling history'):
        select_rolling_bars([bar(utc(2025,1,1)),bar(now-3600)],'H1',4,now_utc_s=now)
    rows=[bar(utc(2022,1,1)+i*7*86400) for i in range(210)]
    with pytest.raises(ValueError,match='endpoint is stale'):
        select_rolling_bars(rows,'H1',4,now_utc_s=now)


def test_rolling_window_in_mt5_acquisition_and_m1_bounds(tmp_path,monkeypatch):
    import quant_rl.data.mt5_history as store_module
    # Four years of mock MT5 H1 source bars + extra older history; test does
    # not use synthetic data in the runtime path or make a real broker request.
    end=utc(2026,10,30,12)
    now=end+60
    origin=datetime(2022,10,1,tzinfo=UTC)
    rows=[]
    day=origin
    while int(day.timestamp())<end:
        for hour in (8,9,10,11):
            ts=int((day+timedelta(hours=hour)).timestamp())
            if ts+3600<=end:
                rows.append(bar(ts))
        day+=timedelta(days=1)
    calls=[]
    api=NS(account_info=lambda:NS(currency='USD'),
           symbol_info=lambda s:NS(spread=10),symbol_info_tick=lambda s:None)
    class MT5BrokerStub:
        def __init__(self,**kwargs):self._mt5=api
        def connect(self):calls.append(('connect',))
        def close(self):calls.append(('close',))
        def get_symbol_spec(self,s,c):return default_symbol_spec(s)
        def fetch_bars_since(self,s,tf,requested_start):
            calls.append(('H1',requested_start))
            return [b for b in rows if b.time_s>=requested_start]
        def fetch_m1_range(self,s,start,end):
            calls.append(('M1',start,end))
            return []
    monkeypatch.setattr(store_module,'MT5DataSource',MT5BrokerStub)
    monkeypatch.setattr(store_module.time,'time',lambda:now)
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path)})
    frozen=MT5History(cfg).fetch('EURUSD')
    actual_start=rolling_start_utc(now,4)
    assert frozen.bars[0].time_s>=actual_start
    assert frozen.bars[-1].time_s+3600==end
    assert any(c[0]=='H1' and c[1]<utc(2022,10,30) for c in calls)
    # The requested rolling boundary is the exact execution instant, while
    # M1 evidence stops at the last fully completed H1 candle.
    assert ('M1',frozen.bars[0].time_s,end) in calls
    assert frozen.manifest.data_origin=='MT5_TERMINAL'
    assert frozen.manifest.calendar_span_days>1450
