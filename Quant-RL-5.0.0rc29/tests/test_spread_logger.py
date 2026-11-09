"""Regression for the user's all-symbol 'Insufficient MT5 spread evidence' startup failure."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from quant_rl.config.logger import log
from quant_rl.config.schema import QuantRLConfig
from quant_rl.data.mt5_history import MT5History
from quant_rl.data.mt5_source import OHLCVBar
from tests._specs import default_symbol_spec
from quant_rl.market.state_builder import ObservationBuilder
from quant_rl.routing import _qualified_history


def historical_fixture():
    # Fully complete H4 buckets (four real-format H1 records per trading day).
    # The data are test fixtures, never a production research source.
    first=1640995200
    return [OHLCVBar(first + day*86400 + hour*3600,
                     1.1,1.1004,1.0996,1.1,125,None)
            for day in range(1100) for hour in range(4)]


def test_all_zero_mt5_historical_spread_with_broker_quote_can_train(tmp_path,monkeypatch):
    import quant_rl.data.mt5_history as h
    api=NS(account_info=lambda:NS(currency='USD'),
           symbol_info=lambda symbol:NS(spread=15),
           symbol_info_tick=lambda symbol:NS(bid=1.1, ask=1.10012))
    hits=[]
    class Broker:
        def __init__(self,**kw):self._mt5=api
        def connect(self):hits.append('connected')
        def get_symbol_spec(self,symbol,currency):return default_symbol_spec(symbol)
        def fetch_bars_since(self,symbol,tf,start):hits.append('fetch');return historical_fixture()
        def fetch_m1_range(self,*args):return []
        def close(self):hits.append('closed')
    monkeypatch.setattr(h,'MT5DataSource',Broker)
    # The broker-source stub predates rolling windows; this test checks spreads.
    monkeypatch.setattr(h,'select_rolling_bars',lambda bars,tf,years,**kw:
                        (bars,bars[0].time_s,bars[-1].time_s+3600))
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path),
                                      'research':{'symbols':['EURUSD'],'lookback_years':2}, 'data':{'observation_timeframes':['H1','H4']}})
    frozen=MT5History(cfg).fetch('EURUSD')
    assert frozen.manifest.data_origin=='MT5_TERMINAL'
    assert frozen.manifest.verified_spread_fraction==0
    assert all(b.spread_points is None for b in frozen.bars)
    assert frozen.fallback_spread_points==pytest.approx(15.)
    # Qualification test isolates spread semantics from the 2-year coverage policy.
    monkeypatch.setattr('quant_rl.routing.select_rolling_bars',lambda bars,tf,years,**kw:
                        (bars,bars[0].time_s,bars[-1].time_s+3600))
    eligible, reason=_qualified_history(cfg,'EURUSD',frozen)
    assert eligible, reason
    obs=ObservationBuilder(['H4'],8).build_dataset(
        base_bars=frozen.bars,symbol='EURUSD',primary_timeframe='H4',
        spec=frozen.spec,bar_price_side='BID',
        fallback_spread_points=frozen.fallback_spread_points)
    assert len(obs.spread_pips)>300
    assert obs.spread_pips[100]==pytest.approx(1.5)
    assert hits==['connected','fetch','closed']


def test_missing_broker_quote_and_historical_spread_is_rejected(tmp_path):
    store=MT5History()
    frozen=store.freeze('EURUSD','H1',historical_fixture(),
                        default_symbol_spec('EURUSD'))
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path)})
    ok,why=_qualified_history(cfg,'EURUSD',frozen)
    assert not ok
    assert 'spread' in why.lower()


def test_reported_mt5_spread_is_not_overwritten_by_broker_estimate(tmp_path):
    bars=historical_fixture()[:8]
    bars[0]=OHLCVBar(bars[0].time_s,1.1,1.1004,1.0996,1.1,125,4.)
    frozen=MT5History().freeze('EURUSD','H1',bars,
        default_symbol_spec('EURUSD'),
        fallback_spread_points=14.)
    assert frozen.spread_points[0]==4
    assert frozen.fallback_spread_points==14
    assert frozen.manifest.verified_spread_fraction==0.125


def test_quant_unified_logger_writes_jsonl_and_redacts_secrets(tmp_path,capsys):
    secret='FAKE_TEST_PASSWORD_23843'
    log.configure(tmp_path,secrets=[secret])
    try:
        log.info('ROUTER','Connected')
        log.warning('MT5 DATA',f'No history: {secret}')
        logs=list((tmp_path/'logs').glob('quant_rl_*.log'))
        assert len(logs)==1
        data=[json.loads(line) for line in logs[0].read_text(encoding='utf8').splitlines()]
        assert data[-2]['component']=='ROUTER'
        assert data[-1]['level']=='WARNING'
        assert secret not in logs[0].read_text(encoding='utf8')
        assert secret not in capsys.readouterr().out
        assert data[-1]['timestamp_utc']
    finally:
        log.shutdown()
        log.configure(Path(__file__).resolve().parents[1])
