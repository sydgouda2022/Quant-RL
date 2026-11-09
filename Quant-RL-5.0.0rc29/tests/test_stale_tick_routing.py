"""Regression: stale MT5 ticks must not block missing-model historical research."""
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from quant_rl.routing import choose_mode
from tests.test_auto_mt5_routing import FakeAPI, FakeSource, FakeStore, FakePopulation, make_cfg


FRIDAY_MORNING = int(datetime(2026, 10, 30, 7, 36, tzinfo=timezone.utc).timestamp())
SATURDAY_NOON = int(datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc).timestamp())


def _route(tmp_path, now, models=(), *, history_missing=(), quote_age=86400, trade_mode=4):
    cfg = make_cfg(tmp_path)
    api = FakeAPI(now, supported={'EURUSD','GBPUSD'})
    api.symbol_info = lambda symbol: NS(point=.00001, trade_contract_size=100000,
                                         trade_mode=trade_mode) if symbol in api.supported else None
    api.symbol_info_tick = lambda symbol: NS(bid=1.1, ask=1.1001,
                                           time=now-quote_age)
    src = FakeSource(api)
    history = FakeStore(history_missing)
    import quant_rl.routing as routing
    original = routing._qualified_history
    routing._qualified_history = lambda c,s,d: (True, 'Qualified')
    try:
        return choose_mode(cfg,source_factory=lambda:src,history=history,
                           population=FakePopulation(models,now,cfg=cfg),now_s=now,preload_research=True),src,history
    finally:
        routing._qualified_history = original


def test_friday_stale_quotes_researches_missing_models(tmp_path):
    (decision,datasets,rejected),src,store = _route(tmp_path, FRIDAY_MORNING,
        models=('EURUSD',))
    assert decision.mode == 'RESEARCH'
    assert decision.reason == 'MISSING_MODELS'
    assert decision.targets == ('GBPUSD',)
    assert not decision.market_open
    assert list(datasets) == ['GBPUSD']
    assert store.read == ['GBPUSD']
    assert not rejected
    assert src.closed


def test_friday_all_missing_models_have_no_tick_but_researches_history(tmp_path):
    (decision,datasets,rejected),src,store = _route(tmp_path,FRIDAY_MORNING)
    assert decision.mode == 'RESEARCH'
    assert decision.targets == ('EURUSD','GBPUSD')
    assert set(datasets) == {'EURUSD','GBPUSD'}
    assert src.closed


def test_friday_stale_quotes_with_all_models_does_not_trade_or_call_history(tmp_path):
    cfg=make_cfg(tmp_path)
    api=FakeAPI(FRIDAY_MORNING,supported={'EURUSD','GBPUSD'})
    api.symbol_info_tick=lambda s:NS(bid=1.1,ask=1.1001,time=FRIDAY_MORNING-600)
    src=FakeSource(api); history=FakeStore()
    with pytest.raises(ConnectionError, match='market closure unconfirmed'):
        choose_mode(cfg,source_factory=lambda:src,history=history,
                    population=FakePopulation(('EURUSD','GBPUSD'),FRIDAY_MORNING,cfg=cfg),now_s=FRIDAY_MORNING)
    assert history.read==[]
    assert src.closed


def test_weekend_stale_quotes_researches_existing_models_too(tmp_path):
    (decision,datasets,rejected),src,store = _route(tmp_path,SATURDAY_NOON,
         models=('EURUSD','GBPUSD'))
    assert decision.reason == 'WEEKLY_SCHEDULE'
    assert decision.targets == ('EURUSD','GBPUSD')
    assert set(datasets)=={'EURUSD','GBPUSD'}
    assert src.closed


def test_missing_model_research_with_stale_quotes_still_requires_fresh_mt5_history(tmp_path):
    with pytest.raises(RuntimeError, match='no selected symbols have sufficient'):
        _route(tmp_path,FRIDAY_MORNING,models=('EURUSD',),history_missing=('GBPUSD',))


def test_friday_server_cutoff_blocks_live_even_with_fresh_quotes(tmp_path):
    # 21:00 UTC = 23:00 server time with the configured +2h broker offset.
    cfg = make_cfg(tmp_path).model_copy(update={
        'broker_connection': make_cfg(tmp_path).broker_connection.model_copy(
            update={'mt5_server_timezone': 'Europe/Athens'})
    })
    now = int(datetime(2026, 11, 6, 21, 0, tzinfo=timezone.utc).timestamp())
    api = FakeAPI(now, supported={'EURUSD', 'GBPUSD'})
    api.symbol_info_tick = lambda symbol: NS(bid=1.1, ask=1.1001, time=now - 5)
    src = FakeSource(api)
    decision, datasets, rejected = choose_mode(
        cfg, source_factory=lambda: src, history=FakeStore(),
        population=FakePopulation(('EURUSD', 'GBPUSD'), now, cfg=cfg, server_timezone='Europe/Athens'), now_s=now,
    )
    assert decision.mode == 'RESEARCH'
    assert decision.reason == 'WEEKLY_SCHEDULE'
    assert decision.targets == ('EURUSD', 'GBPUSD')
    assert decision.market_open is False
    assert datasets == {} and rejected == {}
    assert src.closed


def test_sunday_after_server_open_can_route_live_when_quote_and_model_are_valid(tmp_path):
    now = int(datetime(2026, 11, 8, 21, 0, tzinfo=timezone.utc).timestamp())
    cfg = make_cfg(tmp_path)
    api = FakeAPI(now, supported={'EURUSD', 'GBPUSD'})
    api.symbol_info_tick = lambda symbol: NS(bid=1.1, ask=1.1001, time=now - 5)
    src = FakeSource(api)
    decision, datasets, rejected = choose_mode(
        cfg, source_factory=lambda: src, history=FakeStore(),
        population=FakePopulation(('EURUSD', 'GBPUSD'), now, cfg=cfg), now_s=now,
    )
    assert decision.mode == 'LIVE'
    assert decision.live_symbol in ('EURUSD', 'GBPUSD')
    assert decision.market_open is True
    assert datasets == {} and rejected == {}
    assert src.closed
