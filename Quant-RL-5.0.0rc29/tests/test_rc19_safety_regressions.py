"""Adversarial rc19 assertions; uses fake broker ONLY and never sends MT5 orders."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone, timedelta, time as wall_time
from pathlib import Path
from types import SimpleNamespace as NS
import json
import time
import numpy as np
import pytest

from quant_rl.data.mt5_source import OHLCVBar
from quant_rl.data.time_alignment import resample_ohlcv_utc
from quant_rl.data.mt5_history import MT5History
from quant_rl.data.mt5_source import broker_spec_hash
from quant_rl.config.schema import QuantRLConfig
from quant_rl.execution.live import LiveSession
from quant_rl.execution.state_io import store_json
from quant_rl.execution.operational_safety import kill_switch_engaged,broker_heartbeat_healthy
from quant_rl.routing import choose_mode
from tests._specs import default_symbol_spec
from tests.test_live_mt5_unified import FakeBroker,setup_session
from tests.test_auto_mt5_routing import FakeAPI,FakeSource,FakePopulation

@pytest.fixture(autouse=True)
def stable_friday_clock(monkeypatch):
    now=datetime.now(timezone.utc)
    friday=now.date()-timedelta(days=(now.weekday()-4)%7)
    market_close=datetime.combine(friday,wall_time(20),timezone.utc)
    if market_close>now:
        market_close-=timedelta(days=7)
    monkeypatch.setattr(time,'time',lambda:market_close.timestamp()+30)


def test_sparse_h1_never_masquerades_as_h2():
    src=[OHLCVBar(i*7200,1.1,1.2,1.0,1.1,100,3.0) for i in range(8)]
    assert resample_ohlcv_utc(src,'H4',source_timeframe='H1') == []
    assert len(resample_ohlcv_utc(src,'H4',source_timeframe='H2')) == 4
    with pytest.raises(ValueError,match='aligned'):
        resample_ohlcv_utc([replace(src[0],time_s=30)],'H4',source_timeframe='H1')


def test_fingerprint_broker_identity_and_deep_freezing(tmp_path):
    store=MT5History()
    bars=[OHLCVBar(1704067200+i*3600,1.1,1.2,1.0,1.1,100,3.0) for i in range(8)]
    a=store.freeze('EURUSD','H1',bars,default_symbol_spec('EURUSD'),broker_server='A')
    b=store.freeze('EURUSD','H1',bars,default_symbol_spec('EURUSD'),broker_server='B')
    assert a.fingerprint!=b.fingerprint
    assert isinstance(a.bars,tuple)
    assert not a.close.flags.writeable
    with pytest.raises(ValueError):a.close[0]=0


def test_kill_switch_file_env_and_invalid_file_fail_closed(tmp_path,monkeypatch):
    monkeypatch.delenv('QUANT_RL_KILL_SWITCH',raising=False)
    assert not kill_switch_engaged(tmp_path)
    p=Path(tmp_path)/'live_state'/'kill_switch.json'
    p.parent.mkdir(parents=True)
    p.write_text('{"engaged":true}')
    assert kill_switch_engaged(tmp_path)
    p.write_text('{"engaged":false}')
    assert not kill_switch_engaged(tmp_path)
    p.write_text('not-json')
    assert kill_switch_engaged(tmp_path)
    p.write_text('{"engaged":false}')
    monkeypatch.setenv('QUANT_RL_KILL_SWITCH','1')
    assert kill_switch_engaged(tmp_path)
    monkeypatch.setenv('QUANT_RL_KILL_SWITCH','unknown')
    assert kill_switch_engaged(tmp_path)


def test_operator_kill_switch_never_places_order(tmp_path,monkeypatch):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    monkeypatch.setenv('QUANT_RL_KILL_SWITCH','1')
    assert session.step()['status']=='KILL_SWITCH_ENGAGED'
    assert broker.calls==0


def test_failed_broker_heartbeat_is_observed_by_c2(tmp_path,monkeypatch):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    monkeypatch.setattr('quant_rl.execution.live.broker_heartbeat_healthy',lambda *args,**kw:False)
    result=session.step()
    assert result['status']=='VETOED' and result['veto_code']=='C2'
    assert broker.calls==0


def test_broker_heartbeat_can_fail_on_last_refresh(tmp_path,monkeypatch):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    calls=iter([True,False])
    monkeypatch.setattr('quant_rl.execution.live.broker_heartbeat_healthy',lambda *args,**kw: next(calls))
    result=session.step()
    assert result['status']=='VETOED_REFRESH'
    assert broker.calls==0


def test_broker_spec_drift_never_trades(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    broker.spec=replace(broker.spec,point=broker.spec.point*2)
    with pytest.raises(ValueError,match='economics'):
        session.step()
    assert broker.calls==0


def test_pending_deal_recovery_requires_exact_broker_identity(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    state=session._load_state(broker.account())
    marker='QRL1700000000'
    state.update(status='PENDING',request_id=marker,submit_time_s=int(time.time()),last_bar_time_s=1700000000,
        requested_volume_lots=0.1,requested_direction='BUY')
    broker._mt5.DEAL_ENTRY_IN=0
    broker._mt5.DEAL_TYPE_BUY=0
    broker._mt5.DEAL_TYPE_SELL=1
    store_json(session._journal(broker.account_id),state)
    broker.recent_deals=lambda since:[NS(magic=session.cfg.live.magic,symbol='EURUSD',comment=marker,
        entry=0,type=0,volume=0.1,ticket=73)]
    assert session.step()['status']=='RECOVERED_FILLED'
    assert broker.calls==0
    stored=json.loads(session._journal(broker.account_id).read_text())
    assert stored['status']=='FILLED'
    assert stored['recovered_deal_ids']==[73]


def test_pending_mismatched_deal_stays_unresolved(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    state=session._load_state(broker.account())
    state.update(status='PENDING',request_id='QRL1700000000',submit_time_s=int(time.time()),last_bar_time_s=1700000000,
        requested_volume_lots=0.1,requested_direction='BUY')
    store_json(session._journal(broker.account_id),state)
    broker.recent_deals=lambda since:[NS(magic=session.cfg.live.magic,symbol='EURUSD',comment='QRL_WRONG',ticket=73)]
    with pytest.raises(RuntimeError,match='unresolved'):
        session.step()
    assert broker.calls==0
    assert json.loads(session._journal(broker.account_id).read_text())['status']=='PENDING'


def test_non_usd_account_model_not_rejected_as_missing(tmp_path):
    now=int(time.time())
    class EuroAPI(FakeAPI):
        def account_info(self):return NS(login=123,currency='EUR',trade_mode=0,trade_allowed=True,server='MetaQuotes-Demo')
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path),'research':{'symbols':['EURUSD']},'live':{'symbol':'EURUSD'},'broker_connection':{'mt5_server_timezone':'UTC'}})
    effective=cfg.model_copy(update={'execution':cfg.execution.model_copy(update={'account_currency':'EUR'})})
    pop=FakePopulation(('EURUSD',),now,cfg=cfg)
    pop.data['symbols']['EURUSD'].update(account_currency='EUR',policy_hash=effective.policy_hash)
    source=FakeSource(EuroAPI(now))
    decision,_,errors=choose_mode(cfg,source_factory=lambda:source,population=pop,now_s=now)
    assert decision.mode=='LIVE' and decision.live_symbol=='EURUSD'
    assert not errors


def test_single_position_contract_rejects_multi_position_config(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    execution=session.cfg.execution.model_copy(update={'max_open_positions':2})
    cfg=session.cfg.model_copy(update={'execution':execution})
    with pytest.raises(ValueError,match='single-position'):
        LiveSession(cfg,broker)


def test_distinct_server_broker_heartbeat_fails_closed():
    class Broker:
        def account(self):return NS(login=1,server='unexpected')
    assert not broker_heartbeat_healthy(Broker(),login=1,server='expected')
