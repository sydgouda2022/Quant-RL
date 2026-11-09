"""Persistent C14 losses and dynamic economic blackout negative-path tests."""
from __future__ import annotations
import json
from datetime import datetime, timedelta, timezone, time as wall_time
import time
from types import SimpleNamespace as NS

import pytest

from quant_rl.execution.loss_history import consecutive_loss_streak
from quant_rl.execution.operational_safety import operator_blackout_active


class MT5:
    DEAL_ENTRY_OUT=1
    DEAL_ENTRY_OUT_BY=2
    DEAL_ENTRY_INOUT=3
    DEAL_TYPE_BUY=0
    DEAL_TYPE_SELL=1


class Broker:
    _mt5=MT5()
    def __init__(self):
        self.calls=[]
        self.deals=[]

    def recent_deals(self,since):
        self.calls.append(since)
        return [d for d in self.deals if d.time>=since]


def close(ticket,time_s,pnl,*,symbol='EURUSD',magic=520052):
    return NS(ticket=ticket,time=time_s,symbol=symbol,magic=magic,entry=1,type=0,
              profit=pnl,commission=0.,swap=0.,fee=0.)


def streak(broker,root):
    return consecutive_loss_streak(broker,root=root,login=123,server='Demo-A',
                                   symbol='EURUSD',magic=520052)


def test_persistent_streak_survives_quiet_period_and_restart(tmp_path):
    now=1_900_000_000
    broker=Broker()
    # The last loss occurred five months ago, outside the previous 90-day scan.
    broker.deals=[close(10,now-180*86400,150),close(11,now-160*86400,-20),
                  close(12,now-150*86400,-30)]
    assert streak(broker,tmp_path)==2
    assert broker.calls==[0]
    assert streak(broker,tmp_path)==2
    assert broker.calls[-1]==broker.deals[-1].time-120
    broker.deals.append(close(13,now,-10))
    assert streak(broker,tmp_path)==3
    # Checkpoint survives constructing another broker adapter instance.
    replacement=Broker()
    replacement.deals=broker.deals
    assert streak(replacement,tmp_path)==3
    broker.deals.append(close(14,now+3600,40))
    assert streak(broker,tmp_path)==0


def test_missing_broker_anchor_and_corrupt_loss_identity_fail_closed(tmp_path):
    broker=Broker()
    broker.deals=[close(11,1700000000,-10)]
    assert streak(broker,tmp_path)==1
    broker.deals=[]
    with pytest.raises(RuntimeError,match='Cannot verify previous closing deal'):
        streak(broker,tmp_path)
    path=next((tmp_path/'live_state').glob('*loss_history.json'))
    record=json.loads(path.read_text())
    record['server']='OtherBroker'
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match='identity mismatch'):
        streak(broker,tmp_path)


def test_blackout_file_updates_live_without_restart_and_malformed_fails_closed(tmp_path,monkeypatch):
    monkeypatch.delenv('QUANT_RL_ECONOMIC_BLACKOUT',raising=False)
    root=tmp_path
    assert not operator_blackout_active(root)
    folder=root/'live_state'
    folder.mkdir()
    path=folder/'economic_blackout.json'
    path.write_text('{"active": true}')
    assert operator_blackout_active(root)
    path.write_text('{"active": false}')
    assert not operator_blackout_active(root)
    path.write_text('{"active": "maybe"}')
    assert operator_blackout_active(root)
    path.unlink()
    monkeypatch.setenv('QUANT_RL_ECONOMIC_BLACKOUT','yes')
    assert operator_blackout_active(root)
    monkeypatch.setenv('QUANT_RL_ECONOMIC_BLACKOUT','1')
    assert operator_blackout_active(root)


def _freeze_open_broker_candle(monkeypatch):
    # Follow the LIVE fixture clock's 30-second-after-Friday-H4-close basis.
    now=datetime.now(timezone.utc)
    friday=now.date()-timedelta(days=(now.weekday()-4)%7)
    close=datetime.combine(friday,wall_time(20),timezone.utc)
    if close>now:
        close-=timedelta(days=7)
    monkeypatch.setattr(time,'time',lambda:close.timestamp()+30)


def test_operator_blackout_vetoes_live_decision(tmp_path, monkeypatch):
    from tests.test_live_mt5_unified import FakeBroker, setup_session
    _freeze_open_broker_candle(monkeypatch)
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    live_state=tmp_path/'live_state'
    live_state.mkdir(exist_ok=True)
    (live_state/'economic_blackout.json').write_text('{"active": true}')
    result=session.step()
    assert result['status']=='VETOED'
    assert broker.calls==0


def test_operator_blackout_rechecked_immediately_before_send(tmp_path,monkeypatch):
    from tests.test_live_mt5_unified import FakeBroker, setup_session
    _freeze_open_broker_candle(monkeypatch)
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    # Simulate the operator engaging an emergency blackout between the final
    # Examiner approval and order_send. No broker order may be submitted.
    readings=iter([False,False,True])
    monkeypatch.setattr('quant_rl.execution.live.operator_blackout_active',lambda *args:next(readings))
    result=session.step()
    assert result['status']=='HALTED_BEFORE_SEND'
    assert broker.calls==0


def test_zero_offset_legacy_loss_checkpoint_migrates_and_nonzero_fails_closed(tmp_path):
    broker=Broker()
    broker.deals=[close(10,1_900_000_000,-10)]
    assert streak(broker,tmp_path)==1
    path=next((tmp_path/'live_state').glob('*loss_history.json'))
    record=json.loads(path.read_text())
    record.pop('timestamp_basis',None)
    record['version']=2
    record['offset_seconds']=0
    path.write_text(json.dumps(record))
    assert streak(broker,tmp_path)==1
    migrated=json.loads(path.read_text())
    assert migrated['version']==3 and migrated['timestamp_basis']=='MT5_UTC'
    migrated.pop('timestamp_basis',None)
    migrated['version']=2
    migrated['offset_seconds']=7200
    path.write_text(json.dumps(migrated))
    with pytest.raises(RuntimeError,match='nonzero-offset'):
        streak(broker,tmp_path)
