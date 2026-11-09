"""No network or order submission: broker is a fake MT5 terminal interface."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import time
from datetime import datetime, timezone, timedelta, time as wall_time
import numpy as np
import pytest
import torch
from quant_rl.config.schema import QuantRLConfig
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from tests._fixtures import SimulatedFXDataSource
from quant_rl.market.state_builder import ObservationBuilder
from quant_rl.intelligence.policy import EconomicFeatures,PPOSettings,PPOTrader
from quant_rl.research.evolution import ModelPopulation,architecture_hash
from quant_rl.data.mt5_source import broker_spec_hash
from quant_rl.execution.live import LiveSession

@pytest.fixture(autouse=True)
def stable_market_clock(monkeypatch):
    # Every test executes 30 seconds after an H4 candle closes; no dependency
    # on the wall-clock hour when the test suite happens to run.
    # Set a genuine FX-session Friday H4 close, not a weekend candle which
    # the broker cannot supply when the test suite runs on Saturday/Sunday.
    now=datetime.now(timezone.utc)
    friday=now.date()-timedelta(days=(now.weekday()-4)%7)
    market_close=datetime.combine(friday,wall_time(20),timezone.utc)
    if market_close>now:
        market_close-=timedelta(days=7)
    monkeypatch.setattr(time,'time',lambda: market_close.timestamp()+30)


class FakeTerminal:
    ORDER_TYPE_BUY=0
    ORDER_TYPE_SELL=1
    POSITION_TYPE_BUY=0
    POSITION_TYPE_SELL=1
    def order_calc_profit(self,side,symbol,volume,entry,sl):
        return (sl-entry)*(1 if side==0 else -1)*100000*volume
    def order_calc_margin(self,side,symbol,volume,price):
        return 100000*volume/30

class FakeBroker:
    def __init__(self,account_id=111):
        self._mt5=FakeTerminal()
        self.account_id=account_id
        self.bars=SimulatedFXDataSource(seed=18).fetch_bars('EURUSD','H1',2400)
        self.spec=default_symbol_spec('EURUSD')
        self._positions=[]
        self._orders=[]
        self.calls=0
        self.fail_uncertain=False
        self.position_protection_override=None
        self.tick_time=int(time.time())
    def account(self):
        return SimpleNamespace(login=self.account_id,equity=10000.0,balance=10000.0,
                               leverage=30,currency='USD',margin_free=10000.0,
                               trade_mode=0)
    def closed_bars(self,symbol):
        return self.bars
    def fetch_bars(self,symbol,timeframe,count):
        return self.bars[-count:]
    def get_symbol_spec(self,symbol,currency):
        return self.spec
    def positions(self):return self._positions
    def orders(self):return self._orders
    def recent_deals(self,start_s):return []
    def tick(self,symbol):
        price=self.bars[-1].close
        return SimpleNamespace(bid=price,ask=price+.00009,time=self.tick_time)
    def order_request(self,**kwargs): return kwargs
    def submit(self,request):
        self.calls+=1
        if self.fail_uncertain:
            raise ConnectionError('unknown order outcome')
        position=SimpleNamespace(magic=request['magic'],symbol=request['symbol'],
            comment=request['comment'],type=(0 if request['direction']=='BUY' else 1),
            volume=float(request['volume']),sl=float(request['sl']),tp=float(request['tp']),
            ticket=123)
        if self.position_protection_override:
            position.sl,position.tp=self.position_protection_override
        self._positions=[position]
        return {'status':'FILLED','retcode':10009,'order':121,'deal':122}


def setup_session(tmp_path,broker,direction):
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path),
        'live':{'symbol':'EURUSD','max_entry_delay_seconds':3600}, 
        'research':{'symbols':['EURUSD']},
        'broker_connection':{'mt5_server_timezone':'UTC'},
        'execution':{'max_round_trip_cost_pips':10.0,'max_slippage_points':20}})
    obs=ObservationBuilder(list(cfg.data.observation_timeframes),
         cfg.data.sequence_lookback_bars,cfg.data.include_technical_indicators).build_dataset(
         base_bars=broker.bars,symbol='EURUSD',primary_timeframe='H4',
         volatility_estimator='atr',volatility_period=14,
         spec=broker.spec,bar_price_side='BID')
    feat=EconomicFeatures(obs,np.arange(obs.num_bars-65,obs.num_bars),10000.,
        symbol_spec=broker.spec,account_currency='USD',min_risk_pct=.1,max_risk_pct=1.)
    settings=PPOSettings()
    model=PPOTrader(feat,settings,42)
    for param in model.model.parameters():
        param.data.zero_()
    model.model.direction.bias.data=torch.tensor([10.,-10.,-10.] if direction=='NO_TRADE' else [-10.,10.,-10.])
    pop=ModelPopulation(Path(tmp_path))
    pop.publish({'EURUSD':(model.snapshot(),dict(arch=architecture_hash(cfg,feat,settings),
        origin='MT5_TERMINAL',current_score=1.0,settings=settings.__dict__,
        account_currency='USD',broker_spec_hash=broker_spec_hash(broker.spec),broker_server=str(getattr(broker.account(),'server','')),
        policy_hash=cfg.policy_hash,mt5_server_timezone=cfg.broker_connection.mt5_server_timezone,period={'test_end_utc':__import__('datetime').datetime.fromtimestamp(time.time()-3600,
            __import__('datetime').timezone.utc).isoformat()},
        feature_profile={'feature_group':'all','sequence_bars':32}))},
        {},'EURUSD',{})
    return LiveSession(cfg,broker)


def test_live_no_trade_is_one_decision_per_completed_bar(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'NO_TRADE')
    result=session.step()
    assert result['status']=='NO_TRADE'
    assert broker.calls==0
    assert session.step()['status']=='ALREADY_EVALUATED'
    assert broker.calls==0


def test_other_open_positions_block_new_decisions(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    broker._positions=[object()]
    assert session.step()['status']=='POSITION_OR_ORDER_OPEN'
    assert broker.calls==0
    broker._positions=[]
    assert session.step()['status'] in ('FILLED','VETOED','VETOED_BROKER_CALC','QUOTE_MOVED')


def test_account_segregation_uses_login_not_demo_or_funded_type(tmp_path):
    broker=FakeBroker(111)
    session=setup_session(tmp_path,broker,'NO_TRADE')
    assert session.step()['status']=='NO_TRADE'
    broker.account_id=222
    # Switching accounts is a restart, not a hot-swap inside an open session.
    with pytest.raises(ConnectionError,match='account changed'):
        session.step()
    replacement=LiveSession(session.cfg,broker)
    assert replacement.step()['status']=='NO_TRADE'
    assert (tmp_path/'live_state'/'111_EURUSD.json').is_file()
    assert (tmp_path/'live_state'/'222_EURUSD.json').is_file()


def test_uncertain_order_never_resubmitted(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    broker.fail_uncertain=True
    # Actual order submission only occurs if all 26 checks approve.
    try:
        outcome=session.step()
    except ConnectionError:
        outcome={'status':'UNKNOWN'}
    if outcome['status']=='UNKNOWN':
        assert broker.calls==1
        with pytest.raises(RuntimeError,match='unresolved'):
            session.step()
        assert broker.calls==1
    else:
        assert outcome['status'].startswith('VETOED') or outcome['status']=='QUOTE_MOVED'
        assert broker.calls==0

def test_approved_buy_uses_broker_calc_and_same_api_for_any_login(tmp_path):
    broker=FakeBroker(account_id=999)
    session=setup_session(tmp_path,broker,'BUY')
    result=session.step()
    assert result['status']=='FILLED'
    assert broker.calls==1
    # The idempotency journal persists an actual broker-confirmed fill.
    assert session.step()['status']=='ALREADY_EVALUATED'
    assert broker.calls==1
    import json
    state=json.loads((tmp_path/'live_state'/'999_EURUSD.json').read_text())
    assert state['status']=='FILLED'
    assert state['result']['deal']==122


def test_unknown_broker_outcome_locks_session_instead_of_resending(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    broker.fail_uncertain=True
    with pytest.raises(ConnectionError,match='unknown order outcome'):
        session.step()
    assert broker.calls==1
    with pytest.raises(RuntimeError,match='unresolved'):
        LiveSession(session.cfg,broker).step()
    assert broker.calls==1


def test_stale_closed_candle_is_never_used_as_new_live_entry(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    live=session.cfg.live.model_copy(update={'max_entry_delay_seconds':10})
    session.cfg=session.cfg.model_copy(update={'live':live})
    assert session.step()['status']=='AWAITING_NEXT_BAR'
    assert broker.calls==0


def test_live_blackout_always_vetoes_trade(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    live=session.cfg.live.model_copy(update={'economic_blackout_active':True})
    session.cfg=session.cfg.model_copy(update={'live':live})
    res=session.step()
    assert res['status']=='VETOED'
    assert broker.calls==0


def test_unqualified_model_cannot_enter_live(tmp_path):
    broker=FakeBroker()
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path),'broker_connection':{'mt5_server_timezone':'UTC'}})
    with pytest.raises(ValueError,match='qualified'):
        LiveSession(cfg,broker)


def test_operating_mode_is_auto_routed_and_offline_sources_are_rejected():
    from pydantic import ValidationError
    assert not hasattr(QuantRLConfig().data, 'storage_dir')
    with pytest.raises(ValidationError): QuantRLConfig.model_validate({'mode':'LIVE'})
    with pytest.raises(ValidationError): QuantRLConfig.model_validate({'data':{'source':'simulated'}})
    with pytest.raises(ValidationError): QuantRLConfig.model_validate({'data':{'source':'csv'}})


def test_exit_after_last_bar_waits_for_next_candle(tmp_path):
    broker=FakeBroker()
    broker._mt5.DEAL_ENTRY_OUT=1
    session=setup_session(tmp_path,broker,'BUY')
    ts=int(session.builder.build_dataset(base_bars=broker.bars,symbol='EURUSD',
        primary_timeframe='H4',spec=broker.spec,bar_price_side='BID').decision_time_s[-1])
    broker.recent_deals=lambda start_s:[SimpleNamespace(time=ts+2,entry=1,profit=10.,commission=0.,swap=0.,fee=0.)]
    res=session.step()
    assert res['status']=='AWAITING_NEXT_BAR_AFTER_EXIT'
    assert broker.calls==0

def test_single_account_live_process_lock(tmp_path):
    from quant_rl.research.evolution import exclusive_run
    lock_root=tmp_path/'live_state'/'account_111'
    with exclusive_run(lock_root):
        with pytest.raises(RuntimeError,match='research.lock'):
            with exclusive_run(lock_root): pass
    # The lock path stays to avoid inode replacement races; OS lock is released.
    with exclusive_run(lock_root):
        assert (lock_root/'research.lock').is_file()

class FakeMT5API:
    TRADE_ACTION_DEAL=1
    ORDER_TYPE_BUY=0
    ORDER_TYPE_SELL=1
    ORDER_FILLING_IOC=1
    ORDER_FILLING_FOK=0
    ORDER_FILLING_RETURN=2
    ORDER_TIME_GTC=0
    SYMBOL_TRADE_EXECUTION_MARKET=2
    TRADE_RETCODE_DONE=10009
    TRADE_RETCODE_DONE_PARTIAL=10010
    def __init__(self):
        self.checked=0;self.sent=0
        self._check=SimpleNamespace(retcode=0)
        self._send=SimpleNamespace(retcode=10009,order=10,deal=11)
    def symbol_info(self,symbol):
        return SimpleNamespace(filling_mode=3,trade_exemode=2)
    def order_check(self,request):
        self.checked+=1;return self._check
    def order_send(self,request):
        self.sent+=1;return self._send


def test_mt5_adapter_prechecks_then_sends_identically_for_any_account():
    from quant_rl.execution.mt5_broker import MT5Broker
    for login in (111,222):
        broker=object.__new__(MT5Broker)
        broker._mt5=FakeMT5API()
        broker.login=login
        request=broker.order_request(symbol='EURUSD',direction='BUY',volume=.1,
            price=1.1,sl=1.09,tp=1.12,deviation=10,magic=520052,comment='QRL123')
        assert request['type_filling']==broker._mt5.ORDER_FILLING_FOK
        assert broker.submit(request)=={'status':'FILLED','retcode':10009,'order':10,'deal':11}
        assert broker._mt5.checked==1 and broker._mt5.sent==1


def test_mt5_adapter_rejects_symbols_without_fok_partial_fills_disabled():
    from quant_rl.execution.mt5_broker import MT5Broker
    broker=object.__new__(MT5Broker)
    broker._mt5=FakeMT5API()
    broker._mt5.symbol_info=lambda symbol: SimpleNamespace(filling_mode=2,trade_exemode=2)
    with pytest.raises(ValueError,match='does not support FOK'):
        broker.order_request(symbol='EURUSD',direction='BUY',volume=.1,
            price=1.1,sl=1.09,tp=1.12,deviation=10,magic=520052,comment='QRL123')


def test_mt5_adapter_never_reports_partial_execution_as_filled():
    from quant_rl.execution.mt5_broker import MT5Broker
    broker=object.__new__(MT5Broker)
    broker._mt5=FakeMT5API()
    broker._mt5._send=SimpleNamespace(retcode=10010,order=10,deal=11)
    request={'action':1}
    result=broker.submit(request)
    assert result['status']=='PARTIAL_FILL_UNSUPPORTED'
    assert result['retcode']==10010


def test_mt5_adapter_rejection_never_sends_and_unknown_state_is_not_success():
    from quant_rl.execution.mt5_broker import MT5Broker
    broker=object.__new__(MT5Broker)
    broker._mt5=FakeMT5API()
    broker._mt5._check=SimpleNamespace(retcode=10014)
    assert broker.submit({'action':1})=={'status':'CHECK_REJECTED','retcode':10014}
    assert broker._mt5.sent==0
    broker._mt5._check=SimpleNamespace(retcode=0)
    broker._mt5._send=None
    with pytest.raises(ConnectionError,match='unknown'):
        broker.submit({'action':1})

def test_account_changed_during_session_halts_before_trading(tmp_path):
    broker=FakeBroker(111)
    session=setup_session(tmp_path,broker,'BUY')
    broker.account_id=222
    with pytest.raises(ConnectionError,match='account changed'):
        session.step()
    assert broker.calls==0

def test_account_server_is_part_of_live_identity(tmp_path):
    class OtherServerBroker(FakeBroker):
        server='MetaQuotes-Demo'
        def account(self):
            a=super().account()
            a.server=self.server
            return a
    broker=OtherServerBroker(111)
    session=setup_session(tmp_path,broker,'NO_TRADE')
    assert session.step()['status']=='NO_TRADE'
    broker.server='OtherBroker-Live'
    with pytest.raises(ConnectionError,match='account changed'):
        session.step()
    # A different broker server must first have a separately evaluated and
    # certified model: account separation alone cannot authorize LIVE.
    with pytest.raises(ValueError, match='different broker server'):
        LiveSession(session.cfg,broker)
    assert len(list((tmp_path/'live_state').glob('111_*_EURUSD.json')))==1


def test_unexpected_partial_fill_locks_live_session_for_manual_reconciliation(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    state={'request_id':'QRL123','result':{'status':'PARTIAL_FILL_UNSUPPORTED'}}
    with pytest.raises(RuntimeError,match='manual reconciliation'):
        session._reconcile_pending(state,broker.account())


def test_live_order_deviation_cannot_exceed_risk_slippage_bound():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match='order_deviation_points'):
        QuantRLConfig.model_validate({
            'live': {'order_deviation_points': 21},
            'execution': {'max_slippage_points': 20},
        })


def test_live_stressed_loss_includes_adverse_entry_and_stop_slippage(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    observed=[]
    original=broker._mt5.order_calc_profit
    def capture(side,symbol,volume,entry,sl):
        observed.append((float(entry),float(sl)))
        return original(side,symbol,volume,entry,sl)
    broker._mt5.order_calc_profit=capture
    result=session.step()
    assert result['status']=='FILLED'
    assert len(observed)==2
    slippage=session.cfg.execution.max_slippage_points*broker.spec.point
    assert observed[1][0]==pytest.approx(observed[0][0]+slippage)
    assert observed[1][1]==pytest.approx(observed[0][1]-slippage)


def test_live_rejects_invalid_broker_free_margin(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    original=broker.account
    def invalid_margin_account():
        account=original()
        account.margin_free=float('nan')
        return account
    broker.account=invalid_margin_account
    result=session.step()
    assert result['status']=='VETOED_BROKER_CALC'
    assert broker.calls==0

def _pending_recovery_fixture(tmp_path):
    broker=FakeBroker()
    session=setup_session(tmp_path,broker,'BUY')
    broker._mt5.DEAL_ENTRY_IN=0
    broker._mt5.DEAL_TYPE_BUY=0
    broker._mt5.DEAL_TYPE_SELL=1
    broker._mt5.POSITION_TYPE_BUY=0
    broker._mt5.POSITION_TYPE_SELL=1
    state={'status':'PENDING','request_id':'QRL123','submit_time_s':int(time.time()),
        'last_bar_time_s':int(time.time()),'requested_volume_lots':0.1,
        'requested_direction':'BUY','requested_sl_price':1.09,
        'requested_tp_price':1.12,'broker_point':0.00001,
        'broker_tick_size':0.00001}
    return broker,session,state



def test_live_does_not_mark_entry_filled_without_broker_confirmed_sl_tp(tmp_path):
    broker=FakeBroker()
    broker.position_protection_override=(0.0,0.0)
    session=setup_session(tmp_path,broker,'BUY')
    with pytest.raises(RuntimeError,match='SL/TP protection is missing'):
        session.step()
    import json
    state=json.loads((tmp_path/'live_state'/'111_EURUSD.json').read_text())
    assert state['status']=='PENDING'
    assert state['result']['status']=='FILLED'
    assert broker.calls==1


def test_live_rejects_broker_modified_sl_tp(tmp_path):
    broker=FakeBroker()
    broker.position_protection_override=(1.0,2.0)
    session=setup_session(tmp_path,broker,'BUY')
    with pytest.raises(RuntimeError,match='SL/TP protection is missing'):
        session.step()
    assert broker.calls==1


def test_pending_recovery_requires_confirmed_sl_tp(tmp_path):
    broker,session,state=_pending_recovery_fixture(tmp_path)
    state.update(requested_sl_price=1.09,requested_tp_price=1.12,
        broker_point=0.00001,broker_tick_size=0.00001)
    broker._positions=[SimpleNamespace(magic=session.cfg.live.magic,
        symbol='EURUSD',comment='QRL123',type=0,volume=0.1,
        sl=0.0,tp=0.0,ticket=900)]
    broker.recent_deals=lambda since:[SimpleNamespace(
        magic=session.cfg.live.magic,symbol='EURUSD',comment='QRL123',
        entry=0,type=0,volume=0.1,ticket=503)]
    with pytest.raises(RuntimeError,match='SL/TP protection is missing'):
        session._reconcile_pending(state,broker.account())


def test_pending_recovery_rejects_partial_deal_volume(tmp_path):
    broker,session,state=_pending_recovery_fixture(tmp_path)
    broker.recent_deals=lambda since:[SimpleNamespace(
        magic=session.cfg.live.magic,symbol='EURUSD',comment='QRL123',
        entry=0,type=0,volume=0.04,ticket=501)]
    with pytest.raises(RuntimeError,match='not an exact full fill'):
        session._reconcile_pending(state,broker.account())


def test_pending_recovery_rejects_wrong_deal_direction(tmp_path):
    broker,session,state=_pending_recovery_fixture(tmp_path)
    broker.recent_deals=lambda since:[SimpleNamespace(
        magic=session.cfg.live.magic,symbol='EURUSD',comment='QRL123',
        entry=0,type=1,volume=0.1,ticket=502)]
    with pytest.raises(RuntimeError,match='direction is inconsistent'):
        session._reconcile_pending(state,broker.account())


def test_pending_recovery_accepts_only_exact_full_volume_deal(tmp_path):
    broker,session,state=_pending_recovery_fixture(tmp_path)
    broker._positions=[SimpleNamespace(magic=session.cfg.live.magic,
        symbol='EURUSD',comment='QRL123',type=0,volume=0.1,
        sl=1.09,tp=1.12,ticket=903)]
    broker.recent_deals=lambda since:[SimpleNamespace(
        magic=session.cfg.live.magic,symbol='EURUSD',comment='QRL123',
        entry=0,type=0,volume=0.1,ticket=503)]
    result=session._reconcile_pending(state,broker.account())
    assert result['status']=='RECOVERED_FILLED'
    assert state['status']=='FILLED'
    assert state['recovered_executed_volume_lots']==pytest.approx(0.1)
    assert state['recovered_deal_ids']==[503]


def test_pending_recovery_rejects_partial_open_position(tmp_path):
    broker,session,state=_pending_recovery_fixture(tmp_path)
    broker._positions=[SimpleNamespace(
        magic=session.cfg.live.magic,symbol='EURUSD',comment='QRL123',
        type=0,volume=0.04)]
    with pytest.raises(RuntimeError,match='identity, direction, or volume is inconsistent'):
        session._reconcile_pending(state,broker.account())

