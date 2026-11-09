"""Independent failure stimuli for each C1–C26 condition.

Offline unit tests cannot establish that live broker state is fetched reliably.
These tests verify each local predicate can reject a deliberately invalid input.
"""
from dataclasses import replace
from pathlib import Path
import pytest
from quant_rl.config.loader import ConfigLoader
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from quant_rl.execution.examiner import OpenPositionSnapshot, PortfolioStateSnapshot, PreTradeExaminer
from tests.helpers_current import obs_for_path, proposal_for

CFG=Path(__file__).resolve().parents[1] / 'config.yaml'


def other_position(symbol='AUDUSD', open_risk=10., margin=100.):
    return OpenPositionSnapshot(position_id=1,symbol=symbol,direction='BUY',volume_lots=0.1,
        entry_price=1.0,stop_price=0.99,target_price=1.1,entry_time_s=1700000000,
        open_risk_account_ccy=open_risk,margin_used_account_ccy=margin)


@pytest.mark.parametrize('num',list(range(1,27)),ids=[f'C{i}' for i in range(1,27)])
def test_each_veto_is_exercised_separately(num,monkeypatch):
    cfg=ConfigLoader.load(CFG)
    obs=obs_for_path([(1.1,1.101,1.099,1.1)]*8)
    p=proposal_for(obs,idx=1,action_index=2,risk=.005)
    t=int(obs.decision_time_s[1])
    snap=PortfolioStateSnapshot(current_time_s=t,balance=10000,equity=10000,peak_equity=10000,
          day_start_equity=10000,week_start_equity=10000,last_bar_time_s=t,timeframe_seconds=3600)
    spec=default_symbol_spec('EURUSD')
    kwargs={'mid_price':1.1,'spread_pips':1.0,'volatility_price':.002}
    if num==1: snap=replace(snap,kill_switch_engaged=True)
    elif num==2: snap=replace(snap,broker_healthy=False)
    elif num==3: snap=replace(snap,last_bar_time_s=t-15000)
    elif num==4: spec=replace(spec,trade_mode=0)
    elif num==5: snap=replace(snap,current_time_s=1704542400,last_bar_time_s=1704542400)
    elif num==6: snap=replace(snap,economic_blackout_active=True)
    elif num==7: snap=replace(snap,open_positions=(other_position('EURUSD'),))
    elif num==8:
        snap=replace(snap,open_positions=(other_position(),))
        cfg=cfg.model_copy(update={'execution':cfg.execution.model_copy(update={'max_open_positions':1})})
    elif num==9: snap=replace(snap,open_positions=(other_position(open_risk=1000),))
    elif num==10:
        snap=replace(snap,open_positions=(other_position('EURGBP'),))
        cfg=cfg.model_copy(update={'execution':cfg.execution.model_copy(update={'max_currency_concentration':1})})
    elif num==11: snap=replace(snap,day_start_equity=11000)
    elif num==12: snap=replace(snap,week_start_equity=11000)
    elif num==13: snap=replace(snap,peak_equity=16000)
    elif num==14: snap=replace(snap,consecutive_losses=8)
    elif num==15: p=replace(p,requested_risk_fraction=.02)
    elif num==16: spec=replace(spec,volume_min=10.)
    elif num==17: spec=replace(spec,volume_max=.1)
    elif num==18: snap=replace(snap,open_positions=(other_position(margin=8500),))
    elif num==19:
        cfg=cfg.model_copy(update={'execution':cfg.execution.model_copy(update={'maximum_leverage':1.})})
    elif num==20: p=replace(p,stop_distance_price=.00001)
    elif num==21: p=replace(p,target_distance_price=.00001)
    elif num==22: kwargs['spread_pips']=12.
    elif num==23: kwargs['volatility_price']=.000001
    elif num==24:
        import quant_rl.execution.examiner as module
        orig=module.calculate_broker_position_size
        def invalid_sizer(**args):
            q=orig(**args)
            return replace(q,actual_risk_account_ccy=1000.)
        monkeypatch.setattr(module,'calculate_broker_position_size',invalid_sizer)
    elif num==25: p=replace(p,target_distance_price=-.0001)
    elif num==26: p=replace(p,target_distance_price=.0003)
    verdict=PreTradeExaminer(cfg.execution).inspect(p,snap,spec,**kwargs)
    assert verdict.checks_passed[f'C{num}'] is False, f'C{num} did not veto the attempted violation'
    assert not verdict.approved
