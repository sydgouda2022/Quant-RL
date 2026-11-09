"""Right-censored trade equity must drive economic learning without fake exits."""
from __future__ import annotations
from dataclasses import replace
import csv
import json
import numpy as np
import pytest

from quant_rl.config.loader import ConfigLoader
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from quant_rl.execution.accounting import PortfolioAccountingSimulator
from quant_rl.intelligence.policy import advantage_estimates
from quant_rl.reporting.economic_report import write_economic_report
from tests.helpers_current import CFG, obs_for_path, proposal_for


def _censored_sim(final_mid: float):
    # Entry at 1.10000, SL ~1.097, TP ~1.106; no boundary touch.
    initial=(1.10000,1.1001,1.0999,1.10000)
    later=(final_mid,final_mid+.0001,final_mid-.0001,final_mid)
    obs=obs_for_path([initial]+[later]*7)
    cfg=ConfigLoader.load(CFG)
    calls=[]
    def policy(bar,state):
        calls.append(bar)
        return proposal_for(obs,bar,2,.005)
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=7,
        inactivity_penalty_lots=cfg.ppo.penalty_reference_lots)
    return sim,cfg,calls


@pytest.mark.parametrize('final_mid,expected_sign',[(1.1010,1),(1.0990,-1)])
def test_open_winning_or_losing_mark_is_learnable_not_realized(final_mid,expected_sign):
    sim,cfg,calls=_censored_sim(final_mid)
    assert calls == [0]  # no additional inference while position is held
    assert not sim.executed_trades
    assert len(sim.open_positions_at_end)==1
    assert sim.ending_balance==pytest.approx(sim.starting_balance)
    mark=sim.ending_equity-sim.ending_balance
    assert mark*expected_sign>0
    event,=sim.decision_events
    assert event.status=='OPEN' and event.resolved is False
    assert event.resolved_bar_index==7
    assert event.terminal_mark_to_market_account_ccy==pytest.approx(mark)
    assert event.reward_account_ccy==pytest.approx(mark)
    assert sim.learning_reward_account_ccy==pytest.approx(mark)
    assert sim.penalty_total_account_ccy==0
    # A terminal marked-to-market reward is directly usable by PPO as a
    # truncated episode with no subsequent critic bootstrap.
    advantages,returns=advantage_estimates(
        np.array([mark],dtype=np.float32),np.array([0.],dtype=np.float32),
        np.array([0],dtype=np.int64),.995,.95,
        np.array([event.resolved or event.terminal_mark_to_market_account_ccy is not None]))
    assert float(returns[0])==pytest.approx(mark,abs=1e-4)


def test_censored_mark_and_abstention_penalties_reconcile(tmp_path):
    obs=obs_for_path([(1.1,1.1001,1.0999,1.1)]*9)
    cfg=ConfigLoader.load(CFG)
    def policy(i,s):
        return proposal_for(obs,i,0) if i<2 else proposal_for(obs,i,2,.005)
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=8,
        inactivity_penalty_lots=cfg.ppo.penalty_reference_lots)
    assert len(sim.open_positions_at_end)==1 and len(sim.decision_events)==3
    assert sim.penalty_total_account_ccy>0
    assert sim.learning_reward_account_ccy==pytest.approx(
        sim.ending_equity-sim.starting_balance-sim.penalty_total_account_ccy)
    assert sum(e.reward_account_ccy for e in sim.decision_events)==pytest.approx(
        sim.learning_reward_account_ccy)
    folder=write_economic_report(tmp_path,'EURUSD',sim,origin='SYNTHETIC_SIMULATOR',
        settings={},period={'test_start_utc':'2025','test_end_utc':'2026'},
        qualified=False,reasons=['SYNTHETIC'],trial_report=[])
    result=json.loads((folder/'summary.json').read_text())
    assert result['terminal_censored_learning_mark']==pytest.approx(
        sim.ending_equity-sim.ending_balance)
    assert result['total_learning_reward']==pytest.approx(sim.learning_reward_account_ccy)
    with (folder/'decisions.csv').open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    assert rows[-1]['status']=='OPEN' and rows[-1]['resolved']=='False'
    assert float(rows[-1]['terminal_mark_to_market_account_ccy'])==pytest.approx(
        sim.ending_equity-sim.ending_balance)
    assert not list(csv.DictReader((folder/'trades.csv').open(newline='')))


def test_closed_trades_still_have_only_realized_rewards():
    prices=[(1.1,1.1001,1.0999,1.1)]*8
    prices[2]=(1.1,1.1080,1.0999,1.107)
    obs=obs_for_path(prices)
    cfg=ConfigLoader.load(CFG)
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,lambda i,s:proposal_for(obs,i,2,.005) if i==0 else proposal_for(obs,i,0),
        default_symbol_spec('EURUSD'),start_index=0,end_index=7)
    assert len(sim.executed_trades)==1
    ev=sim.decision_events[0]
    assert ev.resolved is True and ev.status=='CLOSED'
    assert ev.terminal_mark_to_market_account_ccy is None
    assert ev.reward_account_ccy==pytest.approx(sim.executed_trades[0].net_pnl_account_ccy)


def test_ppo_uses_completed_trade_rewards_for_weight_updates():
    """Integration: profitable settled trades must reach PPO's learnable batch."""
    import torch
    from quant_rl.intelligence.policy import EconomicFeatures, PPOTrader, PPOSettings
    prices=[(1.1,1.1001,1.0999,1.1)] + [(1.1,1.14,1.09995,1.1)]*19
    obs=obs_for_path(prices)
    cfg=ConfigLoader.load(CFG)
    spec=default_symbol_spec('EURUSD')
    train=np.arange(0,12,dtype=np.int64)
    val=np.arange(12,20,dtype=np.int64)
    features=EconomicFeatures(obs,train,cfg.execution.starting_balance,
        symbol_spec=spec,account_currency='USD',min_risk_pct=.1,max_risk_pct=1.)
    settings=PPOSettings(hidden_dim=16,episodes=1,ppo_epochs=1,learning_rate=.001)
    trader=PPOTrader(features,settings,seed=42)
    with torch.no_grad():
        trader.model.direction.weight.zero_()
        trader.model.direction.bias.copy_(torch.tensor([-15.,15.,-15.]))
    before=trader.snapshot()
    training,_=trader.replay(obs,spec,train,cfg,deterministic=False)
    closed=[event for event in training.decision_events if event.status=='CLOSED']
    assert closed
    assert all(event.resolved and event.reward_account_ccy>0 for event in closed)
    trader.fit(obs,spec,train,val,cfg,seed=42)
    # At least some of the network must change as the advantage and critic
    # receive completed net trade P&L (previously all trades were masked out).
    changed=sum(not torch.equal(t,before['weights'][k])
                for k,t in trader.snapshot()['weights'].items())
    assert changed>0
