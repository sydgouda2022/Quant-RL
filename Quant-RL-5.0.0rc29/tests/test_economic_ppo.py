"""Economic trading lifecycle, PPO and report regression tests for v5."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from quant_rl.config.loader import ConfigLoader
from quant_rl.config.schema import QuantRLConfig
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from tests._fixtures import SimulatedFXDataSource
from quant_rl.contracts import ActionProposal, TradeDirection
from quant_rl.intelligence.actions import HybridAction,to_proposal
from quant_rl.intelligence.model import HybridActorCritic
from quant_rl.intelligence.policy import EconomicFeatures,PPOSettings,PPOTrader,advantage_estimates
from quant_rl.execution.accounting import PortfolioAccountingSimulator
from quant_rl.market.state_builder import ObservationBuilder
from quant_rl.research.evolution import chronological_split,select_champion,ModelPopulation
from quant_rl.research.optimizer import trial_decision_diagnostics
from quant_rl.reporting.economic_report import write_economic_report
from tests.helpers_current import obs_for_path,proposal_for,CFG


def test_hybrid_action_continuous_bounded_and_no_trade():
    cfg=ConfigLoader.load(CFG)
    obs=obs_for_path([(1.1,1.102,1.098,1.1)]*12)
    p=to_proposal(HybridAction(1,.34,.89,.5),obs,3,cfg.intelligence.geometry,cfg.intelligence.position_sizing)
    assert p.direction == TradeDirection.BUY
    assert cfg.intelligence.geometry.min_stop_units < p.stop_units < cfg.intelligence.geometry.max_stop_units
    assert p.stop_distance_price == pytest.approx(p.stop_units*.002)
    assert p.requested_risk_fraction <= .01
    no=to_proposal(HybridAction(0,.1,.9,.5),obs,4,cfg.intelligence.geometry,cfg.intelligence.position_sizing)
    assert no.stop_distance_price==0 and no.requested_risk_fraction==0


def test_sampled_policy_logprob_matches_training_distribution():
    model=HybridActorCritic(9, 16)
    x=torch.zeros((1,9),dtype=torch.float32)
    for deterministic in (False, True):
        for _ in range(8):
            action,logp,_,entropy=model.act(x,deterministic=deterministic)
            manual_logp,manual_entropy=model.evaluate_actions(
                x,torch.tensor([action.direction_index]),
                torch.tensor([[action.stop_fraction,action.target_fraction,action.risk_fraction]]))
            assert logp.item()==pytest.approx(manual_logp.item(),abs=1e-5)
            assert entropy.item()==pytest.approx(manual_entropy.item(),abs=1e-5)


def test_cpu_thread_configuration_is_validated():
    from pydantic import ValidationError
    from quant_rl.config.schema import PPOConfig
    assert PPOConfig().cpu_threads==1
    with pytest.raises(ValidationError): PPOConfig(cpu_threads=0)


def test_conditional_action_logp_ignores_geometry_when_abstaining():
    m=HybridActorCritic(12,16)
    state=torch.zeros((2,12))
    dirs=torch.tensor([0,1])
    a=torch.tensor([[.2,.3,.4],[.2,.3,.4]])
    b=torch.tensor([[.8,.7,.6],[.2,.3,.4]])
    loga,ea=m.evaluate_actions(state,dirs,a)
    logb,eb=m.evaluate_actions(state,dirs,b)
    assert loga[0].item()==pytest.approx(logb[0].item())
    assert loga[1].item()==pytest.approx(logb[1].item())
    assert torch.isfinite(loga).all()


def test_variable_time_advantage_uses_elapsed_bars():
    r=np.array([1.,0.],dtype=np.float32)
    v=np.array([0.,0.],dtype=np.float32)
    indices=np.array([0,10])
    a,returns=advantage_estimates(r,v,indices,.9,.95,np.array([True,True]))
    assert returns[0]==pytest.approx(1.)
    r=np.array([0.,1.],dtype=np.float32)
    a,returns=advantage_estimates(r,v,indices,.9,1.,np.array([True,True]))
    assert returns[0]==pytest.approx(.9**10)


def test_single_position_avoids_decisions_and_penalty_during_holding():
    prices=[(1.1,1.1001,1.0999,1.1)]*11
    prices[4]=(1.1,1.11,1.0999,1.109)
    obs=obs_for_path(prices)
    cfg=ConfigLoader.load(CFG)
    calls=[]
    def policy(i,state):
        calls.append(i)
        return proposal_for(obs,i,2,.005) if i==0 else proposal_for(obs,i,0)
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(obs,policy,default_symbol_spec('EURUSD'),
                                           start_index=0,end_index=10,inactivity_penalty_lots=.1)
    assert sim.executed_trades and sim.executed_trades[0].exit_bar_index==4
    assert 1 not in calls and 2 not in calls and 3 not in calls and 4 not in calls
    assert calls[0]==0 and calls[1]==5
    assert sim.penalty_total_account_ccy == pytest.approx(
        sum(e.spread_penalty_account_ccy for e in sim.decision_events))
    assert sim.learning_reward_account_ccy == pytest.approx(
        sum(t.net_pnl_account_ccy for t in sim.executed_trades)-sim.penalty_total_account_ccy)
    assert sim.ending_balance == pytest.approx(
        sim.starting_balance+sum(t.net_pnl_account_ccy for t in sim.executed_trades))


def test_permanent_loss_breaker_ends_decisions_without_repeated_veto_penalties():
    prices=[(1.1,1.1001,1.0999,1.1)]*10
    prices[1]=(1.1,1.1001,1.095,1.099)
    obs=obs_for_path(prices)
    cfg=ConfigLoader.load(CFG)
    restrictive=cfg.execution.model_copy(update={'max_consecutive_losses':1})
    calls=[]
    def policy(i,state):
        calls.append(i)
        return proposal_for(obs,i,2,.005)
    sim=PortfolioAccountingSimulator(restrictive).simulate(
        obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=9)
    assert sim.executed_trades and sim.executed_trades[0].net_pnl_account_ccy<0
    assert calls==[0]
    assert sim.terminal_halt_reason=='C14_CONSECUTIVE_LOSSES'
    assert sim.terminal_halt_bar_index==1
    assert sim.veto_count==0
    assert sim.penalty_total_account_ccy==0
    assert sim.learning_reward_account_ccy==pytest.approx(sim.executed_trades[0].net_pnl_account_ccy)
    assert sim.equity_curve[-1]==pytest.approx(sim.ending_balance)


def test_permanent_account_drawdown_breaker_ends_decisions():
    prices=[(1.1,1.1001,1.0999,1.1)]*10
    prices[1]=(1.1,1.1001,1.095,1.099)
    obs=obs_for_path(prices)
    cfg=ConfigLoader.load(CFG)
    calls=[]
    def policy(i,state):
        calls.append(i)
        return proposal_for(obs,i,2,.005)
    sim=PortfolioAccountingSimulator(cfg.execution,max_drawdown_pct=.1).simulate(
        obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=9)
    assert calls==[0]
    assert sim.terminal_halt_reason=='C13_MAX_DRAWDOWN'
    assert sim.penalty_total_account_ccy==0


def test_veto_reports_all_failed_checks_and_one_penalty():
    obs=obs_for_path([(1.1,1.1001,1.0999,1.1)]*7)
    cfg=ConfigLoader.load(CFG)
    obs=replace(obs,spread_price=np.full(obs.num_bars,.001))
    def policy(i,state):return proposal_for(obs,i,2,.005)
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(obs,policy,default_symbol_spec('EURUSD'),
                                           start_index=0,end_index=6,inactivity_penalty_lots=.1)
    assert sim.executed_trades == ()
    assert sim.veto_count == 6
    assert len(sim.decision_events)==6
    assert all('C22' in e.veto_codes for e in sim.decision_events)
    assert all(e.failed_check_details for e in sim.decision_events)
    assert all(e.observed_features for e in sim.decision_events)
    assert all(e.observed_spread_pips>0 for e in sim.decision_events)
    assert sim.penalty_total_account_ccy == pytest.approx(sum(e.spread_penalty_account_ccy for e in sim.decision_events))
    assert sim.ending_balance==cfg.execution.starting_balance


def test_optuna_trial_diagnostics_explain_every_veto():
    obs=obs_for_path([(1.1,1.1001,1.0999,1.1)]*7)
    cfg=ConfigLoader.load(CFG)
    obs=replace(obs,spread_price=np.full(obs.num_bars,.001))
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,lambda i,s:proposal_for(obs,i,2,.005),default_symbol_spec('EURUSD'),
        start_index=0,end_index=6,inactivity_penalty_lots=.1)
    x=trial_decision_diagnostics(sim)
    assert x['eligible_decisions']==6
    assert x['vetoed_proposals']==6
    assert x['no_trade_decisions']==0
    assert x['veto_fraction']==pytest.approx(1.)
    assert x['all_failed_check_counts']['C22']==6
    assert x['average_veto_stop_units']>0
    assert x['learning_penalty_account_ccy']==sim.penalty_total_account_ccy


def test_unresolved_trade_not_force_closed_and_no_fabricated_loss():
    obs=obs_for_path([(1.1,1.1001,1.0999,1.1)]*7)
    cfg=ConfigLoader.load(CFG)
    def policy(i,state):return proposal_for(obs,i,2,.005)
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(obs,policy,default_symbol_spec('EURUSD'),
                                           start_index=0,end_index=6)
    assert len(sim.open_positions_at_end)==1
    assert sim.executed_trades==()
    assert len(sim.decision_events)==1
    assert not sim.decision_events[0].resolved
    # Censored does not mean worthless: a known boundary mark is the
    # truncated-episode learning signal, not a realized TP/SL outcome.
    mark=sim.ending_equity-sim.ending_balance
    assert sim.ending_equity != sim.ending_balance
    assert sim.decision_events[0].reward_account_ccy==pytest.approx(mark)
    assert sim.decision_events[0].terminal_mark_to_market_account_ccy==pytest.approx(mark)
    assert sim.learning_reward_account_ccy==pytest.approx(mark)
    assert sim.ending_balance==sim.starting_balance


def test_report_costs_and_balance_reconcile(tmp_path):
    obs=obs_for_path([(1.1,1.1001,1.0999,1.1),
        (1.1,1.11,1.0999,1.109)]*7)
    cfg=ConfigLoader.load(CFG)
    def policy(i,state):return proposal_for(obs,i,2,.005) if i%3==0 else proposal_for(obs,i,0)
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(obs,policy,default_symbol_spec('EURUSD'),
                                           start_index=0,end_index=13)
    path=write_economic_report(tmp_path,'EURUSD',sim,origin='SYNTHETIC_SIMULATOR',settings={},
         period={'test_start_utc':'2025','test_end_utc':'2026'},qualified=False,reasons=['DEMO'],trial_report=[])
    x=json.loads((path/'summary.json').read_text())
    assert x['starting_balance']+x['net_realized_pnl']==pytest.approx(x['ending_balance'])
    assert x['gross_realized_pnl']-x['total_trading_costs']==pytest.approx(x['net_realized_pnl'])
    assert (path/'trades.csv').exists() and (path/'decisions.csv').exists()


def test_model_population_atomic_and_checksum(tmp_path):
    p=ModelPopulation(tmp_path)
    snap={'weights':{'a':torch.arange(3.)},'mean':torch.zeros(1),'std':torch.ones(1)}
    p.publish({'EURUSD':(snap,{'arch':'test','current_score':.2})},{},'EURUSD',{})
    assert p.data['format']==ModelPopulation.FORMAT
    rec=p.data['symbols']['EURUSD']
    assert torch.equal(p.load(rec,'test')['weights']['a'],snap['weights']['a'])
    file=tmp_path/'models'/rec['file'];file.write_bytes(b'TAMPERED')
    with pytest.raises(ValueError,match='checksum'):p.load(rec,'test')


def synthetic_observations():
    s=SimulatedFXDataSource(seed=1,default_count=1800)
    bars=s.fetch_bars('EURUSD','H1',1800)
    spec=s.get_symbol_spec('EURUSD')
    obs=ObservationBuilder(['H4'],8).build_dataset(base_bars=bars,symbol='EURUSD',primary_timeframe='H4',spec=spec)
    return obs,spec


def test_ppo_training_validation_and_frozen_replay():
    cfg=ConfigLoader.load(CFG)
    obs,spec=synthetic_observations()
    train,val,test=chronological_split(obs.time_s,40)
    policy=PPOTrader(EconomicFeatures(obs,train,cfg.execution.starting_balance,
                 symbol_spec=spec,account_currency=cfg.execution.account_currency,
                 min_risk_pct=.1,max_risk_pct=1.),
                 PPOSettings(hidden_dim=16,episodes=1,ppo_epochs=1),seed=42)
    assert np.isfinite(policy.fit(obs,spec,train,val,cfg,seed=42))
    before=policy.snapshot()
    result,_=policy.replay(obs,spec,test,cfg)
    after=policy.snapshot()
    assert all(torch.equal(before['weights'][k],after['weights'][k]) for k in before['weights'])
    assert len(result.time_s_curve)==len(test)


def test_legacy_scoreboard_rejected_instead_of_reusing_incompatible_scores(tmp_path):
    (tmp_path/'best_score.json').write_text(json.dumps({'symbols':{},'champion':None,'cycles':{}}))
    with pytest.raises(ValueError,match='Invalid global'):
        ModelPopulation(tmp_path)


def test_input_normalizer_training_only_and_future_mutation_has_no_effect():
    obs,spec=synthetic_observations()
    train,val,test=chronological_split(obs.time_s,40)
    original=EconomicFeatures(obs,train,10000.,symbol_spec=spec,account_currency='USD',
                             min_risk_pct=.1,max_risk_pct=1.,feature_group='all',sequence_bars=8)
    corrupted=replace(obs,feature_matrix=np.vstack([obs.feature_matrix[:int(train[-1])+1],
             np.full_like(obs.feature_matrix[int(train[-1])+1:],9999.)]))
    other=EconomicFeatures(corrupted,train,10000.,symbol_spec=spec,account_currency='USD',
                             min_risk_pct=.1,max_risk_pct=1.,feature_group='all',sequence_bars=8)
    assert np.array_equal(original.mean,other.mean)
    assert np.array_equal(original.std,other.std)


def test_optuna_trials_explore_feature_profiles_without_final_data():
    from quant_rl.research.optimizer import optimize_candidate
    cfg=ConfigLoader.load(CFG)
    cfg=cfg.model_copy(update={
      'optimization':cfg.optimization.model_copy(update={'trials':2}),
      'ppo':cfg.ppo.model_copy(update={'episodes':1,'ppo_epochs':1,'hidden_dim':16}),
      'research':cfg.research.model_copy(update={'batch_size':128})})
    obs,spec=synthetic_observations()
    train,val,test=chronological_split(obs.time_s,40)
    result=optimize_candidate(cfg,obs,spec,train,val,(None,None,None))
    assert len(result.trial_report)==2
    assert all('feature_group' in x['parameters'] and 'sequence_bars' in x['parameters']
               for x in result.trial_report)
    assert result.trader.features.n_features==result.features.n_features
