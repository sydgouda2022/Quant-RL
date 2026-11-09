"""rc4 replay-cache correctness and parameter-diagnostic regression tests."""
from dataclasses import replace

import numpy as np
import pytest

from quant_rl.config.loader import ConfigLoader
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from quant_rl.execution import accounting as accounting_module
from quant_rl.execution.accounting import PortfolioAccountingSimulator, CausalMarketContextCache
from quant_rl.execution.market_safety import market_context_asof
from quant_rl.research.optimizer import trial_decision_diagnostics
from tests.helpers_current import CFG, obs_for_path, proposal_for


def case():
    obs = obs_for_path([(1.1, 1.1001, 1.0999, 1.1)] * 51)
    # Rejection on every eligible bar: many repeated market-context queries.
    obs = replace(obs, spread_price=np.full(obs.num_bars, .001, dtype=float))
    cfg = ConfigLoader.load(CFG)
    def policy(bar, state):
        return proposal_for(obs, bar, 2, .005)
    return obs, cfg, policy


def test_market_context_cache_matches_uncached_financial_and_veto_results(monkeypatch):
    obs, cfg, policy = case()
    spec = default_symbol_spec('EURUSD')
    engine = PortfolioAccountingSimulator(cfg.execution)
    original = accounting_module.market_context_asof
    calls=[]
    def track(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(accounting_module, 'market_context_asof', track)
    cache=CausalMarketContextCache(obs, cfg.execution)
    def replay(shared=None):
        return engine.simulate(obs,policy,spec,start_index=0,end_index=50,
                               market_context_cache=shared)
    first=replay(cache)
    first_count=len(calls)
    assert first_count > 40
    second=replay(cache)
    assert len(calls)==first_count  # no recomputations for the same bars
    plain=replay()
    assert len(calls)==2*first_count
    for a,b in ((first,second),(first,plain)):
        assert a.veto_breakdown==b.veto_breakdown
        assert a.veto_count==b.veto_count
        assert a.penalty_total_account_ccy==pytest.approx(b.penalty_total_account_ccy)
        assert a.learning_reward_account_ccy==pytest.approx(b.learning_reward_account_ccy)
        assert a.ending_equity==pytest.approx(b.ending_equity)
        np.testing.assert_array_equal(a.equity_curve,b.equity_curve)
        assert a.decision_events == b.decision_events


def test_market_cache_fails_closed_on_another_dataset_or_safety_settings():
    obs, cfg, policy = case()
    cache=CausalMarketContextCache(obs,cfg.execution)
    other=replace(obs,spread_price=obs.spread_price.copy())
    engine=PortfolioAccountingSimulator(cfg.execution)
    with pytest.raises(ValueError,match='different dataset'):
        engine.simulate(other, lambda i,s:proposal_for(other,i,0),
            default_symbol_spec('EURUSD'),start_index=0,end_index=10,
            market_context_cache=cache)
    changed=cfg.execution.model_copy(update={'market_history_bars':256})
    with pytest.raises(ValueError,match='safety configuration'):
        PortfolioAccountingSimulator(changed).simulate(
            obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=10,
            market_context_cache=cache)


def test_precomputed_spread_values_are_equivalent_and_causal():
    obs,cfg,_=case()
    i=39
    params=dict(lookback_bars=cfg.execution.market_history_bars,
                min_history_bars=cfg.execution.market_min_history_bars,
                candidate_windows=cfg.execution.market_window_candidates,
                regime_shift_ratio=cfg.execution.market_regime_shift_ratio)
    a=market_context_asof(obs,i,**params)
    b=market_context_asof(obs,i,spread_values=obs.spread_pips,**params)
    assert a==b
    future=obs.spread_price.copy()
    future[i:]*=10
    modified=replace(obs,spread_price=future)
    assert market_context_asof(modified,i,**params)==a


def test_trial_diagnostics_preserve_multi_check_failure_context():
    obs,cfg,policy=case()
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=50)
    diag=trial_decision_diagnostics(sim)
    assert diag['veto_context_by_check']
    assert diag['vetoed_proposals']==sim.veto_count
    for code,stats in diag['veto_context_by_check'].items():
        assert stats['failure_count']==diag['all_failed_check_counts'][code]
        assert stats['mean_observed_spread_pips']>0
        assert stats['mean_observed_volatility_price']>0
        assert 'mean_proposed_stop_units' in stats
        assert 'mean_proposed_risk_fraction' in stats


def test_per_trial_veto_context_exported_as_structured_json(tmp_path):
    import csv
    import json
    from quant_rl.reporting.economic_report import write_economic_report
    obs,cfg,policy=case()
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=50)
    diagnostic=trial_decision_diagnostics(sim)
    report=write_economic_report(tmp_path,'EURUSD',sim,origin='SYNTHETIC_SIMULATOR',
        settings={},period={'test_start_utc':'2025','test_end_utc':'2026'},
        qualified=False,reasons=['SYNTHETIC_DATA_NOT_CERTIFIED'],
        trial_report=[{'trial':0,'validation_reward_account_ccy':sim.learning_reward_account_ccy,
                       'parameters':{'feature_group':'all','sequence_bars':32},
                       'decision_diagnostics':diagnostic}])
    with (report/'optimization_trials.csv').open(newline='',encoding='utf-8') as f:
        row=next(csv.DictReader(f))
    context=json.loads(row['veto_context_by_check'])
    assert context==diagnostic['veto_context_by_check']
    assert json.loads(row['failed_check_counts'])==diagnostic['all_failed_check_counts']


def test_final_evaluation_aggregated_veto_export(tmp_path):
    import csv
    from quant_rl.reporting.economic_report import write_economic_report
    from quant_rl.execution.accounting import summarize_veto_context
    obs,cfg,policy=case()
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=50)
    output=write_economic_report(tmp_path,'EURUSD',sim,origin='SYNTHETIC_SIMULATOR',
        settings={},period={'test_start_utc':'2025','test_end_utc':'2026'},
        qualified=False,reasons=['SYNTHETIC_DATA_NOT_CERTIFIED'],trial_report=[])
    with (output/'veto_by_check.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    context=summarize_veto_context(sim.decision_events)
    assert {r['check'] for r in rows}==set(context)
    for row in rows:
        assert int(row['failure_count'])==context[row['check']]['failure_count']
        assert float(row['mean_observed_spread_pips'])==pytest.approx(context[row['check']]['mean_observed_spread_pips'])


def test_veto_includes_exact_broker_volume_evidence():
    obs,cfg,policy=case()
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,policy,default_symbol_spec('EURUSD'),start_index=0,end_index=50)
    veto=next(e for e in sim.decision_events if e.status=='VETO')
    measurements=dict(veto.measured)
    assert measurements['raw_requested_lots'] >= 0
    assert measurements['broker_min_volume_lots'] > 0
    assert measurements['broker_volume_step_lots'] > 0
    assert measurements['requested_stop_pips'] > 0
    assert measurements['broker_min_volume_lots']==pytest.approx(default_symbol_spec('EURUSD').volume_min)


def test_broker_min_lot_rejection_preserves_requested_volume():
    from quant_rl.execution.sizing import calculate_broker_position_size
    spec=default_symbol_spec('EURUSD')
    q=calculate_broker_position_size(
        equity=10000.,requested_risk_fraction=.001,max_risk_fraction=.01,
        stop_distance_price=.03,entry_price=1.1,symbol_spec=spec,
        account_currency='USD')
    assert not q.valid
    assert q.volume_lots==0
    assert 0 < q.raw_requested_lots < spec.volume_min
    assert q.actual_risk_account_ccy > q.risk_budget_account_ccy
