"""Point-in-time broker affordability observations and explainable C16 vetoes."""
from dataclasses import replace
import numpy as np
import pytest

from quant_rl.config.loader import ConfigLoader
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from quant_rl.execution.accounting import PortfolioAccountingSimulator, summarize_veto_context
from quant_rl.execution.examiner import PortfolioStateSnapshot
from quant_rl.intelligence.policy import EconomicFeatures
from quant_rl.research.evolution import architecture_hash
from quant_rl.intelligence.policy import PPOSettings
from tests.helpers_current import CFG, obs_for_path, proposal_for


def _features(obs, spec, train=None):
    cfg = ConfigLoader.load(CFG)
    return EconomicFeatures(obs, np.arange(obs.num_bars) if train is None else train, 10000.,
        symbol_spec=spec, account_currency=cfg.execution.account_currency,
        min_risk_pct=cfg.intelligence.position_sizing.min_risk_per_trade_pct,
        max_risk_pct=cfg.intelligence.position_sizing.max_risk_per_trade_pct)


def _state(equity=10000.):
    return PortfolioStateSnapshot(current_time_s=0, balance=equity, equity=equity,
        peak_equity=equity, day_start_equity=equity, week_start_equity=equity,
        open_positions=(), consecutive_losses=0, last_bar_time_s=0,
        timeframe_seconds=14400)


def test_affordability_observation_tracks_equity_and_broker_minimum():
    obs = obs_for_path([(1.1, 1.1001, 1.0999, 1.1)] * 15)
    spec = default_symbol_spec('EURUSD')
    f = _features(obs, spec)
    high_equity = f(3, _state(10000.))[-4:]
    low_equity = f(3, _state(1000.))[-4:]
    assert high_equity[0] > low_equity[0]
    assert high_equity[1] > low_equity[1]
    assert high_equity[1] > high_equity[0]  # 1% can afford more stop than 0.1%
    larger_minimum = _features(obs, replace(spec, volume_min=.1))(3, _state(10000.))[-4:]
    assert larger_minimum[1] < high_equity[1]
    np.testing.assert_allclose(larger_minimum[2:], high_equity[2:])
    assert np.all(np.isfinite(high_equity))
    assert f.n_features == len(f(3, _state()))


def test_future_changes_do_not_affect_point_in_time_broker_state():
    obs = obs_for_path([(1.1, 1.1001, 1.0999, 1.1)] * 15)
    i = 6
    spec = default_symbol_spec('EURUSD')
    initial = _features(obs, spec)
    mutated = replace(obs, close=np.concatenate([obs.close[:i+1],
                         np.full(obs.num_bars-i-1, 2.0)]))
    future = _features(mutated, spec, np.arange(0, i+1))
    # Training normalizer uses only training bars, not the corrupted future.
    baseline = _features(obs, spec, np.arange(0, i+1))
    np.testing.assert_allclose(future(i, _state()), baseline(i, _state()), atol=0, rtol=0)
    # Actual current spread matters, not spread from future bars.
    mutated_spread = replace(obs, spread_price=np.concatenate([obs.spread_price[:i+1],
                              np.full(obs.num_bars-i-1, .1)]))
    another = _features(mutated_spread, spec, np.arange(0, i+1))
    np.testing.assert_allclose(another(i, _state())[-4:], baseline(i, _state())[-4:])


def test_c16_reports_exact_economically_feasible_stop_and_risk():
    obs = obs_for_path([(1.1, 1.1001, 1.0999, 1.1)] * 10)
    cfg = ConfigLoader.load(CFG)
    spec = default_symbol_spec('EURUSD')
    # A stop whose min-lot risk is beyond the 0.1% model risk request.
    sim = PortfolioAccountingSimulator(cfg.execution).simulate(
        obs, lambda i,s:replace(proposal_for(obs,i,2,.001),
                        stop_distance_price=.012, stop_units=6.0), spec,
        start_index=0, end_index=9)
    v = next(e for e in sim.decision_events if 'C16' in e.veto_codes)
    measured = dict(v.measured)
    assert measured['raw_requested_lots'] < spec.volume_min
    assert measured['volume_shortfall_lots'] == pytest.approx(
        spec.volume_min - measured['raw_requested_lots'])
    assert measured['max_feasible_stop_pips_at_requested_risk'] == pytest.approx(
        measured['requested_stop_pips'] * measured['raw_requested_lots'] / spec.volume_min)
    assert measured['required_risk_fraction_for_min_lot_at_requested_stop'] > .001
    assert measured['max_feasible_stop_pips_at_requested_risk'] < measured['requested_stop_pips']
    by_check = summarize_veto_context(sim.decision_events)['C16']
    assert by_check['mean_volume_shortfall_lots'] > 0
    assert by_check['mean_max_feasible_stop_pips_at_requested_risk'] > 0
    assert sim.ending_balance == pytest.approx(sim.starting_balance)
    assert sim.learning_reward_account_ccy < 0


def test_broker_feature_contract_changes_architecture_identity():
    obs = obs_for_path([(1.1, 1.1001, 1.0999, 1.1)] * 12)
    cfg = ConfigLoader.load(CFG)
    spec = default_symbol_spec('EURUSD')
    features = _features(obs, spec)
    h = architecture_hash(cfg, features, PPOSettings())
    assert len(h) == 16
    assert h != architecture_hash(cfg.model_copy(update={'execution':cfg.execution.model_copy(
        update={'account_currency':'EUR'})}),features,PPOSettings())


def test_c16_fields_are_present_in_csv_report(tmp_path):
    import csv
    from quant_rl.reporting.economic_report import write_economic_report
    obs = obs_for_path([(1.1, 1.1001, 1.0999, 1.1)] * 10)
    cfg = ConfigLoader.load(CFG)
    sim = PortfolioAccountingSimulator(cfg.execution).simulate(
        obs, lambda i,s:replace(proposal_for(obs,i,2,.001),
                 stop_distance_price=.012, stop_units=6.0),
        default_symbol_spec('EURUSD'), start_index=0, end_index=9)
    output = write_economic_report(tmp_path, 'EURUSD', sim, origin='SYNTHETIC_SIMULATOR',
         settings={}, period={'test_start_utc':'2025','test_end_utc':'2026'},
         qualified=False, reasons=['SYNTHETIC'], trial_report=[])
    with (output / 'veto_by_check.csv').open(newline='', encoding='utf-8') as file:
        row = next(csv.DictReader(file))
    assert row['check'] == 'C16'
    assert float(row['mean_volume_shortfall_lots']) > 0
    assert float(row['mean_max_feasible_stop_pips_at_requested_risk']) > 0
    assert float(row['mean_required_risk_fraction_for_min_lot_at_requested_stop']) > 0
