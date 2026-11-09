"""Market-conditioned safety must be causal and cannot relax hard risk limits."""
from dataclasses import replace
import numpy as np
import pytest

from quant_rl.config.loader import ConfigLoader
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from quant_rl.execution.accounting import PortfolioAccountingSimulator
from quant_rl.execution.examiner import PortfolioStateSnapshot, PreTradeExaminer
from quant_rl.execution.market_safety import market_context_asof
from tests.helpers_current import obs_for_path, proposal_for, CFG


def setup_obs(n=48):
    obs = obs_for_path([(1.1, 1.1001, 1.0999, 1.1)] * n)
    return replace(obs, spread_price=np.full(n, 0.0001))  # 1 pip


def inspect(obs, idx, *, vol=None, spread=1.0, cfg=None, context=None):
    cfg = cfg or ConfigLoader.load(CFG)
    p = proposal_for(obs, idx=idx, action_index=2, risk=0.005)
    t = int(obs.decision_time_s[idx])
    snap = PortfolioStateSnapshot(current_time_s=t, balance=10000, equity=10000,
        peak_equity=10000, day_start_equity=10000, week_start_equity=10000,
        last_bar_time_s=t, timeframe_seconds=3600)
    if context is None:
        context = market_context_asof(obs, idx, lookback_bars=cfg.execution.market_history_bars,
            min_history_bars=cfg.execution.market_min_history_bars)
    return PreTradeExaminer(cfg.execution).inspect(p, snap, default_symbol_spec('EURUSD'),
            mid_price=1.1, spread_pips=spread, volatility_price=vol or .002,
            market_context=context)


def test_c22_market_cost_uses_prior_history_and_preserves_conservative_fallback():
    obs = setup_obs()
    dynamic = inspect(obs, 30)
    insufficient_history = inspect(obs, 19)
    assert dynamic.round_trip_cost_pips > 3.0
    assert dynamic.checks_passed['C22'] is True
    assert dynamic.safety_mode == 'market_adaptive'
    assert insufficient_history.checks_passed['C22'] is False
    assert insufficient_history.safety_mode == 'conservative_cold_start'


def test_spread_shock_cannot_expand_its_own_acceptance_band():
    obs = setup_obs()
    verdict = inspect(obs, 30, spread=2.5)
    assert not verdict.checks_passed['C22']
    assert verdict.veto_code == 'C22'
    assert 'spread regime acceptable=False' in verdict.veto_reason


def test_volatility_shock_rejects_and_dampens_risk():
    obs = setup_obs()
    normal = inspect(obs, 30)
    medium = inspect(obs, 30, vol=.003)
    extreme = inspect(obs, 30, vol=.02)
    assert medium.effective_risk_ceiling_fraction < normal.effective_risk_ceiling_fraction
    assert medium.actual_risk_account_ccy <= normal.actual_risk_account_ccy
    assert not extreme.checks_passed['C23']


def test_hard_drawdown_and_kill_switch_remain_inviolable():
    obs = setup_obs()
    cfg=ConfigLoader.load(CFG)
    idx=30
    context=market_context_asof(obs, idx, lookback_bars=cfg.execution.market_history_bars,
                          min_history_bars=cfg.execution.market_min_history_bars)
    p=proposal_for(obs, idx=idx, action_index=2, risk=.005)
    t=int(obs.decision_time_s[idx])
    snap=PortfolioStateSnapshot(current_time_s=t, balance=10000, equity=10000,
        peak_equity=10000, day_start_equity=10000, week_start_equity=10000,
        last_bar_time_s=t, timeframe_seconds=3600, kill_switch_engaged=True)
    ex=PreTradeExaminer(cfg.execution)
    v=ex.inspect(p,snap,default_symbol_spec('EURUSD'),1.1,1.,.002,market_context=context)
    assert v.veto_code=='C1'
    snap=replace(snap, kill_switch_engaged=False, day_start_equity=11000)
    v=ex.inspect(p,snap,default_symbol_spec('EURUSD'),1.1,1.,.002,market_context=context)
    assert not v.checks_passed['C11']


def test_market_reference_ignores_all_future_bars_and_uses_completed_only():
    obs=setup_obs(55)
    idx=30
    a=market_context_asof(obs, idx, lookback_bars=128, min_history_bars=20)
    mut=replace(obs, spread_price=np.concatenate((obs.spread_price[:idx+1],
                    np.full(obs.num_bars-idx-1, .02))))
    b=market_context_asof(mut, idx, lookback_bars=128, min_history_bars=20)
    assert a==b
    assert a.sample_count==30
    assert market_context_asof(obs, 19, lookback_bars=128, min_history_bars=20) is None


def test_market_context_stale_time_refused():
    obs=setup_obs()
    context=market_context_asof(obs, 30, lookback_bars=128, min_history_bars=20)
    assert context is not None
    with pytest.raises(ValueError,match='decision timestamp'):
        inspect(obs,31,context=context)


def test_single_account_uses_market_context_in_real_replay():
    obs=setup_obs(45)
    cfg=ConfigLoader.load(CFG)
    p=proposal_for(obs, idx=30, action_index=2, risk=.005)
    no_trades=[proposal_for(obs,idx=i,action_index=0) for i in range(31,44)]
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(obs,[p,*no_trades],default_symbol_spec('EURUSD'))
    assert any(s=='EXECUTED' for _,_,s in sim.bar_decisions)


def test_market_history_configuration_requires_consistent_window():
    from quant_rl.config.schema import ExecutionConfig
    with pytest.raises(ValueError,match='market_min_history_bars'):
        ExecutionConfig(market_history_bars=20,market_min_history_bars=40)


def test_adaptive_stress_rechecks_absolute_transaction_cost():
    obs = setup_obs(45)
    idx=30
    cfg = ConfigLoader.load(CFG)
    p = proposal_for(obs, idx=idx, action_index=2, risk=.005)
    t=int(obs.decision_time_s[idx])
    context=market_context_asof(obs,idx,lookback_bars=cfg.execution.market_history_bars,
        min_history_bars=cfg.execution.market_min_history_bars)
    snap=PortfolioStateSnapshot(current_time_s=t,balance=10000,equity=10000,
         peak_equity=10000,day_start_equity=10000,week_start_equity=10000,
         last_bar_time_s=t,timeframe_seconds=3600)
    ex=PreTradeExaminer(cfg.execution)
    a=ex.inspect(p,snap,default_symbol_spec('EURUSD'),1.1,1.,.002,
       cost_multiplier=1.,market_context=context)
    b=ex.inspect(p,snap,default_symbol_spec('EURUSD'),1.1,1.,.002,
       cost_multiplier=3.,market_context=context)
    assert a.approved
    assert b.round_trip_cost_pips > a.round_trip_cost_pips
    assert not b.checks_passed['C22']
