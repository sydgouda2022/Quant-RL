"""Regression guards for remaining market-sensitive checks (C8-10/18-21/26)."""
from dataclasses import replace
import numpy as np
import pytest

from quant_rl.config.schema import ExecutionConfig
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from quant_rl.execution.examiner import OpenPositionSnapshot, PortfolioStateSnapshot, PreTradeExaminer
from quant_rl.execution.market_safety import MarketSafetyContext, market_context_asof
from tests.helpers_current import obs_for_path, proposal_for


def fixtures(*, vol=.002, spread=1.0, other=(), max_positions=5, **changes):
    obs = obs_for_path([(1.1, 1.103, 1.098, 1.1)] * 75)
    idx = 50
    t = int(obs.decision_time_s[idx])
    p = proposal_for(obs, idx=idx, action_index=2, risk=.005)
    cfg = ExecutionConfig(max_open_positions=max_positions, **changes)
    ctx = MarketSafetyContext(t, 60, 1.0, 1.5, .002, .004, 60)
    port = PortfolioStateSnapshot(t, 10000., 10000., 10000., 10000., 10000.,
                                  open_positions=tuple(other), last_bar_time_s=t,
                                  timeframe_seconds=3600)
    def run(*, proposal=p, portfolio=port, config=cfg, context=ctx, volatility=vol, spread_now=spread):
        return PreTradeExaminer(config).inspect(proposal, portfolio, default_symbol_spec('EURUSD'),
            mid_price=1.1, spread_pips=spread_now, volatility_price=volatility,
            market_context=context)
    return run, p, port, cfg, ctx


def position(i, symbol, risk=50.0, margin=50.0, direction='BUY'):
    return OpenPositionSnapshot(i, symbol, direction, .1, 1.1, 1.08, 1.15,
                                1700000000, risk, margin)


def test_market_candidates_validated():
    with pytest.raises(ValueError, match='market_window_candidates'):
        ExecutionConfig(market_window_candidates=(12, 64))
    with pytest.raises(ValueError, match='market_window_candidates'):
        ExecutionConfig(market_window_candidates=(32, 32))
    assert ExecutionConfig().market_window_candidates == (32, 64, 128)


def test_rolling_window_selects_shorter_when_regime_shifts():
    obs = obs_for_path([(1.1,1.102,1.099,1.1)]*200)
    spreads = np.array([.0001] * 150 + [.0003] * 50)
    obs = replace(obs, spread_price=spreads)
    ctx = market_context_asof(obs, 195, lookback_bars=128,
             min_history_bars=20, candidate_windows=(32,64,128))
    assert ctx.spread_window_bars < 128
    # Future/decision-time spreads cannot change selected lookback or statistics.
    future = replace(obs, spread_price=np.concatenate((spreads[:195], np.full(5, .5))))
    assert market_context_asof(future,195,lookback_bars=128,
             min_history_bars=20,candidate_windows=(32,64,128)) == ctx


def test_c8_adapts_capacity_but_never_exceeds_hard_count():
    positions = tuple(position(i,f'{s}USD') for i,s in enumerate(('AUD','NZD','GBP'), 1))
    run, _, _, _, _ = fixtures(other=positions)
    normal = run()
    stressed = run(volatility=.003)
    assert normal.checks_passed['C8']
    assert not stressed.checks_passed['C8']
    assert dict(stressed.adaptive_limits)['C8_max_positions'] <= 5


def test_c9_dynamic_budget_can_only_tighten():
    run, _, _, _, _ = fixtures(other=(position(1,'AUDUSD',risk=240),))
    normal = run()
    stressed = run(volatility=.003)
    assert normal.checks_passed['C9']
    assert not stressed.checks_passed['C9']
    assert dict(stressed.adaptive_limits)['C9_open_risk_budget'] < dict(normal.adaptive_limits)['C9_open_risk_budget']


def test_c10_signed_currency_risk_tightens_without_overriding_hard_count():
    run, _, _, _, _ = fixtures(other=(position(1,'GBPUSD',risk=175),))
    normal = run()
    stressed = run(volatility=.003)
    assert normal.checks_passed['C10']
    assert not stressed.checks_passed['C10']
    assert dict(stressed.adaptive_limits)['C10_max_net_currency_risk'] > 0


def test_c18_margin_buffer_tightens():
    run, _, _, _, _ = fixtures(other=(position(1,'AUDUSD',margin=5900),))
    assert run().checks_passed['C18']
    assert not run(volatility=.003).checks_passed['C18']


def test_c19_leverage_cap_tightens():
    run, _, _, _, _ = fixtures(other=(position(1,'AUDUSD',margin=20),), maximum_leverage=3.0)
    assert run().checks_passed['C19']
    assert (dict(run(volatility=.003).adaptive_limits)['C19_max_leverage']
            < dict(run().adaptive_limits)['C19_max_leverage'])


def test_c20_c21_geometry_has_market_floor_and_broker_floor():
    run, p, _, _, ctx = fixtures()
    ctx = replace(ctx,median_volatility_price=.010,upper_volatility_price=.02)
    tight_stop = replace(p,stop_distance_price=.0005)
    tight_target = replace(p,target_distance_price=.0005)
    assert not run(proposal=tight_stop,context=ctx).checks_passed['C20']
    assert not run(proposal=tight_target,context=ctx).checks_passed['C21']


def test_c26_market_stress_increases_mechanical_feasibility_floor():
    run, p, _, _, _ = fixtures()
    p = replace(p, target_distance_price=.0011, stop_distance_price=.002)
    normal=run(proposal=p)
    stressed=run(proposal=p,volatility=.004)
    assert normal.checks_passed['C26']
    assert not stressed.checks_passed['C26']
    assert dict(stressed.adaptive_limits)['C26_min_net_ratio'] > dict(normal.adaptive_limits)['C26_min_net_ratio']


def test_market_adaptation_never_overrides_kill_switch_or_absolute_risk():
    run,p,port,_,_ = fixtures()
    blocked=run(portfolio=replace(port,kill_switch_engaged=True))
    assert not blocked.approved and blocked.veto_code == 'C1'
    risky=run(proposal=replace(p,requested_risk_fraction=.03))
    assert not risky.checks_passed['C15']


def test_spread_and_volatility_lookbacks_can_differ():
    obs=obs_for_path([(1.1,1.102,1.099,1.1)]*200)
    obs=replace(obs,spread_price=np.r_[np.full(150,.0001),np.full(50,.0003)],
                volatility_price=np.full(200,.002))
    ctx=market_context_asof(obs,195,lookback_bars=128,min_history_bars=20,
                             candidate_windows=(32,64,128))
    assert ctx.spread_window_bars < 128
    assert ctx.volatility_window_bars == 128
    assert ctx.sample_count == min(ctx.spread_window_bars,ctx.volatility_window_bars)


