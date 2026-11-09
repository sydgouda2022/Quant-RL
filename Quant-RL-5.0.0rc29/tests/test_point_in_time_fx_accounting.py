"""USD base/account pairs must convert quote-currency P&L at the observed price."""
from dataclasses import replace
import numpy as np
import pytest
from quant_rl.config.loader import ConfigLoader
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from quant_rl.execution.accounting import PortfolioAccountingSimulator
from quant_rl.execution.sizing import compute_pip_value_per_lot
from quant_rl.contracts import ActionProposal, TradeDirection
from tests.helpers_current import CFG, obs_for_path


def usd_jpy_obs(prices):
    obs = obs_for_path(prices)
    return replace(obs, symbol='USDJPY',
                   spread_price=np.full(obs.num_bars, .002),
                   volatility_price=np.full(obs.num_bars, .50),
                   bar_price_side='BID')


def action(obs, index, kind='BUY'):
    direction = TradeDirection(kind)
    return ActionProposal(
        symbol='USDJPY', timeframe=obs.timeframe,
        bar_index=index, decision_time_s=int(obs.decision_time_s[index]),
        action_index=1 if direction==TradeDirection.BUY else 2,
        action_name=direction.value, direction=direction,
        stop_units=2., target_units=2.,
        stop_distance_price=1., target_distance_price=1.,
        requested_risk_fraction=.005, model_id='test')


def test_no_trade_penalty_uses_current_dollar_pip_value_not_stale_spec():
    prices=[(150.,150.05,149.95,150.)]*4 + [(120.,120.05,119.95,120.)]*4
    obs=usd_jpy_obs(prices)
    cfg=ConfigLoader.load(CFG)
    spec=default_symbol_spec('USDJPY',reference_price=150.,account_currency='USD')
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,lambda i,s: replace(action(obs,i),direction=TradeDirection.NO_TRADE, action_name='NO_TRADE',
             action_index=0,stop_units=0.,target_units=0.,stop_distance_price=0.,
             target_distance_price=0.,requested_risk_fraction=0.),
        spec,start_index=0,end_index=7,inactivity_penalty_lots=.1)
    assert len(sim.decision_events)==7
    # 1 pip is JPY 1000 per lot; JPY/USD changes from 1/150 to 1/120.
    assert sim.decision_events[0].spread_penalty_account_ccy == pytest.approx(.2*1000/150.001*.1)
    assert sim.decision_events[4].spread_penalty_account_ccy == pytest.approx(.2*1000/120.001*.1)
    assert sim.decision_events[4].spread_penalty_account_ccy>sim.decision_events[0].spread_penalty_account_ccy
    assert sim.ending_equity==cfg.execution.starting_balance
    assert sim.learning_reward_account_ccy==pytest.approx(-sim.penalty_total_account_ccy)


def test_completed_usdjpy_buy_uses_exit_fx_rate_for_realized_profit():
    prices=[(150.,150.03,149.97,150.), (150.2,151.40,150.1,151.2)] + [(151.2,151.3,151.1,151.2)]*4
    obs=usd_jpy_obs(prices)
    cfg=ConfigLoader.load(CFG)
    spec=default_symbol_spec('USDJPY',reference_price=150.,account_currency='USD')
    test_execution=cfg.execution.model_copy(update={'max_round_trip_cost_pips':4.0})
    sim=PortfolioAccountingSimulator(test_execution).simulate(
        obs,lambda i,s:action(obs,i) if i==0 else replace(action(obs,i),direction=TradeDirection.NO_TRADE,
             action_index=0,action_name='NO_TRADE',stop_units=0.,target_units=0.,
             stop_distance_price=0.,target_distance_price=0.,requested_risk_fraction=0.),
        spec,start_index=0,end_index=5)
    assert len(sim.executed_trades)==1
    t=sim.executed_trades[0]
    expected=(t.exit_price-t.entry_price)*spec.contract_size*t.volume_lots/t.exit_price
    assert t.net_pnl_account_ccy == pytest.approx(expected-t.commission_cost_account_ccy-t.financing_cost_account_ccy)
    assert t.exit_pip_value_per_lot_account_ccy == pytest.approx(spec.contract_size*spec.pip_size/t.exit_price)
    assert t.entry_pip_value_per_lot_account_ccy == pytest.approx(spec.contract_size*spec.pip_size/t.entry_price,rel=1e-3)
    assert t.exit_pip_value_per_lot_account_ccy != pytest.approx(t.entry_pip_value_per_lot_account_ccy,rel=1e-4)
    assert t.gross_pnl_account_ccy-t.total_cost_account_ccy == pytest.approx(t.net_pnl_account_ccy)
    assert sim.ending_balance == pytest.approx(sim.starting_balance+sum(x.net_pnl_account_ccy for x in sim.executed_trades))


def test_unresolved_usdjpy_position_marks_at_boundary_rate_without_closing():
    prices=[(150.,150.03,149.97,150.)]+[(150.4,150.43,150.37,150.4)]*6
    obs=usd_jpy_obs(prices)
    cfg=ConfigLoader.load(CFG)
    spec=default_symbol_spec('USDJPY',reference_price=150.,account_currency='USD')
    test_execution=cfg.execution.model_copy(update={'max_round_trip_cost_pips':4.0})
    sim=PortfolioAccountingSimulator(test_execution).simulate(
        obs,lambda i,s:action(obs,i),spec,start_index=0,end_index=6)
    assert len(sim.executed_trades)==0 and len(sim.open_positions_at_end)==1
    p=sim.open_positions_at_end[0]
    mark_bid=float(obs.close[-1]); fee=cfg.execution.commission_per_lot_round_trip*p.volume_lots
    slip=(cfg.execution.simulated_slippage_points_per_side*spec.point/spec.pip_size
          *compute_pip_value_per_lot(spec,mark_bid,'USD')*p.volume_lots)
    expected=(mark_bid-p.entry_price)*spec.contract_size*p.volume_lots/mark_bid-fee-slip
    assert sim.ending_equity-sim.ending_balance == pytest.approx(expected)
    assert sim.decision_events[0].terminal_mark_to_market_account_ccy == pytest.approx(expected)
    assert sim.ending_balance==sim.starting_balance


def test_missing_conversion_path_fails_closed_not_using_stale_broker_tick_value():
    spec=default_symbol_spec('EURGBP')
    with pytest.raises(ValueError,match='Missing point-in-time'):
        compute_pip_value_per_lot(spec,0.85,'USD')


def test_completed_usdjpy_sell_converts_profit_at_ask_exit_quote():
    prices=[(150.,150.03,149.97,150.),(149.6,149.9,148.60,148.8)]+[(148.8,148.9,148.7,148.8)]*4
    obs=usd_jpy_obs(prices)
    cfg=ConfigLoader.load(CFG)
    spec=default_symbol_spec('USDJPY',reference_price=150.,account_currency='USD')
    test_execution=cfg.execution.model_copy(update={'max_round_trip_cost_pips':4.0})
    def decide(i,state):
        if i==0:return action(obs,i,'SELL')
        p=action(obs,i)
        return replace(p,action_index=0,direction=TradeDirection.NO_TRADE,action_name='NO_TRADE',
                       stop_units=0.,target_units=0.,stop_distance_price=0.,target_distance_price=0.,
                       requested_risk_fraction=0.)
    sim=PortfolioAccountingSimulator(test_execution).simulate(
        obs,decide,spec,start_index=0,end_index=5)
    assert len(sim.executed_trades)==1
    t=sim.executed_trades[0]
    assert t.direction==TradeDirection.SELL and t.exit_reason=='TP'
    expected=(t.entry_price-t.exit_price)*spec.contract_size*t.volume_lots/t.exit_price
    assert t.net_pnl_account_ccy==pytest.approx(expected-t.commission_cost_account_ccy-t.financing_cost_account_ccy)
    assert t.gross_pnl_account_ccy-t.total_cost_account_ccy==pytest.approx(t.net_pnl_account_ccy)
