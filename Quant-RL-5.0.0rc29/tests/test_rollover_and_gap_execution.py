"""Broker-financing calendar and opening-gap execution regression tests."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import numpy as np
import pytest

from quant_rl.config.schema import ExecutionConfig
from quant_rl.config.loader import ConfigLoader
from quant_rl.data.mt5_source import SymbolSpec
from tests._specs import default_symbol_spec
from quant_rl.execution.financing import financing_cost
from quant_rl.execution.accounting import PortfolioAccountingSimulator
from quant_rl.contracts import TradeDirection
from quant_rl.reporting.economic_report import write_economic_report
from tests.helpers_current import obs_for_path, proposal_for, CFG


def utc(y, mo, d, hour=0, minute=0):
    return int(datetime(y, mo, d, hour, minute, tzinfo=timezone.utc).timestamp())


def test_rollover_is_calendar_based_not_elapsed_24_hours():
    cfg=ExecutionConfig()
    spec=default_symbol_spec('EURUSD')
    # A holding of only two hours crosses Tuesday rollover and owes one unit.
    cost, units, provenance=financing_cost(utc(2024,1,2,20),utc(2024,1,2,22),1.,TradeDirection.BUY,spec,cfg)
    assert cost==pytest.approx(4.5) and units==1 and provenance=='MODELED_CONFIG'
    # Nearly 24 hours with no rollover crossed incurs nothing.
    assert financing_cost(utc(2024,1,2,21,1),utc(2024,1,3,20,59),1.,TradeDirection.BUY,spec,cfg)[0]==0
    # Wednesday triple rollover plus Thursday and Friday singles.
    c,u,_=financing_cost(utc(2024,1,3,20),utc(2024,1,5,22),2.,TradeDirection.BUY,spec,cfg)
    assert (u,c)==pytest.approx((5,45.))
    # Weekend has no additional rollover, Monday does.
    assert financing_cost(utc(2024,1,5,22),utc(2024,1,7,22),1.,TradeDirection.BUY,spec,cfg)[1]==0
    assert financing_cost(utc(2024,1,5,22),utc(2024,1,8,22),1.,TradeDirection.BUY,spec,cfg)[1]==1


def test_side_specific_broker_normalized_cash_and_conservative_stress():
    spec=replace(default_symbol_spec('EURUSD'),
                 financing_long_account_ccy_per_lot_rollover=-8.,
                 financing_short_account_ccy_per_lot_rollover=2.)
    cfg=ExecutionConfig(financing_long_cash_per_lot_rollover=-4.5,
                        financing_short_cash_per_lot_rollover=-4.5)
    start,end=utc(2024,1,2,20),utc(2024,1,2,22)
    debit, units, source=financing_cost(start,end,.5,TradeDirection.BUY,spec,cfg,cost_multiplier=1.5)
    assert debit==pytest.approx(6.) and units==1 and source=='SYMBOL_NORMALIZED_UNVERIFIED'
    assert financing_cost(start,end,.5,TradeDirection.SELL,spec,cfg,cost_multiplier=1)[0]==pytest.approx(-1.)
    assert financing_cost(start,end,.5,TradeDirection.SELL,spec,cfg,cost_multiplier=2)[0]==pytest.approx(-.5)
    # MT5 raw swap fields may be points, interest or a different currency.
    # Their numerical values must not be applied as account-currency cash.
    raw=replace(default_symbol_spec('EURUSD'),swap_mode=1,
                swap_long_per_lot_per_day=-9999.,swap_short_per_lot_per_day=9999.)
    assert financing_cost(start,end,1.,TradeDirection.BUY,raw,cfg)[0]==pytest.approx(4.5)
    with pytest.raises(ValueError,match='both'):
        financing_cost(start,end,1.,TradeDirection.BUY,
            replace(spec,financing_short_account_ccy_per_lot_rollover=None),cfg)


def test_open_position_mark_and_trade_close_use_same_rollover_accounting(tmp_path):
    obs=obs_for_path([(1.1,1.1001,1.0999,1.1)]*8)
    times=np.array([utc(2024,1,2,19)+3600*i for i in range(8)],dtype=np.int64)
    obs=replace(obs,time_s=times,decision_time_s=times+3600)
    cfg=ConfigLoader.load(CFG)
    sim=PortfolioAccountingSimulator(cfg.execution).simulate(
        obs,lambda i,s: proposal_for(obs,i,2,.005),default_symbol_spec('EURUSD'),
        start_index=0,end_index=6)
    assert len(sim.open_positions_at_end)==1
    assert sim.open_positions_at_end[0].unrealized_pnl_account_ccy < 0
    # The terminal valuation and learner's reward are identical.
    assert sim.learning_reward_account_ccy==pytest.approx(sim.ending_equity-sim.ending_balance)
    # Force TP on the next bar, after rollover, for a realized ledger check.
    op,hi,lo,cl=obs.open.copy(),obs.high.copy(),obs.low.copy(),obs.close.copy()
    op[6]=1.1;hi[6]=1.12;lo[6]=1.0999;cl[6]=1.11
    closed=replace(obs,open=op,high=hi,low=lo,close=cl)
    result=PortfolioAccountingSimulator(cfg.execution).simulate(
        closed,lambda i,s:proposal_for(closed,i,2,.005),default_symbol_spec('EURUSD'),
        start_index=0,end_index=7)
    trade=result.executed_trades[0]
    assert trade.financing_rollover_units==1
    assert trade.financing_source=='MODELED_CONFIG'
    assert trade.financing_cost_account_ccy==pytest.approx(trade.volume_lots*4.5)
    assert trade.gross_pnl_account_ccy-trade.total_cost_account_ccy==pytest.approx(trade.net_pnl_account_ccy)
    report=write_economic_report(tmp_path,'EURUSD',result,origin='SYNTHETIC_SIMULATOR',
                settings={},period={'test_start_utc':'2024-01-02','test_end_utc':'2024-01-03'},
                qualified=False,reasons=['SYNTHETIC'],trial_report=[])
    summary=json.loads((report/'summary.json').read_text())
    assert summary['financing_sources']==['MODELED_CONFIG']
    assert summary['financing_rollover_units_total']==1
    assert 'MODELED_CONFIG' in (report/'report.md').read_text()


def test_favorable_gap_tp_precedes_later_same_bar_stop_without_m1():
    base=[(1.1,1.1001,1.0999,1.1)]*6
    obs=obs_for_path([base[0],(1.11,1.112,1.08,1.1),*base[2:]])
    sim=PortfolioAccountingSimulator(ConfigLoader.load(CFG).execution).simulate(
       obs,lambda i,s: proposal_for(obs,i,2,.005),default_symbol_spec('EURUSD'),
       start_index=0,end_index=5)
    assert len(sim.executed_trades)==1
    assert sim.executed_trades[0].exit_reason=='TP'
    assert sim.executed_trades[0].exit_bar_index==1
    # Favorable SELL opening gap is symmetrical.
    obs=obs_for_path([base[0],(1.089,1.12,1.08,1.10),*base[2:]])
    sim=PortfolioAccountingSimulator(ConfigLoader.load(CFG).execution).simulate(
       obs,lambda i,s: proposal_for(obs,i,4,.005),default_symbol_spec('EURUSD'),
       start_index=0,end_index=5)
    assert sim.executed_trades[0].exit_reason=='TP'
    # An adverse opening gap still executes the stop at its gapped quote.
    obs=obs_for_path([base[0],(1.09,1.12,1.08,1.10),*base[2:]])
    sim=PortfolioAccountingSimulator(ConfigLoader.load(CFG).execution).simulate(
       obs,lambda i,s: proposal_for(obs,i,2,.005),default_symbol_spec('EURUSD'),
       start_index=0,end_index=5)
    assert sim.executed_trades[0].exit_reason=='SL_GAP'
    assert sim.executed_trades[0].exit_price < sim.executed_trades[0].stop_price


def test_financing_input_validation_and_boundary_utc():
    cfg=ExecutionConfig(financing_rollover_utc_hour=0,financing_triple_weekday_utc=4)
    spec=default_symbol_spec('EURUSD')
    ts=utc(2024,1,5,0)
    assert financing_cost(ts,ts,1.,TradeDirection.BUY,spec,cfg)[1]==0
    assert financing_cost(ts-1,ts,1.,TradeDirection.BUY,spec,cfg)[1]==3
    with pytest.raises(ValueError):financing_cost(ts,ts-1,1.,TradeDirection.BUY,spec,cfg)
    with pytest.raises(ValueError):financing_cost(ts-1,ts,0.,TradeDirection.BUY,spec,cfg)
