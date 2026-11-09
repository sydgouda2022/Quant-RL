"""Regression tests for native MT5 BID-versus-midpoint quote semantics."""
from dataclasses import replace

import numpy as np
import pytest

from quant_rl.config.loader import ConfigLoader
from quant_rl.data.mt5_history import MT5History
from quant_rl.data.mt5_source import OHLCVBar
from tests._specs import default_symbol_spec
from quant_rl.execution.accounting import PortfolioAccountingSimulator
from quant_rl.contracts import TradeDirection
from quant_rl.market.state_builder import ObservationBuilder
from tests.helpers_current import CFG, obs_for_path, proposal_for


def test_bid_quote_uses_full_spread_to_buy_and_no_spread_offset_to_sell(tmp_path):
    cfg = ConfigLoader.load(CFG)
    spec = default_symbol_spec('EURUSD')
    px = [(1.1, 1.1001, 1.0999, 1.1)] * 5
    base = obs_for_path(px)
    base = replace(base, spread_price=np.full(5, .0002))  # 2-pip quote spread
    # Use a narrow enough spread for the examiner after adjusting the cost cap.
    safety = cfg.execution.model_copy(update={'max_round_trip_cost_pips': 5.0,
                                              'max_cost_to_stop_fraction': 0.9})
    # Trade decision at candle 0, next bars cannot touch wide stop/target.
    def proposal(idx, state):
        return proposal_for(base, idx, 2, .005)
    mid = PortfolioAccountingSimulator(safety).simulate(
        replace(base, bar_price_side='MID'), proposal, spec,
        start_index=0, end_index=4)
    bid = PortfolioAccountingSimulator(safety).simulate(
        replace(base, bar_price_side='BID'), proposal, spec,
        start_index=0, end_index=4)
    assert mid.open_positions_at_end and bid.open_positions_at_end
    assert bid.open_positions_at_end[0].entry_price - mid.open_positions_at_end[0].entry_price == pytest.approx(.0001)


def test_bid_side_m1_tp_first_uses_raw_bid_not_mid_minus_half_spread():
    t = 1704067200
    args = dict(direction=TradeDirection.BUY, stop_price=1.0995,
                target_price=1.10075, half_spread=.0001,
                bar_start_s=t, bar_end_s=t+120,
                m1_time_s=np.array([t, t+60], dtype=np.int64),
                m1_high=np.array([1.1008, 1.0999]),
                m1_low=np.array([1.1000, 1.0994]))
    resolve = PortfolioAccountingSimulator._resolve_dual_touch_m1
    assert resolve(**args, bar_price_side='BID') == 'TP'
    assert resolve(**args, bar_price_side='MID') == 'SL'


def test_mt5_origin_carries_bid_metadata_to_observation_builder(tmp_path):
    store = MT5History()
    spec = default_symbol_spec('EURUSD')
    start = 1704067200
    bars = [OHLCVBar(start+3600*i, 1.1,1.1002,1.0998,1.1,100,2.) for i in range(200)]
    frozen = store.freeze('EURUSD', 'H1', bars, spec)
    assert frozen.manifest.bar_price_side == 'BID'
    obs = ObservationBuilder(['H4'], 8).build_dataset(
        base_bars=frozen.bars, symbol='EURUSD', primary_timeframe='H4',
        spec=frozen.spec, bar_price_side=frozen.manifest.bar_price_side)
    assert obs.bar_price_side == 'BID'
    assert obs.spread_pips[0] == pytest.approx(2 * spec.point / spec.pip_size)


def test_unverified_quote_side_rejected_before_trading():
    cfg=ConfigLoader.load(CFG)
    obs=replace(obs_for_path([(1.1,1.1001,1.0999,1.1)]*6), bar_price_side='UNVERIFIED')
    with pytest.raises(ValueError, match='MID or BID'):
        PortfolioAccountingSimulator(cfg.execution).simulate(
            obs, lambda i,state:proposal_for(obs,i,0),default_symbol_spec('EURUSD'),
            start_index=0,end_index=5)


def test_resampled_spread_is_latest_observable_not_historical_median():
    from quant_rl.data.time_alignment import resample_ohlcv_utc
    from quant_rl.data.mt5_source import OHLCVBar
    time=1704067200
    bars=[OHLCVBar(time+3600*i,1.1,1.101,1.099,1.1,100,sp)
          for i,sp in enumerate([1.,1.,1.,20.])]
    h4=resample_ohlcv_utc(bars,'H4',source_timeframe='H1')
    assert len(h4)==1
    assert h4[0].spread_points==pytest.approx(20.)
    # Missing final spread is not silently replaced by an earlier cheap quote.
    bars[-1]=replace(bars[-1],spread_points=None)
    assert resample_ohlcv_utc(bars,'H4',source_timeframe='H1')[0].spread_points is None
