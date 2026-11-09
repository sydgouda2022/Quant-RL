"""Current policy safety test fixtures."""
from pathlib import Path
import numpy as np
from quant_rl.market.observations import MarketObservationDataset
from quant_rl.config.loader import ConfigLoader
from quant_rl.contracts import ActionProposal, TradeDirection
CFG = Path(__file__).resolve().parents[1] / "config.yaml"

def obs_for_path(prices: list[tuple[float,float,float,float]]) -> MarketObservationDataset:
    n = len(prices)
    assert n >= 4
    op, hi, lo, cl = map(lambda xs: np.asarray(xs,dtype=float), zip(*prices))
    return MarketObservationDataset(
        symbol="EURUSD", timeframe="H1", timeframe_seconds=3600, feature_names=("simple",),
        sequence_feature_names=("simple",), time_s=1704067200+3600*np.arange(n,dtype=np.int64),
        decision_time_s=1704070800+3600*np.arange(n,dtype=np.int64),
        open=op,high=hi,low=lo,close=cl,
        spread_price=np.full(n, 0.00002),volatility_price=np.full(n,0.002),
        feature_matrix=np.zeros((n,1)), sequence_tensor=np.zeros((n,4,1)),
    )


def proposal_for(obs,idx=1,action_index=2,risk=0.001):
    direction=TradeDirection.NO_TRADE if action_index==0 else (TradeDirection.SELL if action_index>=4 else TradeDirection.BUY)
    su,tu=(0.,0.) if direction==TradeDirection.NO_TRADE else (1.5,3.)
    return ActionProposal(symbol=obs.symbol,timeframe=obs.timeframe,bar_index=idx,
        decision_time_s=int(obs.decision_time_s[idx]),action_index=action_index,
        action_name=direction.value,direction=direction,stop_units=su,target_units=tu,
        stop_distance_price=su*.002,target_distance_price=tu*.002,
        requested_risk_fraction=0. if action_index==0 else risk)
