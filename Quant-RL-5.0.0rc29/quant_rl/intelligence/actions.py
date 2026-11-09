"""Hybrid trading policy action conversion; no fixed discrete geometry catalog."""
from __future__ import annotations
from dataclasses import dataclass
from quant_rl.contracts import ActionProposal, TradeDirection
from quant_rl.config.schema import GeometryModeConfig, PositionSizingConfig


@dataclass(frozen=True)
class HybridAction:
    direction_index: int  # 0=NO_TRADE, 1=BUY, 2=SELL
    stop_fraction: float
    target_fraction: float
    risk_fraction: float


def to_proposal(action: HybridAction, obs, index: int,
                geometry: GeometryModeConfig, sizing: PositionSizingConfig) -> ActionProposal:
    if action.direction_index not in (0, 1, 2):
        raise ValueError('Invalid hybrid direction')
    d = (TradeDirection.NO_TRADE, TradeDirection.BUY, TradeDirection.SELL)[action.direction_index]
    if d == TradeDirection.NO_TRADE:
        su=tu=risk=0.0
    else:
        su=geometry.min_stop_units + min(1.,max(0.,action.stop_fraction))*(geometry.max_stop_units-geometry.min_stop_units)
        tu=geometry.min_target_units + min(1.,max(0.,action.target_fraction))*(geometry.max_target_units-geometry.min_target_units)
        risk=(sizing.min_risk_per_trade_pct + min(1.,max(0.,action.risk_fraction))*
              (sizing.max_risk_per_trade_pct-sizing.min_risk_per_trade_pct)) / 100.0
    return ActionProposal(
        symbol=obs.symbol, timeframe=obs.timeframe, bar_index=index,
        decision_time_s=int(obs.decision_time_s[index]),
        action_index=action.direction_index, action_name=d.value,
        direction=d, stop_units=su, target_units=tu,
        stop_distance_price=su*float(obs.volatility_price[index]),
        target_distance_price=tu*float(obs.volatility_price[index]),
        requested_risk_fraction=risk, model_id='hybrid_ppo',
    )
