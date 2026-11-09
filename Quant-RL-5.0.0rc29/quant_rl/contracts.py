"""Shared immutable action and trade-proposal contracts."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum

class TradeDirection(StrEnum):
    NO_TRADE='NO_TRADE'
    BUY='BUY'
    SELL='SELL'

@dataclass(frozen=True)
class ActionProposal:
    symbol: str
    timeframe: str
    bar_index: int
    decision_time_s: int
    action_index: int
    action_name: str
    direction: TradeDirection
    stop_units: float
    target_units: float
    stop_distance_price: float
    target_distance_price: float
    requested_risk_fraction: float
    expected_net_r: float = 0.
    confidence: float = 0.
    model_id: str = 'hybrid_ppo'
