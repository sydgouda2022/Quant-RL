"""Structured market and account observation containers for tabular and temporal models."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from quant_rl.data.mt5_source import pip_size_for_symbol


@dataclass(frozen=True)
class AccountStateObservation:
    """
    Point-in-time portfolio/account state observed by the single Market Intelligence:
    - `drawdown_fraction`: current peak-to-valley drawdown in [0, 1]
    - `daily_dd_utilization`: fraction of daily drawdown limit consumed in [0, 1]
    - `open_risk_fraction`: total open monetary risk divided by account equity
    - `open_positions_ratio`: open positions divided by max allowed concurrent positions
    - `loss_streak_ratio`: consecutive losses divided by circuit-breaker limit
    """

    drawdown_fraction: float = 0.0
    daily_dd_utilization: float = 0.0
    open_risk_fraction: float = 0.0
    open_positions_ratio: float = 0.0
    loss_streak_ratio: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.drawdown_fraction,
                self.daily_dd_utilization,
                self.open_risk_fraction,
                self.open_positions_ratio,
                self.loss_streak_ratio,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class MarketObservation:
    """
    Single point-in-time market and account observation at decision index `bar_index`.
    `open_time_s` is the bar open timestamp; `decision_time_s` is the bar close timestamp.
    """

    symbol: str
    timeframe: str
    bar_index: int
    open_time_s: int
    decision_time_s: int
    close_price: float
    spread_price: float
    volatility_price: float
    features: np.ndarray          # Shape: (F,)
    sequence_window: np.ndarray   # Shape: (L, F_seq)
    account_state: AccountStateObservation = AccountStateObservation()


@dataclass(frozen=True)
class MarketObservationDataset:
    """Aligned dataset of point-in-time observations across an entire symbol timeline."""

    symbol: str
    timeframe: str
    timeframe_seconds: int
    feature_names: tuple[str, ...]
    sequence_feature_names: tuple[str, ...]
    time_s: np.ndarray            # Shape: (N,) bar open time in UTC seconds
    decision_time_s: np.ndarray   # Shape: (N,) bar close time (`time_s + timeframe_seconds`)
    open: np.ndarray              # Shape: (N,)
    high: np.ndarray              # Shape: (N,)
    low: np.ndarray               # Shape: (N,)
    close: np.ndarray             # Shape: (N,)
    spread_price: np.ndarray      # Shape: (N,)
    volatility_price: np.ndarray  # Shape: (N,)
    feature_matrix: np.ndarray    # Shape: (N, F)
    sequence_tensor: np.ndarray   # Shape: (N, L, F_seq)
    bar_price_side: str = "MID"  # Test fixtures may use MID; acquired MT5 market bars always BID

    @property
    def num_bars(self) -> int:
        return int(len(self.time_s))

    @property
    def spread_pips(self) -> np.ndarray:
        pip_sz = pip_size_for_symbol(self.symbol)
        return self.spread_price / max(pip_sz, 1e-9)

    def at(
        self,
        index: int,
        account_state: AccountStateObservation | None = None,
    ) -> MarketObservation:
        return MarketObservation(
            symbol=self.symbol,
            timeframe=self.timeframe,
            bar_index=index,
            open_time_s=int(self.time_s[index]),
            decision_time_s=int(self.decision_time_s[index]),
            close_price=float(self.close[index]),
            spread_price=float(self.spread_price[index]),
            volatility_price=float(self.volatility_price[index]),
            features=self.feature_matrix[index],
            sequence_window=self.sequence_tensor[index],
            account_state=account_state or AccountStateObservation(),
        )
