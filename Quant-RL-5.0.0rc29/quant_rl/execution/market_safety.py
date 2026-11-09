"""Causal rolling reference for market-adaptive pre-trade safeguards.

All reference distributions exclude the decision bar. The window choice is a
conservative trailing-data heuristic, NOT a learned or validated edge model.
No historical spread/ATR proxy is represented as real liquidity or fill evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from quant_rl.market.observations import MarketObservationDataset


@dataclass(frozen=True)
class MarketSafetyContext:
    observation_time_s: int
    sample_count: int
    median_spread_pips: float
    upper_spread_pips: float
    median_volatility_price: float
    upper_volatility_price: float
    spread_window_bars: int = 0
    volatility_window_bars: int = 0

    def __post_init__(self) -> None:
        values = (self.median_spread_pips, self.upper_spread_pips,
                  self.median_volatility_price, self.upper_volatility_price)
        if self.sample_count < 1 or not all(math.isfinite(v) for v in values):
            raise ValueError("Invalid market safety context")
        if self.median_spread_pips < 0 or self.upper_spread_pips < 0:
            raise ValueError("Historical spread cannot be negative")
        if self.median_volatility_price <= 0 or self.upper_volatility_price <= 0:
            raise ValueError("Historical volatility must be strictly positive")
        if self.spread_window_bars < 0 or self.volatility_window_bars < 0:
            raise ValueError("Invalid market safety window")



def _choose_trailing_window(
    spreads: np.ndarray, vols: np.ndarray, candidates: tuple[int, ...], minimum: int, regime_shift_ratio: float,
) -> int:
    """Pick the most recent stable reference; adapt shorter under regime shifts.

    Selection is deterministic and uses completed history only. In a calm regime
    the longest eligible window reduces noise. A strong recent median shift
    selects a shorter window to avoid anchoring to obsolete conditions.
    """
    eligible = sorted(set(w for w in candidates if minimum <= w <= len(spreads)))
    if not eligible:
        return len(spreads)
    chosen = eligible[-1]
    if len(eligible) == 1:
        return chosen
    for w in eligible[:-1]:
        if w < 2 * minimum:
            continue
        recent_spread = float(np.median(spreads[-w:]))
        older_spread = float(np.median(spreads[-chosen:-w])) if chosen > w else recent_spread
        recent_vol = float(np.median(vols[-w:]))
        older_vol = float(np.median(vols[-chosen:-w])) if chosen > w else recent_vol
        # A substantial observed regime change shortens history. This is a
        # heuristic only; thresholds must be validated by walk-forward studies.
        if (max(recent_spread, older_spread) > regime_shift_ratio * max(min(recent_spread, older_spread), 0.1)
                or max(recent_vol, older_vol) > regime_shift_ratio * max(min(recent_vol, older_vol), 1e-12)):
            return w
    return chosen


def market_context_asof(
    observations: MarketObservationDataset, bar_index: int, *,
    lookback_bars: int, min_history_bars: int,
    spread_quantile: float = 0.90,
    candidate_windows: tuple[int, ...] | None = None,
    regime_shift_ratio: float = 1.5,
    spread_values: np.ndarray | None = None,
) -> MarketSafetyContext | None:
    """Reference uses completed bars strictly BEFORE the decision bar."""
    if bar_index < 0 or bar_index >= observations.num_bars:
        raise IndexError("Invalid market safety observation index")
    if lookback_bars < min_history_bars or min_history_bars < 1:
        raise ValueError("Invalid lookback and minimum history")
    if not 0 < spread_quantile < 1:
        raise ValueError("Invalid spread quantile")
    if not math.isfinite(regime_shift_ratio) or regime_shift_ratio <= 1:
        raise ValueError("Invalid regime shift ratio")
    if bar_index < min_history_bars:
        return None
    start = max(0, bar_index - lookback_bars)
    spreads = np.asarray((observations.spread_pips if spread_values is None else spread_values)[start:bar_index], dtype=float)
    vols = np.asarray(observations.volatility_price[start:bar_index], dtype=float)
    if len(spreads) < min_history_bars:
        return None
    if (not np.all(np.isfinite(spreads)) or np.any(spreads < 0)
        or not np.all(np.isfinite(vols)) or np.any(vols <= 0)):
        raise ValueError("Invalid historical market statistics: fail closed")
    if np.any(np.diff(observations.decision_time_s[start:bar_index + 1]) <= 0):
        raise ValueError("Non-monotonic market history")
    if candidate_windows is None:
        spread_w = vol_w = len(spreads)
    else:
        # Independent rolling references avoid forcing spread and volatility
        # to react on the same timescale. The non-target indicator is held
        # constant to eliminate its influence on the lookback choice.
        spread_w = _choose_trailing_window(spreads, np.full_like(vols, float(np.median(vols))),
            candidate_windows, min_history_bars, regime_shift_ratio)
        vol_w = _choose_trailing_window(np.full_like(spreads, float(np.median(spreads))), vols,
            candidate_windows, min_history_bars, regime_shift_ratio)
    spreads, vols = spreads[-spread_w:], vols[-vol_w:]
    return MarketSafetyContext(
        observation_time_s=int(observations.decision_time_s[bar_index]),
        sample_count=min(len(spreads), len(vols)),
        median_spread_pips=float(np.median(spreads)),
        upper_spread_pips=float(np.quantile(spreads, spread_quantile)),
        median_volatility_price=float(np.median(vols)),
        upper_volatility_price=float(np.quantile(vols, 0.90)),
        spread_window_bars=len(spreads),
        volatility_window_bars=len(vols),
    )
