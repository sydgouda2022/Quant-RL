"""Configurable volatility estimators and raw-price / DST-aware session / optional technical features."""

from __future__ import annotations

from datetime import UTC, datetime
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quant_rl.data.mt5_source import OHLCVBar, SymbolSpec

_LONDON_TZ = ZoneInfo("Europe/London")
_NEW_YORK_TZ = ZoneInfo("America/New_York")


def compute_volatility_series(
    open_px: np.ndarray,
    high_px: np.ndarray,
    low_px: np.ndarray,
    close_px: np.ndarray,
    estimator: str = "atr",
    period: int = 14,
) -> np.ndarray:
    """
    Compute strictly causal volatility in price units at each bar close.
    Supports:
    - 'atr': Wilder's Average True Range
    - 'realized_vol': Strictly causal expanding/rolling standard deviation of close-to-close changes
    - 'parkinson_vol': High-low range estimator in price units
    """
    n = len(close_px)
    if n == 0:
        return np.array([], dtype=np.float64)
    p = max(2, int(period))
    est = str(estimator or "atr").strip().lower()

    prev_close = np.roll(close_px, 1)
    prev_close[0] = open_px[0]
    tr = np.maximum(
        high_px - low_px,
        np.maximum(np.abs(high_px - prev_close), np.abs(low_px - prev_close)),
    )

    if est == "atr":
        vol = np.full(n, np.nan, dtype=np.float64)
        if n >= p:
            vol[p - 1] = float(np.mean(tr[:p]))
            alpha = 1.0 / float(p)
            for i in range(p, n):
                vol[i] = vol[i - 1] * (1.0 - alpha) + tr[i] * alpha
        for i in range(min(p - 1, n)):
            vol[i] = float(np.mean(tr[: i + 1]))
        return np.maximum(vol, 1e-8)

    if est == "realized_vol":
        diff = np.abs(np.diff(close_px, prepend=open_px[0]))
        vol = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start_i = max(0, i - p + 1)
            window = diff[start_i : i + 1]
            if len(window) >= 2:
                std_val = float(np.std(window, ddof=1)) * math.sqrt(2.0)
                vol[i] = std_val if std_val > 1e-9 else float(np.mean(tr[start_i : i + 1]))
            else:
                vol[i] = float(tr[i])
        return np.maximum(vol, 1e-8)

    if est == "parkinson_vol":
        hl_log = np.log(np.maximum(high_px, 1e-8) / np.maximum(low_px, 1e-8))
        factor = 1.0 / (4.0 * math.log(2.0))
        park_pct = np.sqrt(
            pd.Series(factor * (hl_log**2)).rolling(p, min_periods=1).mean().to_numpy(dtype=np.float64)
        )
        return np.maximum(park_pct * close_px, 1e-8)

    raise ValueError(f"Unsupported volatility estimator: {estimator!r}")


def _compute_dst_session_flags(decision_times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute DST-aware London (08:00-17:00 Europe/London) and New York (08:00-17:00 America/New_York)
    session indicators.
    """
    n = len(decision_times_s)
    london = np.zeros(n, dtype=np.float64)
    ny = np.zeros(n, dtype=np.float64)
    overlap = np.zeros(n, dtype=np.float64)
    for i, ts in enumerate(decision_times_s):
        dt_utc = datetime.fromtimestamp(int(ts), tz=UTC)
        lon_h = dt_utc.astimezone(_LONDON_TZ).hour
        ny_h = dt_utc.astimezone(_NEW_YORK_TZ).hour
        is_lon = 8 <= lon_h < 17
        is_ny = 8 <= ny_h < 17
        london[i] = 1.0 if is_lon else 0.0
        ny[i] = 1.0 if is_ny else 0.0
        overlap[i] = 1.0 if (is_lon and is_ny) else 0.0
    return london, ny, overlap


def compute_feature_frame(
    bars: list[OHLCVBar],
    spec: SymbolSpec,
    volatility_estimator: str = "atr",
    volatility_period: int = 14,
    fallback_spread_points: float = 10.0,
    include_technical_indicators: bool = True,
    timeframe_seconds: int = 14400,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build causal point-in-time feature DataFrame and volatility price series.
    Every row `i` uses ONLY `bars[0..i]`.
    Raises `ValueError` (fail-closed) if input prices contain NaN/Inf/non-positive values.
    """
    time_s = np.array([b.time_s for b in bars], dtype=np.int64)
    open_px = np.array([b.open for b in bars], dtype=np.float64)
    high_px = np.array([b.high for b in bars], dtype=np.float64)
    low_px = np.array([b.low for b in bars], dtype=np.float64)
    close_px = np.array([b.close for b in bars], dtype=np.float64)
    tick_vol = np.array([b.tick_volume for b in bars], dtype=np.float64)

    for arr_name, arr in (("open", open_px), ("high", high_px), ("low", low_px), ("close", close_px)):
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise ValueError(f"Non-finite or non-positive {arr_name} prices detected in input bars")

    spread_pts = np.array(
        [
            float(b.spread_points) if (b.spread_points is not None and math.isfinite(b.spread_points) and b.spread_points > 0) else fallback_spread_points
            for b in bars
        ],
        dtype=np.float64,
    )

    vol_price = compute_volatility_series(
        open_px, high_px, low_px, close_px, estimator=volatility_estimator, period=volatility_period
    )
    safe_vol = np.maximum(vol_price, spec.point * 5.0)
    close_s = pd.Series(close_px)
    high_s = pd.Series(high_px)
    low_s = pd.Series(low_px)
    vol_s = pd.Series(safe_vol)

    # 1. Volatility-normalized returns across multiple horizons
    ret_1 = (close_s - close_s.shift(1)).fillna(0.0) / vol_s
    ret_2 = (close_s - close_s.shift(2)).fillna(0.0) / (vol_s * math.sqrt(2.0))
    ret_4 = (close_s - close_s.shift(4)).fillna(0.0) / (vol_s * 2.0)
    ret_8 = (close_s - close_s.shift(8)).fillna(0.0) / (vol_s * math.sqrt(8.0))
    ret_16 = (close_s - close_s.shift(16)).fillna(0.0) / (vol_s * 4.0)

    # 2. Bar structure & range features
    bar_range = np.maximum(high_px - low_px, spec.point)
    body_ratio = (close_px - open_px) / bar_range
    upper_wick_ratio = (high_px - np.maximum(open_px, close_px)) / bar_range
    lower_wick_ratio = (np.minimum(open_px, close_px) - low_px) / bar_range
    range_to_vol = bar_range / safe_vol

    # 3. Volatility regime (compression / expansion)
    vol_ma_20 = vol_s.rolling(20, min_periods=1).mean()
    vol_regime_ratio = (vol_s / np.maximum(vol_ma_20, 1e-8)) - 1.0

    # 4. Structure: position within rolling 20-bar channel
    roll_high_20 = high_s.rolling(20, min_periods=1).max()
    roll_low_20 = low_s.rolling(20, min_periods=1).min()
    channel_span = np.maximum(roll_high_20 - roll_low_20, vol_s)
    channel_pos_20 = ((close_s - roll_low_20) / channel_span) * 2.0 - 1.0

    # 5. Trading session & cyclical time features at decision close timestamp (`time_s + timeframe_seconds`)
    decision_times_s = time_s + int(timeframe_seconds)
    hours = ((decision_times_s // 3600) % 24).astype(np.float64)
    weekdays = (((decision_times_s // 86400) + 4) % 7).astype(np.float64)
    hour_sin = np.sin(2.0 * math.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * math.pi * hours / 24.0)
    dow_sin = np.sin(2.0 * math.pi * weekdays / 7.0)
    dow_cos = np.cos(2.0 * math.pi * weekdays / 7.0)
    session_london, session_ny, session_overlap = _compute_dst_session_flags(decision_times_s)

    # 6. Execution & liquidity conditions
    spread_price = spread_pts * spec.point
    spread_to_vol = spread_price / safe_vol
    vol_ma = pd.Series(tick_vol).rolling(20, min_periods=1).mean().to_numpy(dtype=np.float64)
    rel_tick_vol = np.clip((tick_vol / np.maximum(vol_ma, 1.0)) - 1.0, -2.0, 5.0)

    cols: dict[str, np.ndarray] = {
        "ret_1": ret_1.to_numpy(dtype=np.float64),
        "ret_2": ret_2.to_numpy(dtype=np.float64),
        "ret_4": ret_4.to_numpy(dtype=np.float64),
        "ret_8": ret_8.to_numpy(dtype=np.float64),
        "ret_16": ret_16.to_numpy(dtype=np.float64),
        "body_ratio": body_ratio,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "range_to_vol": range_to_vol,
        "vol_regime_ratio": vol_regime_ratio.to_numpy(dtype=np.float64),
        "channel_pos_20": channel_pos_20.to_numpy(dtype=np.float64),
    }

    # 7. Optional technical indicators
    if include_technical_indicators:
        delta = close_s.diff().fillna(0.0)
        gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
        rs = gain / np.maximum(loss, 1e-8)
        rsi_centered = (100.0 - (100.0 / (1.0 + rs)) - 50.0) / 50.0

        ema_fast = close_s.ewm(span=12, adjust=False).mean()
        ema_slow = close_s.ewm(span=26, adjust=False).mean()
        macd_norm = (ema_fast - ema_slow) / vol_s

        sma_20 = close_s.rolling(20, min_periods=1).mean()
        std_20 = close_s.rolling(20, min_periods=2).std().fillna(vol_s)
        bb_zscore = (close_s - sma_20) / np.maximum(std_20, 1e-8)

        cols["rsi_centered"] = rsi_centered.to_numpy(dtype=np.float64)
        cols["macd_norm"] = macd_norm.to_numpy(dtype=np.float64)
        cols["bb_zscore"] = bb_zscore.to_numpy(dtype=np.float64)

    cols.update(
        {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "session_london": session_london,
            "session_ny": session_ny,
            "session_overlap": session_overlap,
            "spread_to_vol": spread_to_vol,
            "rel_tick_vol": rel_tick_vol,
        }
    )

    df = pd.DataFrame(cols)
    df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df, safe_vol
