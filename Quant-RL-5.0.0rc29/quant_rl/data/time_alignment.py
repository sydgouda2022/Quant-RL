"""UTC-aligned timeframe construction and strict point-in-time multi-timeframe alignment."""

from __future__ import annotations

import numpy as np

from quant_rl.data.mt5_source import OHLCVBar

TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D1": 86400,
}


def timeframe_to_seconds(timeframe: str) -> int:
    tf = str(timeframe or "").strip().upper()
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe {timeframe!r}; valid={list(TIMEFRAME_SECONDS)}")
    return TIMEFRAME_SECONDS[tf]


def resample_ohlcv_utc(bars: list[OHLCVBar], target_timeframe: str, *, source_timeframe: str) -> list[OHLCVBar]:
    """
    Construct higher-timeframe candles (including H2, H6, H8, H12, D1) aligned to UTC midnight boundaries.
    Every bucket is keyed by `bucket_open = (bar.time_s // period_s) * period_s`.
    """
    if not bars:
        return []
    period_s = timeframe_to_seconds(target_timeframe)
    ordered = sorted(bars, key=lambda b: b.time_s)
    gaps = np.diff([b.time_s for b in ordered])
    if np.any(gaps <= 0):
        raise ValueError("Source bars must have unique strictly increasing timestamps")
    # Caller must identify the broker-requested timeframe. Sparse H1 bars must
    # never be reinterpreted as regular H2 bars from their observed gaps.
    source_s = timeframe_to_seconds(source_timeframe)
    if any(b.time_s % source_s for b in ordered):
        raise ValueError("Source bar timestamp is not aligned to its declared timeframe")
    if source_s not in TIMEFRAME_SECONDS.values() or period_s < source_s or period_s % source_s:
        raise ValueError(f"Cannot resample source interval {source_s}s to {period_s}s")
    buckets: dict[int, list[OHLCVBar]] = {}
    for b in ordered:
        bucket_open = (b.time_s // period_s) * period_s
        buckets.setdefault(bucket_open, []).append(b)

    resampled: list[OHLCVBar] = []
    for bucket_open in sorted(buckets):
        group = buckets[bucket_open]
        expected = np.arange(bucket_open, bucket_open + period_s, source_s, dtype=np.int64)
        actual = np.array([b.time_s for b in group], dtype=np.int64)
        # Incomplete daily/weekend candles and missing intrabucket observations are not closed bars.
        if not np.array_equal(actual, expected):
            continue
        open_px = group[0].open
        high_px = max(x.high for x in group)
        low_px = min(x.low for x in group)
        close_px = group[-1].close
        vol = sum(x.tick_volume for x in group)
        # Trading decisions occur at the CLOSE of the resampled bar. A median
        # spread over the entire candle can conceal a rollover/spread shock
        # near entry. Use the most recently observable source-bar spread.
        last_spread = group[-1].spread_points
        decision_spread = (float(last_spread) if last_spread is not None
                           and np.isfinite(last_spread) and last_spread > 0 else None)
        resampled.append(
            OHLCVBar(
                time_s=bucket_open,
                open=open_px,
                high=high_px,
                low=low_px,
                close=close_px,
                tick_volume=vol,
                spread_points=decision_spread,
            )
        )
    return resampled


def align_completed_bars_asof(
    decision_close_times_s: np.ndarray,
    higher_tf_bars: list[OHLCVBar],
    higher_tf: str,
) -> np.ndarray:
    """
    For each primary decision close timestamp `t_close = primary_open + primary_period`,
    return the index into `higher_tf_bars` of the most recent COMPLETED higher-timeframe bar
    satisfying `htf_bar.time_s + higher_period <= t_close`.
    Returns `-1` where no completed higher-timeframe bar exists yet.
    Strict point-in-time guarantee: forming higher-timeframe bars are NEVER visible.
    """
    if not higher_tf_bars:
        return np.full(len(decision_close_times_s), -1, dtype=np.int64)
    higher_period_s = timeframe_to_seconds(higher_tf)
    htf_close_times = np.array([b.time_s + higher_period_s for b in higher_tf_bars], dtype=np.int64)
    if np.any(np.diff(htf_close_times) <= 0):
        raise ValueError(f"Higher-timeframe ({higher_tf}) bar timestamps must be strictly increasing")
    # searchsorted(..., side='right') - 1 gives the largest index j where htf_close_times[j] <= t_close
    indices = np.searchsorted(htf_close_times, decision_close_times_s, side="right") - 1
    return indices.astype(np.int64)
