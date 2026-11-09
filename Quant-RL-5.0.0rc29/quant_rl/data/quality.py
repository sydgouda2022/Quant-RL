"""Data quality verification and honest historical coverage manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import math
from typing import Any

import numpy as np

from quant_rl.data.mt5_source import OHLCVBar
from quant_rl.data.time_alignment import timeframe_to_seconds


@dataclass(frozen=True)
class DataCoverageManifest:
    """Explicit audit of what the historical archive can and cannot verify."""

    symbol: str
    timeframe: str
    total_bars: int
    first_bar_open_utc: str
    last_bar_open_utc: str
    first_bar_time_s: int
    last_bar_time_s: int
    calendar_span_days: float
    data_origin: str  # MT5_TERMINAL for acquired research; never offline imports
    verified_spread_fraction: float
    spread_provenance: str  # "MT5_BAR_REPORTED_SPREAD_POINTS" | "PARTIAL_MT5_SPREAD" | "MISSING_MT5_SPREAD"
    m1_intrabar_verified_bars: int
    m1_window_coverage_fraction: float
    unexpected_gap_count: int
    ohlc_integrity_passed: bool
    m1_spread_coverage_fraction: float = 0.0
    fallback_spread_points: float | None = None
    fallback_spread_source: str = 'NONE'
    bar_price_side: str = "UNVERIFIED"  # A bid quote must never be silently treated as midpoint.

    @property
    def verified_spread_coverage(self) -> float:
        return self.verified_spread_fraction

    @property
    def source_type(self) -> str:
        return f"{self.data_origin}:{self.spread_provenance}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataQualityAuditor:
    """Verifies OHLCV monotonicity, price positivity, M1 sequence coverage, and explicit data origin."""

    @staticmethod
    def validate_m1(bars: list[OHLCVBar]) -> None:
        """Reject malformed intrabar evidence before freezing a research dataset.

        Missing minute spreads are allowed as missing *evidence*, not substituted
        with an H4 spread. The M1 resolver must fail conservatively when needed.
        """
        previous = None
        for bar in bars:
            if (bar.time_s < 0 or bar.time_s % 60
                    or (previous is not None and bar.time_s <= previous)
                    or not all(math.isfinite(x) and x > 0 for x in
                               (bar.open,bar.high,bar.low,bar.close))
                    or bar.low > min(bar.open,bar.close)
                    or bar.high < max(bar.open,bar.close)):
                raise ValueError(f"Invalid M1 intrabar price/time evidence at {bar.time_s}")
            spread = bar.spread_points
            if spread is not None and (not math.isfinite(spread) or spread <= 0):
                raise ValueError(f"Invalid M1 spread evidence at {bar.time_s}")
            previous = bar.time_s

    @staticmethod
    def audit_bars(
        symbol: str,
        timeframe: str,
        bars: list[OHLCVBar],
        *,
        m1_time_s: np.ndarray | None = None,
        m1_spread_points: np.ndarray | None = None,
        fallback_spread_points: float | None = None,
        fallback_spread_source: str = 'NONE',
        bar_price_side: str = "UNVERIFIED",
    ) -> DataCoverageManifest:
        if not bars:
            raise ValueError(f"Cannot audit empty bar sequence for {symbol} {timeframe}")
        if bar_price_side not in ("UNVERIFIED", "BID", "MID", "ASK"):
            raise ValueError("Invalid bar price side")

        period_s = timeframe_to_seconds(timeframe)
        ohlc_ok = True
        unexpected_gaps = 0
        populated_spreads = 0
        covered_by_m1 = 0

        has_m1_arr = m1_time_s is not None and len(m1_time_s) > 0
        if m1_spread_points is not None and (m1_time_s is None or len(m1_spread_points) != len(m1_time_s)):
            raise ValueError("M1 spread and time arrays must be aligned")
        m1_spread_fraction = (round(float(np.count_nonzero(np.isfinite(m1_spread_points) &
                             (m1_spread_points > 0))) / len(m1_spread_points), 4)
                             if m1_spread_points is not None and len(m1_spread_points) else 0.0)
        # Count discontinuities only once for the entire M1 series. Prefix
        # sums verify a bar contains *every* minute without allocating a fresh
        # 60/240/480-element expected-time array for each source candle.
        if has_m1_arr:
            m1_times = np.asarray(m1_time_s, dtype=np.int64)
            if np.any(np.diff(m1_times) <= 0) or np.any(m1_times % 60):
                raise ValueError('Unsorted, duplicated or unaligned M1 timestamps')
            discontinuities = np.empty(len(m1_times), dtype=np.int64)
            discontinuities[0] = 0
            discontinuities[1:] = np.cumsum(np.diff(m1_times) != 60)
        else:
            m1_times = np.empty(0, dtype=np.int64)
            discontinuities = np.empty(0, dtype=np.int64)

        for i, bar in enumerate(bars):
            if (
                not all(math.isfinite(v) and v > 0 for v in (bar.open, bar.high, bar.low, bar.close))
                or bar.low > min(bar.open, bar.close) + 1e-9
                or bar.high < max(bar.open, bar.close) - 1e-9
                or bar.high < bar.low
            ):
                ohlc_ok = False
                raise ValueError(f"Corrupt OHLC bar on {symbol} {timeframe} at {bar.time_s}: {bar}")

            if bar.spread_points is not None and math.isfinite(bar.spread_points) and bar.spread_points > 0:
                populated_spreads += 1

            if has_m1_arr:
                i0 = int(np.searchsorted(m1_times, bar.time_s, side="left"))
                i1 = int(np.searchsorted(m1_times, bar.time_s + period_s, side="left"))
                expected_count = period_s // 60
                if (i1-i0 == expected_count and
                        m1_times[i0] == bar.time_s and
                        m1_times[i1-1] == bar.time_s+period_s-60 and
                        discontinuities[i1-1] == discontinuities[i0]):
                    covered_by_m1 += 1

            if i > 0:
                dt = bar.time_s - bars[i - 1].time_s
                if dt <= 0:
                    raise ValueError(
                        f"Non-monotonic bar timestamps on {symbol} {timeframe}: {bars[i - 1].time_s} -> {bar.time_s}"
                    )
                if dt > max(period_s * 3, 72 * 3600):
                    unexpected_gaps += 1

        n = len(bars)
        m1_cov_frac = round(covered_by_m1 / max(1, n), 4) if has_m1_arr else 0.0

        if bar_price_side != 'BID':
            raise ValueError('Only MT5 bid-price history is accepted for research')
        spread_frac=round(populated_spreads/max(1,n),4)
        provenance=('MT5_BAR_REPORTED_SPREAD_POINTS' if spread_frac>=0.95
                    else 'PARTIAL_MT5_SPREAD' if spread_frac>0 else 'MISSING_MT5_SPREAD')

        span_days = round((bars[-1].time_s + period_s - bars[0].time_s) / 86400.0, 2)

        return DataCoverageManifest(
            symbol=symbol,
            timeframe=timeframe,
            total_bars=n,
            first_bar_open_utc=datetime.fromtimestamp(bars[0].time_s, UTC).isoformat(),
            last_bar_open_utc=datetime.fromtimestamp(bars[-1].time_s, UTC).isoformat(),
            first_bar_time_s=bars[0].time_s,
            last_bar_time_s=bars[-1].time_s,
            calendar_span_days=span_days,
            data_origin='MT5_TERMINAL',
            verified_spread_fraction=spread_frac,
            spread_provenance=provenance,
            m1_intrabar_verified_bars=covered_by_m1,
            m1_window_coverage_fraction=m1_cov_frac,
            unexpected_gap_count=unexpected_gaps,
            ohlc_integrity_passed=ohlc_ok,
            bar_price_side=bar_price_side,
            fallback_spread_points=fallback_spread_points,
            fallback_spread_source=fallback_spread_source,
            m1_spread_coverage_fraction=m1_spread_fraction,
        )
