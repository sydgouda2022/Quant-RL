"""Strict, calendar-year MT5 rolling historical research window.

When an execution timestamp is supplied, the requested window is anchored to
that exact UTC instant. Only fully completed broker candles inside that window
are returned. Without an execution timestamp, the latest completed candle close
is used. Calendar subtraction clamps February 29 to February 28 when needed.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timezone
from quant_rl.data.time_alignment import timeframe_to_seconds

# Broker holidays/weekends can legitimately delay the earliest/last bar.
MAX_COVERAGE_GAP_SECONDS = 7 * 86400
# Request an extra boundary week so a broker holiday at the window start does
# not silently shorten the requested lookback.
ACQUISITION_PAD_SECONDS = 8 * 86400


def rolling_start_utc(end_utc_s: int, lookback_years: int) -> int:
    """Subtract full calendar years at the same UTC time, safely on Feb 29."""
    if not isinstance(lookback_years, int) or isinstance(lookback_years, bool) or lookback_years < 1:
        raise ValueError('lookback_years must be a positive integer')
    end = datetime.fromtimestamp(int(end_utc_s), timezone.utc)
    year = end.year - lookback_years
    if year < 1970:
        raise ValueError('Rolling window starts before supported MT5 history')
    day = min(end.day, calendar.monthrange(year, end.month)[1])
    return int(end.replace(year=year, day=day).timestamp())


def rolling_request_start_utc(now_utc_s: int, lookback_years: int) -> int:
    """Initial MT5 request with buffer; never an offline-data fallback."""
    return max(0, rolling_start_utc(now_utc_s, lookback_years) - ACQUISITION_PAD_SECONDS)


def select_rolling_bars(bars, timeframe: str, lookback_years: int, *, now_utc_s: int | None = None,
                        require_coverage: bool = True):
    """Return (bounded completed bars, start, end); reject a shortened window.

    With ``now_utc_s``, ``end`` is that exact UTC execution instant, rather
    than the close of the most recent candle. This preserves the requested
    four-calendar-year interval even when execution occurs between candle
    closes. Every returned candle is complete and has OPEN >= start and
    CLOSE <= end. Missing weekends are legitimate; a broker with more than a
    week's boundary gap is not.
    """
    period = timeframe_to_seconds(timeframe)
    completed = sorted((b for b in bars if now_utc_s is None or b.time_s + period <= now_utc_s),
                       key=lambda b: b.time_s)
    if not completed:
        raise ValueError('No completed MT5 bars for the requested rolling window')
    if any(b.time_s >= following.time_s for b, following in zip(completed, completed[1:])):
        raise ValueError('Unsorted or duplicate MT5 bar timestamps')
    end = int(now_utc_s) if now_utc_s is not None else completed[-1].time_s + period
    start = rolling_start_utc(end, lookback_years)
    selected = [b for b in completed if start <= b.time_s and b.time_s + period <= end]
    if not selected:
        raise ValueError('No completed MT5 candles within rolling window')
    if require_coverage:
        if selected[0].time_s - start > MAX_COVERAGE_GAP_SECONDS:
            raise ValueError(f'Insufficient MT5 rolling history: requested start '
                             f'{datetime.fromtimestamp(start, timezone.utc).isoformat()}, '
                             f'first available {datetime.fromtimestamp(selected[0].time_s, timezone.utc).isoformat()}')
        # A full window containing a large unseen interruption is not enough
        # evidence for chronological training even if both endpoints exist.
        if any(b.time_s - previous.time_s > MAX_COVERAGE_GAP_SECONDS
               for previous, b in zip(selected, selected[1:])):
            raise ValueError('Insufficient MT5 rolling history: internal broker-history gap exceeds seven days')
        latest_completed_close = completed[-1].time_s + period
        if now_utc_s is not None and now_utc_s - latest_completed_close > MAX_COVERAGE_GAP_SECONDS:
            raise ValueError('MT5 historical endpoint is stale by more than seven days')
    return selected, start, end
