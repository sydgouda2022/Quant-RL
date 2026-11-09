"""Multi-timeframe point-in-time observation builder."""

from __future__ import annotations

import numpy as np

from quant_rl.data.mt5_source import OHLCVBar, SymbolSpec
from quant_rl.data.time_alignment import (
    align_completed_bars_asof,
    resample_ohlcv_utc,
    timeframe_to_seconds,
)
from quant_rl.market.features import compute_feature_frame
from quant_rl.market.observations import MarketObservationDataset


class ObservationBuilder:
    """
    Constructs strictly causal, multi-timeframe observations.
    When observing higher timeframes (e.g. H8, D1) from an H4 or H1 decision clock,
    only higher-timeframe candles that have ALREADY CLOSED at or before the primary bar's close
    are visible.
    Never downsamples coarse bars into finer timeframes.
    """

    SEQ_COLUMNS = (
        "ret_1",
        "ret_4",
        "body_ratio",
        "range_to_vol",
        "vol_regime_ratio",
        "channel_pos_20",
        "rsi_centered",
        "macd_norm",
        "bb_zscore",
        "spread_to_vol",
    )

    HTF_SUMMARY_COLUMNS = (
        "ret_1",
        "ret_4",
        "vol_regime_ratio",
        "channel_pos_20",
        "rsi_centered",
        "macd_norm",
    )

    def __init__(
        self,
        observation_timeframes: list[str] | None = None,
        sequence_lookback_bars: int = 32,
        include_technical_indicators: bool = True,
    ) -> None:
        self.observation_timeframes = observation_timeframes or ["H1", "H4", "H8", "D1"]
        self.sequence_lookback_bars = sequence_lookback_bars
        self.include_technical_indicators = include_technical_indicators

    def build_dataset(
        self,
        *,
        base_bars: list[OHLCVBar],
        symbol: str,
        primary_timeframe: str,
        volatility_estimator: str = "atr",
        volatility_period: int = 14,
        spec: SymbolSpec | None = None,
        bar_price_side: str = "MID",
        fallback_spread_points: float = float("nan"),
        source_timeframe: str = "H1",
    ) -> MarketObservationDataset:
        if spec is None:
            raise ValueError("A broker-provided SymbolSpec is required; no synthetic defaults")
        resolved_spec = spec
        if bar_price_side not in ("MID", "BID"):
            raise ValueError("Research execution requires explicit MID or BID bar prices")
        # `resample_ohlcv_utc` enforces `target_period_s >= source_period_s` and drops incomplete trailing buckets (#4, #34)
        primary_bars = resample_ohlcv_utc(base_bars, primary_timeframe, source_timeframe=source_timeframe)
        return self.build(
            symbol=symbol,
            primary_timeframe=primary_timeframe,
            primary_bars=primary_bars,
            spec=resolved_spec,
            observation_timeframes=self.observation_timeframes,
            volatility_estimator=volatility_estimator,
            volatility_period=volatility_period,
            sequence_lookback_bars=self.sequence_lookback_bars,
            include_technical_indicators=self.include_technical_indicators,
            base_bars_for_htf=base_bars,
            base_timeframe=source_timeframe,
            bar_price_side=bar_price_side,
            fallback_spread_points=fallback_spread_points,
        )

    @classmethod
    def build(
        cls,
        symbol: str,
        primary_timeframe: str,
        primary_bars: list[OHLCVBar],
        spec: SymbolSpec,
        observation_timeframes: list[str] | None = None,
        volatility_estimator: str = "atr",
        volatility_period: int = 14,
        sequence_lookback_bars: int = 32,
        fallback_spread_points: float = float("nan"),
        include_technical_indicators: bool = True,
        base_bars_for_htf: list[OHLCVBar] | None = None,
        base_timeframe: str | None = None,
        bar_price_side: str = "MID",
    ) -> MarketObservationDataset:
        if bar_price_side not in ("MID", "BID"):
            raise ValueError("Research execution requires explicit MID or BID bar prices")
        if len(primary_bars) < max(32, sequence_lookback_bars + 4):
            raise ValueError(
                f"Insufficient primary bars ({len(primary_bars)}) for {symbol} {primary_timeframe}"
            )

        primary_period_s = timeframe_to_seconds(primary_timeframe)
        base_df, vol_price = compute_feature_frame(
            primary_bars,
            spec,
            volatility_estimator=volatility_estimator,
            volatility_period=volatility_period,
            fallback_spread_points=fallback_spread_points,
            include_technical_indicators=include_technical_indicators,
            timeframe_seconds=primary_period_s,
        )

        open_times = np.array([b.time_s for b in primary_bars], dtype=np.int64)
        decision_close_times = open_times + primary_period_s

        feature_blocks = [base_df.to_numpy(dtype=np.float64)]
        feature_names = list(base_df.columns)

        obs_tfs = observation_timeframes or []
        source_for_htf = base_bars_for_htf if base_bars_for_htf is not None else primary_bars
        for htf in obs_tfs:
            htf_u = htf.strip().upper()
            htf_period_s = timeframe_to_seconds(htf_u)
            if htf_period_s <= primary_period_s:
                continue
            htf_bars = resample_ohlcv_utc(source_for_htf, htf_u,
                source_timeframe=base_timeframe or primary_timeframe)
            if not htf_bars:
                raise ValueError(f"Insufficient {htf_u} history for required model features")
            htf_df, _ = compute_feature_frame(
                htf_bars,
                spec,
                volatility_estimator=volatility_estimator,
                volatility_period=max(5, volatility_period // 2),
                fallback_spread_points=fallback_spread_points,
                include_technical_indicators=include_technical_indicators,
                timeframe_seconds=htf_period_s,
            )
            avail_htf_cols = [c for c in cls.HTF_SUMMARY_COLUMNS if c in htf_df.columns]
            htf_sub = htf_df[avail_htf_cols].to_numpy(dtype=np.float64)
            asof_idx = align_completed_bars_asof(decision_close_times, htf_bars, htf_u)
            aligned = np.zeros((len(primary_bars), htf_sub.shape[1]), dtype=np.float64)
            valid = asof_idx >= 0
            aligned[valid] = htf_sub[asof_idx[valid]]
            feature_blocks.append(aligned)
            for col in avail_htf_cols:
                feature_names.append(f"{htf_u}_{col}")

        feature_matrix = np.hstack(feature_blocks).astype(np.float64)
        feature_matrix = np.clip(np.nan_to_num(feature_matrix, nan=0.0, posinf=5.0, neginf=-5.0), -10.0, 10.0)

        seq_cols = [c for c in cls.SEQ_COLUMNS if c in base_df.columns]
        seq_raw = np.clip(base_df[seq_cols].to_numpy(dtype=np.float64), -10.0, 10.0)
        lookback = int(sequence_lookback_bars)
        padded = np.vstack([np.repeat(seq_raw[:1], lookback - 1, axis=0), seq_raw])
        sequence_tensor = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=lookback, axis=0
        )
        sequence_tensor = np.ascontiguousarray(np.transpose(sequence_tensor, (0, 2, 1)), dtype=np.float64)

        spread_pts = np.array(
            [
                float(b.spread_points)
                if (b.spread_points is not None and b.spread_points > 0)
                else fallback_spread_points
                for b in primary_bars
            ],
            dtype=np.float64,
        )

        return MarketObservationDataset(
            symbol=symbol,
            timeframe=primary_timeframe.upper(),
            timeframe_seconds=primary_period_s,
            feature_names=tuple(feature_names),
            sequence_feature_names=tuple(seq_cols),
            time_s=open_times,
            decision_time_s=decision_close_times,
            open=np.array([b.open for b in primary_bars], dtype=np.float64),
            high=np.array([b.high for b in primary_bars], dtype=np.float64),
            low=np.array([b.low for b in primary_bars], dtype=np.float64),
            close=np.array([b.close for b in primary_bars], dtype=np.float64),
            spread_price=spread_pts * spec.point,
            volatility_price=vol_price,
            feature_matrix=feature_matrix,
            sequence_tensor=sequence_tensor,
            bar_price_side=bar_price_side,
        )
