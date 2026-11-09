"""MT5-only, in-memory historical datasets with fingerprints and M1 evidence."""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from array import array as typed_array
import hashlib
import json

import numpy as np

from quant_rl.config.schema import QuantRLConfig
from quant_rl.config.logger import log
from quant_rl.data.mt5_source import (
    MT5DataSource,
    OHLCVBar,
    SymbolSpec,
)
from quant_rl.data.quality import DataCoverageManifest, DataQualityAuditor
from quant_rl.data.time_alignment import timeframe_to_seconds
from quant_rl.data.rolling_window import rolling_request_start_utc, select_rolling_bars
import time
from datetime import datetime, timezone


@dataclass(frozen=True)
class FrozenDataset:
    """Immutable, fingerprinted OHLCV dataset with coverage manifest, symbol spec, and M1 intrabar arrays."""

    symbol: str
    timeframe: str
    fingerprint: str
    bars: tuple[OHLCVBar, ...]
    time_s: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_volume: np.ndarray
    spread_points: np.ndarray
    m1_time_s: np.ndarray
    m1_high: np.ndarray
    m1_low: np.ndarray
    m1_spread_points: np.ndarray
    spec: SymbolSpec
    manifest: DataCoverageManifest
    fallback_spread_points: float | None = None
    broker_server: str = ""
    # Set only by freeze() after validating every timestamp in the M1 stream.
    m1_ordered: bool = False


class MT5History:
    """Fresh MT5 retrieval; keep only current in-memory training data.

    MT5 is the sole source. Fingerprints and coverage reports are retained,
    but no duplicate historical database or NPZ snapshots are written.
    """

    def __init__(self, config: QuantRLConfig | None = None) -> None:
        self.config = config

    @staticmethod
    def compute_fingerprint(symbol: str, timeframe: str, arrays: dict[str, np.ndarray]) -> str:
        """
        Compute SHA-256 fingerprint across ALL price and M1 intrabar arrays.
        Changing any M1 high/low/timestamp deterministically changes the dataset fingerprint.
        """
        hasher = hashlib.sha256()
        hasher.update(f"{symbol}|{timeframe}|".encode("utf-8"))
        for key in (
            "time_s",
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread_points",
            "m1_time_s",
            "m1_high",
            "m1_low",
            "m1_spread_points",
            "spec_json",
            "origin_json",
            "broker_server_json",
            "manifest_json",
        ):
            if key in arrays:
                hasher.update(key.encode("utf-8"))
                hasher.update(np.ascontiguousarray(arrays[key]).tobytes())
        return hasher.hexdigest()[:16]

    @staticmethod
    def historical_spread_estimate(bars: list[OHLCVBar], *, train_end_utc_s: int | None = None) -> float | None:
        """Broker-reported spread estimate from the first half of available H1 rows.

        The estimate is a fallback, not proof of the final calendar train split
        or of the spread at a missing bar.
        """
        training = (bars[:len(bars)//2] if train_end_utc_s is None
                    else [b for b in bars if b.time_s < train_end_utc_s])
        measured = [float(b.spread_points) for b in training
                    if b.spread_points is not None and
                       np.isfinite(b.spread_points) and b.spread_points > 0]
        if len(measured) < 64:
            return None
        return float(np.percentile(np.asarray(measured), 95))

    def fetch(self, symbol: str, base_timeframe: str = "H1", *, deadline_monotonic: float | None = None) -> FrozenDataset:
        """Always fetch fresh MT5 history; never read snapshots or offline data."""
        if self.config is None:
            raise ValueError('MT5 connection configuration is required for market acquisition')
        cfg = self.config
        mt5_src = MT5DataSource(
            login=cfg.broker_connection.mt5_login,
            password=cfg.broker_connection.mt5_password,
            server=cfg.broker_connection.mt5_server,
            path=cfg.broker_connection.mt5_path,
        )
        try:
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f'Research session deadline reached before MT5 acquisition for {symbol}')
            mt5_src.connect()
            account=mt5_src._mt5.account_info()
            if account is None:
                raise ConnectionError('MT5 account metadata unavailable')
            spec=mt5_src.get_symbol_spec(symbol,str(account.currency))
            # A contemporaneous broker quote is a cost-estimation basis, NOT
            # a claim about the historical spread at every candle.
            info=getattr(mt5_src._mt5, "symbol_info", lambda _: None)(symbol)
            tick=getattr(mt5_src._mt5, "symbol_info_tick", lambda _: None)(symbol)
            candidates=[]
            if info is not None and float(getattr(info,'spread',0) or 0)>0:
                candidates.append(float(info.spread))
            if tick is not None:
                bid=float(getattr(tick,'bid',0) or 0)
                ask=float(getattr(tick,'ask',0) or 0)
                if bid>0 and ask>bid and spec.point>0:
                    candidates.append((ask-bid)/spec.point)
            broker_spread=max(candidates) if candidates else None
            now_s=int(time.time())
            requested_start=rolling_request_start_utc(now_s,cfg.research.lookback_years)
            downloaded=mt5_src.fetch_bars_since(symbol,base_timeframe,requested_start)
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f'Research session deadline reached during base-bar acquisition for {symbol}')
            bars,window_start,window_end=select_rolling_bars(
                downloaded,base_timeframe,cfg.research.lookback_years,now_utc_s=now_s)
            log.info('MT5 DATA', f'{symbol}: rolling {cfg.research.lookback_years}-year '
                     f'window {datetime.fromtimestamp(window_start,timezone.utc).isoformat()} '
                     f'to {datetime.fromtimestamp(window_end,timezone.utc).isoformat()}; '
                     f'{len(bars)} completed {base_timeframe} candles')
            raw_count=sum(b.spread_points is not None and b.spread_points>0 for b in bars)
            training_end=int(window_start+(window_end-window_start)*0.50)
            training_estimate=self.historical_spread_estimate(bars,train_end_utc_s=training_end)
            if broker_spread is not None and training_estimate is not None:
                broker_spread=max(broker_spread,training_estimate)
                spread_basis='MT5_BROKER_SNAPSHOT_AND_TRAINING_P95_MAX'
            elif broker_spread is not None:
                spread_basis='MT5_BROKER_SPREAD_SNAPSHOT'
            elif training_estimate is not None:
                broker_spread=training_estimate
                spread_basis='MT5_TRAINING_SPREAD_P95_ESTIMATE'
            else:
                spread_basis='NONE'
            if spread_basis=='MT5_TRAINING_SPREAD_P95_ESTIMATE':
                log.warning('MT5 DATA', f'{symbol}: no usable broker spread quote; using '
                            f'{broker_spread:.3f} points (95th percentile of broker-reported '
                            'spread observations in training history). Economic costs are estimated.')
            log.info('MT5 DATA', f'{symbol}: downloaded {len(bars)} completed {base_timeframe} candles in rolling window; '
                     f'{raw_count}/{len(bars)} have broker-reported historical spread; '
                     f'spread_estimate={broker_spread if broker_spread else "unavailable"} points; '
                     f'spread_basis={spread_basis}')
            # M1 evidence must stop at the last completed base candle, not at
            # the execution instant (which may fall inside a forming candle).
            m1_end=bars[-1].time_s+timeframe_to_seconds(base_timeframe)
            if hasattr(mt5_src, 'iter_m1_range'):
                m1=mt5_src.iter_m1_range(symbol,bars[0].time_s,m1_end,
                        deadline_monotonic=deadline_monotonic)
            else:
                # Test doubles and external adapters predating the streaming contract.
                m1=mt5_src.fetch_m1_range(symbol,bars[0].time_s,m1_end)
        except TimeoutError:
            # A session deadline is a control-flow signal, not an MT5 connection failure.
            raise
        except Exception as exc:
            raise ConnectionError(f'MT5 data acquisition failed for {symbol}: {exc}') from exc
        finally:
            mt5_src.close()
        frozen=self.freeze(symbol,base_timeframe,bars,spec,m1_bars=m1,
                           deadline_monotonic=deadline_monotonic,
                           fallback_spread_points=broker_spread,
                           fallback_spread_source=spread_basis,
                           broker_server=str(getattr(account,'server','') or ''))
        return frozen

    def freeze(
        self,
        symbol: str,
        timeframe: str,
        bars: list[OHLCVBar],
        spec: SymbolSpec,
        m1_bars=None,
        *,
        deadline_monotonic: float | None = None,
        fallback_spread_points: float | None = None,
        fallback_spread_source: str = "NONE",
        broker_server: str = "",
    ) -> FrozenDataset:
        # Compact typed buffers replace millions of Python OHLCVBar objects.
        # The MT5 iterator yields one row at a time from bounded 14-day requests;
        # only the final immutable NumPy evidence arrays remain resident.
        m1_times = typed_array('q')
        m1_highs = typed_array('d')
        m1_lows = typed_array('d')
        m1_spreads = typed_array('d')
        previous_m1 = None
        for row_index, bar in enumerate(() if m1_bars is None else m1_bars):
            if row_index % 4_096 == 0 and deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f'Research session deadline reached while freezing M1 evidence for {symbol}')
            if (bar.time_s < 0 or bar.time_s % 60
                    or (previous_m1 is not None and bar.time_s <= previous_m1)
                    or not all(np.isfinite(x) and x > 0 for x in (bar.open, bar.high, bar.low, bar.close))
                    or bar.low > min(bar.open, bar.close)
                    or bar.high < max(bar.open, bar.close)):
                raise ValueError(f'Invalid M1 intrabar price/time evidence at {bar.time_s}')
            spread = float(bar.spread_points) if bar.spread_points is not None else float('nan')
            if np.isfinite(spread) and spread <= 0:
                raise ValueError(f'Invalid M1 spread evidence at {bar.time_s}')
            m1_times.append(int(bar.time_s))
            m1_highs.append(float(bar.high))
            m1_lows.append(float(bar.low))
            m1_spreads.append(spread)
            previous_m1 = bar.time_s
        # A final check catches deadlines crossed after the last periodic
        # checkpoint, including streams shorter than the polling interval.
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f'Research session deadline reached while freezing M1 evidence for {symbol}')
        m1_time_s = np.frombuffer(m1_times, dtype=np.int64).copy()
        m1_high = np.frombuffer(m1_highs, dtype=np.float64).copy()
        m1_low = np.frombuffer(m1_lows, dtype=np.float64).copy()
        m1_spread_points = np.frombuffer(m1_spreads, dtype=np.float64).copy()
        arrays = {
            "time_s": np.array([b.time_s for b in bars], dtype=np.int64),
            "open": np.array([b.open for b in bars], dtype=np.float64),
            "high": np.array([b.high for b in bars], dtype=np.float64),
            "low": np.array([b.low for b in bars], dtype=np.float64),
            "close": np.array([b.close for b in bars], dtype=np.float64),
            "tick_volume": np.array([b.tick_volume for b in bars], dtype=np.int64),
            "spread_points": np.array(
                [float(b.spread_points) if b.spread_points is not None else np.nan for b in bars],
                dtype=np.float64,
            ),
            "m1_time_s": m1_time_s,
            "m1_high": m1_high,
            "m1_low": m1_low,
            "m1_spread_points": m1_spread_points,
            "spec_json": np.array(json.dumps(asdict(spec), sort_keys=True, separators=(",", ":"))),
            "origin_json": np.array(json.dumps({"origin": "MT5_TERMINAL"}, sort_keys=True, separators=(",", ":"))),
            "broker_server_json": np.array(json.dumps(broker_server)),
        }
        manifest = DataQualityAuditor.audit_bars(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
            m1_time_s=m1_time_s,
            m1_spread_points=m1_spread_points,
            bar_price_side="BID",
            fallback_spread_points=fallback_spread_points,
            fallback_spread_source=(fallback_spread_source if fallback_spread_points is not None else 'NONE'),
        )
        # Provenance is part of the content address; stale or manually
        # edited manifests must not certify a different price-side basis.
        arrays["manifest_json"] = np.array(json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")))
        fingerprint = self.compute_fingerprint(symbol, timeframe, arrays)
        for array in arrays.values():
            array.flags.writeable = False
        return FrozenDataset(
            symbol=symbol,
            timeframe=timeframe,
            fingerprint=fingerprint,
            bars=tuple(bars),
            time_s=arrays["time_s"],
            open=arrays["open"],
            high=arrays["high"],
            low=arrays["low"],
            close=arrays["close"],
            tick_volume=arrays["tick_volume"],
            spread_points=arrays["spread_points"],
            m1_time_s=m1_time_s,
            m1_high=m1_high,
            m1_low=m1_low,
            m1_spread_points=m1_spread_points,
            spec=spec,
            manifest=manifest,
            fallback_spread_points=fallback_spread_points,
            broker_server=broker_server,
            m1_ordered=bool(len(m1_time_s) > 0),
        )
