"""Direct MetaTrader 5 UTC market data adapter."""

from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
from datetime import datetime, timezone
import math
import time
from quant_rl.config.logger import log



@dataclass(frozen=True)
class OHLCVBar:
    """Canonical UTC OHLCV bar (`time_s` is bar open time in UTC seconds)."""

    time_s: int
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread_points: float | None = None


@dataclass(frozen=True)
class SymbolSpec:
    """Authoritative instrument specification including direction permissions and swap rates."""

    name: str
    base_currency: str
    quote_currency: str
    point: float
    trade_tick_size: float
    trade_tick_value: float
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level_points: float = 2.0
    freeze_level_points: float = 0.0
    contract_size: float = 100_000.0
    # MT5 SYMBOL_TRADE_MODE: 0=DISABLED, 1=LONGONLY, 2=SHORTONLY, 3=CLOSEONLY, 4=FULL
    trade_mode: int = 4
    swap_long_per_lot_per_day: float = -4.5
    swap_short_per_lot_per_day: float = -4.5
    # MT5 swap rates are raw values whose units depend on `swap_mode`.
    # They MUST NOT be silently treated as deposit-currency cash per lot.
    swap_mode: int | None = None
    swap_rollover3days: int | None = None
    # Optional explicitly normalized deposit-currency rollover amounts.
    # An imported metadata record is still unverified until broker-reconciled.
    financing_long_account_ccy_per_lot_rollover: float | None = None
    financing_short_account_ccy_per_lot_rollover: float | None = None
    financing_rollover_utc_hour: int | None = None
    financing_triple_weekday_utc: int | None = None

    @property
    def symbol(self) -> str:
        return self.name

    @property
    def pip_size(self) -> float:
        return 0.01 if self.quote_currency == "JPY" else 0.0001

    @property
    def tradable(self) -> bool:
        return self.trade_mode in (1, 2, 4)

    def allows_direction(self, direction: str) -> bool:
        """Check whether `trade_mode` permits opening a position in `direction` (`BUY` or `SELL`)."""
        d = str(direction).upper()
        if self.trade_mode == 4:
            return True
        if self.trade_mode == 1 and d == "BUY":
            return True
        if self.trade_mode == 2 and d == "SELL":
            return True
        return False


def split_fx_symbol(symbol: str) -> tuple[str, str, str]:
    """Split a spot FX symbol into `(base, quote, broker_suffix)`, supporting `.pro`, `m`, etc."""
    cleaned = str(symbol or "").strip()
    if len(cleaned) < 6 or not cleaned[:6].isalpha():
        raise ValueError(f"Expected 6-letter FX symbol prefix (with optional broker suffix), got: {symbol!r}")
    base = cleaned[:3].upper()
    quote = cleaned[3:6].upper()
    suffix = cleaned[6:]
    return base, quote, suffix


def pip_size_for_symbol(symbol: str) -> float:
    """Return canonical FX pip size (0.01 for JPY quote pairs, 0.0001 otherwise), tolerating broker suffixes."""
    _, quote, _ = split_fx_symbol(symbol)
    return 0.01 if quote == "JPY" else 0.0001



def broker_timestamp_utc(raw_seconds: int | float) -> int:
    """Return an MT5 API timestamp as UTC epoch seconds without adjustment.

    MetaTrader5 Python rate, tick and deal timestamps are UTC-based. Broker
    wall-clock timezone is a separate scheduling concern and must never be
    applied to market-data or execution timestamps.
    """
    return int(raw_seconds)


def clock_offset_diagnostic(now_utc_s: int, raw_seconds: int | float) -> str:
    """Report the raw UTC timestamp difference without guessing a correction."""
    raw = int(raw_seconds)
    if raw <= 0:
        return 'no MT5 UTC timestamp available'
    age = int(now_utc_s) - raw
    raw_iso = datetime.fromtimestamp(raw, timezone.utc).isoformat()
    now_iso = datetime.fromtimestamp(int(now_utc_s), timezone.utc).isoformat()
    return (f'now_utc={now_iso}; mt5_timestamp_utc={raw_iso}; age={age}s; '
            'MT5 timestamps are used as UTC; no broker offset is applied')


def broker_tick_age(now_utc_s: int, raw_seconds: int | float) -> int | None:
    """Return quote age using the UTC epoch supplied by the MT5 API."""
    if not raw_seconds:
        return None
    return int(now_utc_s) - broker_timestamp_utc(raw_seconds)



def account_is_demo(account, mt5_module=None) -> bool:
    """Return True only when MT5 explicitly identifies the account as DEMO."""
    mode = getattr(account, "trade_mode", None)
    if mode is None:
        return False
    demo_mode = getattr(mt5_module, "ACCOUNT_TRADE_MODE_DEMO", 0) if mt5_module is not None else 0
    try:
        return int(mode) == int(demo_mode)
    except (TypeError, ValueError):
        return False

def broker_spec_hash(spec: SymbolSpec) -> str:
    """Stable economic certificate for an MT5 symbol specification.

    Current trade permission is session state, not durable instrument
    economics, so it is deliberately excluded from the certificate.
    """
    payload = asdict(spec)
    payload.pop("trade_mode", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()[:20]



class MT5DataSource:
    """
    MetaTrader 5 data source. All market-data timestamps remain in the UTC
    epoch returned by the MT5 Python API. Broker-local time is never applied here.
    """

    def __init__(
        self,
        login: int = 0,
        password: str = "",
        server: str = "",
        path: str = "",
    ) -> None:
        # A single MT5 connection contract is used for historical research and LIVE.
        # Environment secrets take precedence over config.yaml; no passwords are logged.
        import os
        env_login = os.getenv('QUANT_MT5_LOGIN', '').strip()
        self.login = int(env_login) if env_login else login
        self.password = os.getenv('QUANT_MT5_PASSWORD') or password
        self.server = os.getenv('QUANT_MT5_SERVER') or server
        self.path = os.getenv('QUANT_MT5_PATH') or path
        self._mt5 = None

    def connect(self) -> None:
        import MetaTrader5 as mt5

        kwargs = {}
        if self.path:
            kwargs["path"] = self.path
        if self.login > 0:
            kwargs["login"] = self.login
        if self.password:
            kwargs["password"] = self.password
        if self.server:
            kwargs["server"] = self.server
        if not mt5.initialize(**kwargs):
            raise ConnectionError(f"MT5 initialization failed: {mt5.last_error()}")
        self._mt5 = mt5
        if self.login > 0:
            account = mt5.account_info()
            if account is None or int(account.login) != self.login:
                mt5.shutdown()
                self._mt5 = None
                raise ConnectionError('Connected MT5 login does not match requested login')

    def get_symbol_spec(self, symbol: str, account_currency: str = "USD") -> SymbolSpec:
        if self._mt5 is None:
            raise ConnectionError("MT5 terminal not connected; symbol specifications unavailable")
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol {symbol} unavailable on MT5 server")
        base, quote, _ = split_fx_symbol(symbol)
        return SymbolSpec(
            name=symbol,
            base_currency=base,
            quote_currency=quote,
            point=float(info.point),
            trade_tick_size=float(info.trade_tick_size),
            trade_tick_value=float(info.trade_tick_value),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            stops_level_points=float(info.trade_stops_level),
            freeze_level_points=float(info.trade_freeze_level),
            contract_size=float(info.trade_contract_size),
            trade_mode=int(info.trade_mode),
            swap_long_per_lot_per_day=float(getattr(info, "swap_long", -4.5)),
            swap_short_per_lot_per_day=float(getattr(info, "swap_short", -4.5)),
            swap_mode=int(getattr(info, "swap_mode")) if getattr(info, "swap_mode", None) is not None else None,
            swap_rollover3days=int(getattr(info, "swap_rollover3days")) if getattr(info, "swap_rollover3days", None) is not None else None,
        )

    def fetch_bars(self, symbol: str, timeframe: str, count: int) -> list[OHLCVBar]:
        if self._mt5 is None:
            raise ConnectionError("MT5DataSource is not connected")
        from quant_rl.data.time_alignment import resample_ohlcv_utc, timeframe_to_seconds

        tf_u = timeframe.strip().upper()
        mt5_tf = getattr(self._mt5, f"TIMEFRAME_{tf_u}", None)
        if mt5_tf is None or tf_u in ("H2", "H6", "H8", "H12"):
            ratio = max(1, timeframe_to_seconds(tf_u) // 3600)
            h1_bars = self.fetch_bars(symbol, "H1", count * ratio + 24)
            return resample_ohlcv_utc(h1_bars, tf_u, source_timeframe="H1")[-count:]

        if not self._mt5.symbol_select(symbol, True):
            raise ConnectionError(f"MT5 symbol selection failed for {symbol}: {self._mt5.last_error()}")
        rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count + 4)
        if rates is None or len(rates) == 0:
            return []
        now_utc = int(time.time())
        period_s = timeframe_to_seconds(tf_u)
        bars: list[OHLCVBar] = []
        for r in rates:
            true_utc_s = int(r["time"])
            if true_utc_s + period_s > now_utc:
                continue
            bars.append(
                OHLCVBar(
                    time_s=true_utc_s,
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    tick_volume=int(r["tick_volume"]),
                    spread_points=float(r["spread"]) if r["spread"] > 0 else None,
                )
            )
        return bars[-count:]

    def fetch_bars_since(self, symbol: str, timeframe: str, start_utc_s: int) -> list[OHLCVBar]:
        """Fetch completed bars since a caller-supplied UTC rolling-request boundary."""
        if self._mt5 is None:
            raise ConnectionError('MT5DataSource is not connected')
        from datetime import datetime, timezone
        from quant_rl.data.time_alignment import resample_ohlcv_utc, timeframe_to_seconds
        tf = timeframe.strip().upper()
        tf_seconds = timeframe_to_seconds(tf)
        mt5_tf = getattr(self._mt5, f'TIMEFRAME_{tf}', None)
        if mt5_tf is None or tf in ('H2','H6','H8','H12'):
            if tf_seconds < 3600:
                raise ValueError(f'No safe native source for {tf}')
            bars = self.fetch_bars_since(symbol, 'H1', start_utc_s)
            return resample_ohlcv_utc(bars,tf,source_timeframe="H1")
        if not self._mt5.symbol_select(symbol,True):
            raise ConnectionError(f'MT5 symbol selection failed: {symbol}')
        now = int(time.time())
        rates=self._mt5.copy_rates_range(
            symbol,mt5_tf,
            datetime.fromtimestamp(start_utc_s,timezone.utc),
            datetime.fromtimestamp(now,timezone.utc))
        if rates is None:
            raise ConnectionError(f'MT5 date-range retrieval failed: {symbol}: {self._mt5.last_error()}')
        rows={}
        for r in rates:
            ts=int(r['time'])
            if ts < start_utc_s or ts+tf_seconds>now:
                continue
            rows[ts]=OHLCVBar(time_s=ts,open=float(r['open']),high=float(r['high']),
                       low=float(r['low']),close=float(r['close']),
                       tick_volume=int(r['tick_volume']),
                       spread_points=float(r['spread']) if r['spread']>0 else None)
        return [rows[t] for t in sorted(rows)]

    def iter_m1_range(self, symbol: str, start_utc_s: int, end_utc_s: int, *,
                      deadline_monotonic: float | None = None):
        """Yield validated M1 bars window-by-window; never retain the full history here."""
        if self._mt5 is None:
            return
        from datetime import UTC, datetime
        if end_utc_s <= start_utc_s:
            return
        if not self._mt5.symbol_select(symbol, True):
            raise ConnectionError(f"MT5 M1 symbol selection failed for {symbol}: {self._mt5.last_error()}")
        chunk_seconds = 14 * 86400
        windows = (int(end_utc_s) - int(start_utc_s) + chunk_seconds - 1) // chunk_seconds
        log.info('MT5 DATA', f'{symbol}: streaming M1 evidence in {windows} broker requests')
        previous_ts: int | None = None
        for window_index, window_start in enumerate(range(int(start_utc_s), int(end_utc_s), chunk_seconds), start=1):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f'M1 acquisition exceeded the research session deadline for {symbol}')
            window_end = min(window_start + chunk_seconds, int(end_utc_s))
            rates = self._mt5.copy_rates_range(
                symbol, self._mt5.TIMEFRAME_M1,
                datetime.fromtimestamp(window_start, UTC),
                datetime.fromtimestamp(window_end - 1, UTC),
            )
            # MT5 calls are blocking; enforce the deadline as soon as control returns,
            # before inspecting or iterating a potentially large response.
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f'M1 acquisition exceeded the research session deadline for {symbol}')
            if rates is None:
                raise ConnectionError(
                    f"MT5 M1 acquisition failed for {symbol} interval "
                    f"{window_start}:{window_end}: {self._mt5.last_error()}"
                )
            for r in rates:
                raw_ts = int(r['time'])
                if raw_ts < window_start or raw_ts >= window_end:
                    raise ValueError(f'MT5 returned out-of-window M1 candle for {symbol}: {raw_ts}')
                if raw_ts % 60 or (previous_ts is not None and raw_ts <= previous_ts):
                    raise ValueError(f'MT5 M1 candles are misaligned, repeated or unordered: {symbol} {raw_ts}')
                previous_ts = raw_ts
                yield OHLCVBar(
                    time_s=raw_ts, open=float(r['open']), high=float(r['high']),
                    low=float(r['low']), close=float(r['close']),
                    tick_volume=int(r['tick_volume']),
                    spread_points=float(r['spread']) if r['spread'] > 0 else None,
                )
            if window_index == 1 or window_index % 8 == 0 or window_index == windows:
                log.info('MT5 DATA', f'{symbol}: streamed M1 window {window_index}/{windows}')

    def fetch_m1_range(self, symbol: str, start_utc_s: int, end_utc_s: int) -> list[OHLCVBar]:
        """Materializing convenience API; research uses iter_m1_range directly."""
        return list(self.iter_m1_range(symbol, start_utc_s, end_utc_s))


    def close(self) -> None:
        """Release the MT5 terminal connection after historical acquisition."""
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            finally:
                self._mt5 = None

