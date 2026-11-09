"""Deterministic market fixtures for unit tests only; never imported by runtime."""
from quant_rl.data.mt5_source import OHLCVBar, SymbolSpec, split_fx_symbol
from tests._specs import default_symbol_spec
import math, hashlib, time
import numpy as np

class SimulatedFXDataSource:
    """
    Deterministic multi-regime FX market generator producing UTC-aligned H1/M15/M30 base bars
    with regime-switching trend persistence, volatility clustering, and session spread dynamics.
    """

    def __init__(self, seed: int = 42, account_currency: str = "USD", default_count: int = 3600) -> None:
        self.seed = seed
        self.account_currency = account_currency
        self.default_count = default_count
        self._h1_cache: dict[str, list[OHLCVBar]] = {}

    def get_symbol_spec(self, symbol: str, account_currency: str = "USD") -> SymbolSpec:
        bars = self._ensure_h1_series(symbol, 500)
        last_px = bars[-1].close if bars else None
        return default_symbol_spec(symbol, reference_price=last_px, account_currency=account_currency)

    def fetch_symbol_spec(self, symbol: str, account_currency: str = "USD") -> SymbolSpec:
        return self.get_symbol_spec(symbol, account_currency=account_currency)

    def _ensure_h1_series(self, symbol: str, min_h1_bars: int) -> list[OHLCVBar]:
        if symbol in self._h1_cache and len(self._h1_cache[symbol]) >= min_h1_bars:
            return self._h1_cache[symbol]

        base, quote, _ = split_fx_symbol(symbol)
        pair = base + quote
        sym_hash = int(hashlib.sha256(f"{pair}:{self.seed}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(sym_hash)

        anchor_prices = {
            "EURUSD": 1.0800,
            "GBPUSD": 1.2650,
            "USDJPY": 148.00,
            "USDCHF": 0.8900,
            "USDCAD": 1.3550,
            "AUDUSD": 0.6600,
            "NZDUSD": 0.6100,
        }
        start_price = anchor_prices.get(pair, 1.0000)
        spec = default_symbol_spec(symbol, reference_price=start_price, account_currency=self.account_currency)
        digits = 3 if quote == "JPY" else 5

        total_h1 = max(min_h1_bars, self.default_count)
        end_utc_s = int(time.time()) - (int(time.time()) % 3600)
        timestamps: list[int] = []
        cursor = end_utc_s - 3600
        while len(timestamps) < total_h1:
            weekday = time.gmtime(cursor).tm_wday
            hour = time.gmtime(cursor).tm_hour
            if not (weekday == 5 or (weekday == 4 and hour >= 22) or (weekday == 6 and hour < 22)):
                timestamps.append(cursor)
            cursor -= 3600
        timestamps.reverse()

        bars: list[OHLCVBar] = []
        price = start_price
        regime_vol = 0.0012
        momentum = 0.0
        for idx, ts in enumerate(timestamps):
            hour = (ts // 3600) % 24
            session_mult = 1.35 if 12 <= hour <= 16 else (1.10 if 7 <= hour <= 19 else 0.75)
            if idx % 48 == 0:
                regime_vol = float(np.clip(regime_vol * rng.uniform(0.82, 1.22), 0.0006, 0.0028))
            shock = float(rng.normal(0.0, regime_vol * session_mult))
            drift = 0.22 * momentum - 0.03 * (price - start_price) / start_price * regime_vol
            ret = drift + shock
            momentum = 0.74 * momentum + 0.26 * ret

            open_px = round(price, digits)
            close_px = round(max(open_px * (1.0 + ret), spec.point * 100), digits)
            wick_up = abs(float(rng.normal(0.0, regime_vol * 0.65 * session_mult))) * open_px
            wick_dn = abs(float(rng.normal(0.0, regime_vol * 0.65 * session_mult))) * open_px
            high_px = round(max(open_px, close_px) + wick_up, digits)
            low_px = round(max(spec.point * 50, min(open_px, close_px) - wick_dn), digits)
            low_px = min(low_px, open_px, close_px)
            high_px = max(high_px, open_px, close_px)

            base_spread_pts = 9.0 if pair in ("EURUSD", "USDJPY") else 13.0
            spread_pts = round(base_spread_pts * (1.25 if hour in (21, 22, 23, 0) else 1.0), 1)
            vol = int(max(100, round(1200 * session_mult * (1.0 + abs(ret) / max(regime_vol, 1e-6)))))

            bars.append(
                OHLCVBar(
                    time_s=ts,
                    open=open_px,
                    high=high_px,
                    low=low_px,
                    close=close_px,
                    tick_volume=vol,
                    spread_points=spread_pts,
                )
            )
            price = close_px

        self._h1_cache[symbol] = bars
        return bars

    def fetch_bars(self, symbol: str, timeframe: str, count: int | None = None) -> list[OHLCVBar]:
        from quant_rl.data.time_alignment import resample_ohlcv_utc, timeframe_to_seconds

        n_req = int(count) if count is not None else self.default_count
        tf_u = timeframe.strip().upper()
        period_s = timeframe_to_seconds(tf_u)
        if period_s < 3600:
            h1_needed = max(500, math.ceil(n_req * period_s / 3600) + 32)
            h1_bars = self._ensure_h1_series(symbol, h1_needed)
            sub_bars = self._subdivide_h1_bars(symbol, h1_bars, period_s)
            return sub_bars[-n_req:]

        ratio = max(1, period_s // 3600)
        h1_bars = self._ensure_h1_series(symbol, n_req * ratio + 64)
        if tf_u == "H1":
            return h1_bars[-n_req:]
        resampled = resample_ohlcv_utc(h1_bars, tf_u, source_timeframe="H1")
        return resampled[-n_req:]

    def _subdivide_h1_bars(self, symbol: str, h1_bars: list[OHLCVBar], sub_period_s: int) -> list[OHLCVBar]:
        _, quote, _ = split_fx_symbol(symbol)
        digits = 3 if quote == "JPY" else 5
        steps = max(1, 3600 // sub_period_s)
        out: list[OHLCVBar] = []
        for bar in h1_bars:
            bullish = bar.close >= bar.open
            waypoints = (
                [bar.open, bar.low, bar.high, bar.close]
                if bullish
                else [bar.open, bar.high, bar.low, bar.close]
            )
            interp = np.interp(np.linspace(0.0, 1.0, steps + 1), [0.0, 0.33, 0.67, 1.0], waypoints)
            for s in range(steps):
                o = round(float(interp[s]), digits)
                c = round(float(interp[s + 1]), digits)
                h = round(min(bar.high, max(o, c) + (bar.high - bar.low) * 0.12), digits)
                l = round(max(bar.low, min(o, c) - (bar.high - bar.low) * 0.12), digits)
                if s == 0:
                    o = bar.open
                if s == steps - 1:
                    c = bar.close
                h = max(h, o, c)
                l = min(l, o, c)
                out.append(
                    OHLCVBar(
                        time_s=bar.time_s + s * sub_period_s,
                        open=o,
                        high=h,
                        low=low_val if (low_val := l) <= h else h,
                        close=c,
                        tick_volume=max(1, bar.tick_volume // steps),
                        spread_points=bar.spread_points,
                    )
                )
        return out

    def fetch_m1_range(self, symbol: str, start_utc_s: int, end_utc_s: int) -> list[OHLCVBar]:
        h1_bars = self._ensure_h1_series(symbol, self.default_count)
        window = [b for b in h1_bars if start_utc_s <= b.time_s < end_utc_s]
        if not window:
            return []
        return self._subdivide_h1_bars(symbol, window, 60)
