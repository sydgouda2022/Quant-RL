"""Auditable *modeled* FX rollover cash flows, never implicit MT5 swap-unit conversion.

Use broker-normalized deposit-currency cash amounts only when explicitly supplied.
The fallback is a configurable proxy (not authenticated broker financing).
All times are UTC; broker server rollover clocks require independent certification.
"""
from __future__ import annotations
from datetime import datetime, timedelta, time, timezone
import math

from quant_rl.config.schema import ExecutionConfig
from quant_rl.data.mt5_source import SymbolSpec
from quant_rl.contracts import TradeDirection


def financing_terms(spec: SymbolSpec, cfg: ExecutionConfig, direction: TradeDirection):
    long_cash = spec.financing_long_account_ccy_per_lot_rollover
    short_cash = spec.financing_short_account_ccy_per_lot_rollover
    if (long_cash is None) != (short_cash is None):
        raise ValueError('Broker-normalized long/short financing must both be provided')
    if long_cash is not None:
        if not all(math.isfinite(float(v)) for v in (long_cash, short_cash)):
            raise ValueError('Nonfinite broker-normalized financing')
        cash = long_cash if direction == TradeDirection.BUY else short_cash
        source = 'SYMBOL_NORMALIZED_UNVERIFIED'
    else:
        cash = (cfg.financing_long_cash_per_lot_rollover if direction == TradeDirection.BUY
                else cfg.financing_short_cash_per_lot_rollover)
        source = 'MODELED_CONFIG'
    hour = spec.financing_rollover_utc_hour if spec.financing_rollover_utc_hour is not None else cfg.financing_rollover_utc_hour
    triple = (spec.financing_triple_weekday_utc if spec.financing_triple_weekday_utc is not None
              else cfg.financing_triple_weekday_utc)
    if hour not in range(24) or triple not in range(5):
        raise ValueError('Invalid explicitly configured UTC rollover clock/week day')
    return float(cash), int(hour), int(triple), source


def financing_cost(
    entry_time_s: int, mark_time_s: int, volume_lots: float,
    direction: TradeDirection, spec: SymbolSpec, cfg: ExecutionConfig,
    *, cost_multiplier: float = 1.0,
) -> tuple[float, int, str]:
    """Return (cost in account currency, weighted rollover units, evidence label).

    Positive cost is a debit, negative cost a credit. Triple-rollover day
    incorporates weekend carry; Saturday/Sunday are not charged separately.
    The account's reward normalization cannot alter the cash result.
    """
    cash_per_lot, hour, triple, source = financing_terms(spec, cfg, direction)
    if (mark_time_s < entry_time_s or volume_lots <= 0 or
            not math.isfinite(volume_lots) or not math.isfinite(cost_multiplier) or cost_multiplier < 1):
        raise ValueError('Invalid rollover valuation window, volume or stress multiplier')
    start = datetime.fromtimestamp(int(entry_time_s), timezone.utc).date()
    end = datetime.fromtimestamp(int(mark_time_s), timezone.utc).date()
    if (end - start).days > 36525:
        raise ValueError('Unsupported rollover history duration')
    units = 0
    for day_num in range((end - start).days + 1):
        day = start + timedelta(days=day_num)
        if day.weekday() >= 5:
            continue
        moment = datetime.combine(day, time(hour=hour), tzinfo=timezone.utc).timestamp()
        if entry_time_s < moment <= mark_time_s:
            units += 3 if day.weekday() == triple else 1
    # Stress must not make a positive rollover credit artificially more favorable.
    raw_cash = units * cash_per_lot * volume_lots
    cost = -raw_cash * (cost_multiplier if raw_cash < 0 else 1.0 / cost_multiplier)
    return float(cost), units, source
