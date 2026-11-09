"""Broker-accurate position sizing and quote-to-account currency conversion."""

from __future__ import annotations

from dataclasses import dataclass
import math

from quant_rl.data.mt5_source import SymbolSpec, split_fx_symbol


def conversion_rates_asof(latest_mid: dict[str, float]) -> dict[str, float]:
    """Resolve known FX currency conversions from point-in-time midquotes.

    This is a single deterministic shortest-leg graph for sizing and portfolio
    valuation. Missing paths are never assumed to equal one.
    """
    from collections import defaultdict, deque

    edges: dict[str, dict[str, float]] = defaultdict(dict)
    direct: dict[str, float] = {}
    for symbol, price in sorted(latest_mid.items()):
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Invalid current FX price for {symbol}")
        base, quote, _ = split_fx_symbol(symbol)
        edges[base][quote] = float(price)
        edges[quote][base] = 1.0 / float(price)
        direct[base + quote] = float(price)

    rates = dict(direct)
    for source in sorted(edges):
        queue = deque([(source, 1.0)])
        seen = {source}
        while queue:
            currency, rate = queue.popleft()
            rates.setdefault(source + currency, rate)
            for nxt, factor in sorted(edges[currency].items()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, rate * factor))
    return rates


@dataclass(frozen=True)
class PositionSizeQuote:
    """Computed lot size, actual monetary risk, and margin requirement."""

    volume_lots: float
    risk_budget_account_ccy: float
    actual_risk_account_ccy: float
    pip_value_per_lot_account_ccy: float
    required_margin_account_ccy: float
    stop_pips: float
    valid: bool
    reason: str = ""
    raw_requested_lots: float = 0.0


def compute_pip_value_per_lot(
    symbol_spec: SymbolSpec,
    entry_price: float,
    account_currency: str = "USD",
    fx_rates: dict[str, float] | None = None,
) -> float:
    """
    Compute 1-pip value per 1 standard lot in `account_currency`.
    Handles direct pairs (XXX/USD), indirect pairs (USD/XXX), and cross pairs (XXX/YYY).
    """
    if entry_price <= 0:
        return 0.0
    base, quote, _ = split_fx_symbol(symbol_spec.symbol)
    acct = account_currency.upper()
    pip_quote_ccy = symbol_spec.contract_size * symbol_spec.pip_size

    if quote == acct:
        return pip_quote_ccy
    if base == acct:
        return pip_quote_ccy / entry_price

    rates = fx_rates or {}
    pair_direct = f"{quote}{acct}"
    pair_inverse = f"{acct}{quote}"
    if pair_direct in rates and rates[pair_direct] > 0:
        return pip_quote_ccy * rates[pair_direct]
    if pair_inverse in rates and rates[pair_inverse] > 0:
        return pip_quote_ccy / rates[pair_inverse]

    raise ValueError(
        f"Missing point-in-time {quote}/{acct} currency conversion for {symbol_spec.symbol}"
    )


def compute_margin_per_lot(
    symbol_spec: SymbolSpec,
    entry_price: float,
    account_leverage: float,
    account_currency: str = "USD",
    fx_rates: dict[str, float] | None = None,
) -> float:
    """Compute required margin per 1.0 lot in `account_currency`."""
    if account_leverage <= 0 or entry_price <= 0:
        return float("inf")
    base, quote, _ = split_fx_symbol(symbol_spec.symbol)
    acct = account_currency.upper()
    notional_base = symbol_spec.contract_size

    if base == acct:
        notional_acct = notional_base
    elif quote == acct:
        notional_acct = notional_base * entry_price
    else:
        rates = fx_rates or {}
        if f"{base}{acct}" in rates:
            notional_acct = notional_base * rates[f"{base}{acct}"]
        elif f"{acct}{base}" in rates and rates[f"{acct}{base}"] > 0:
            notional_acct = notional_base / rates[f"{acct}{base}"]
        else:
            raise ValueError(
                f"Missing point-in-time {base}/{acct} currency conversion for margin on {symbol_spec.symbol}"
            )

    return notional_acct / account_leverage


def calculate_broker_position_size(
    *,
    equity: float,
    requested_risk_fraction: float,
    max_risk_fraction: float,
    stop_distance_price: float,
    entry_price: float,
    symbol_spec: SymbolSpec,
    account_leverage: float = 30.0,
    account_currency: str = "USD",
    fx_rates: dict[str, float] | None = None,
) -> PositionSizeQuote:
    """
    Calculate broker-valid lot size stepped down (`math.floor`) to `volume_step`
    so actual monetary risk never exceeds `min(requested_risk_fraction, max_risk_fraction) * equity`.
    """
    if equity <= 0 or stop_distance_price <= 0 or entry_price <= 0:
        return PositionSizeQuote(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "Non-positive equity, stop, or price")

    capped_fraction = max(0.0, min(float(requested_risk_fraction), float(max_risk_fraction)))
    risk_budget = equity * capped_fraction
    stop_pips = stop_distance_price / symbol_spec.pip_size
    if stop_pips <= 0:
        return PositionSizeQuote(0.0, risk_budget, 0.0, 0.0, 0.0, 0.0, False, "Zero stop pips")

    pip_val = compute_pip_value_per_lot(symbol_spec, entry_price, account_currency, fx_rates)
    risk_per_lot = stop_pips * pip_val
    if risk_per_lot <= 0:
        return PositionSizeQuote(0.0, risk_budget, 0.0, pip_val, 0.0, stop_pips, False, "Invalid pip value")

    raw_lots = risk_budget / risk_per_lot
    step = symbol_spec.volume_step
    floored_lots = math.floor((raw_lots + 1e-12) / step) * step
    floored_lots = min(floored_lots, symbol_spec.volume_max)
    floored_lots = math.floor((floored_lots + 1e-12) / step) * step
    # Decimal string quantization preserves legitimate finer broker lot increments.
    floored_lots = float(round(floored_lots, max(0, len(f"{step:.12f}".rstrip("0").split(".")[-1]))))

    if floored_lots < symbol_spec.volume_min - 1e-9:
        return PositionSizeQuote(
            volume_lots=0.0,
            risk_budget_account_ccy=risk_budget,
            actual_risk_account_ccy=symbol_spec.volume_min * risk_per_lot,
            pip_value_per_lot_account_ccy=pip_val,
            required_margin_account_ccy=0.0,
            stop_pips=stop_pips,
            valid=False,
            reason=f"Floored volume {floored_lots:.4f} < broker volume_min {symbol_spec.volume_min:.4f}",
            raw_requested_lots=raw_lots,
        )

    actual_risk = floored_lots * risk_per_lot
    margin_per_lot = compute_margin_per_lot(
        symbol_spec, entry_price, account_leverage, account_currency, fx_rates
    )
    req_margin = floored_lots * margin_per_lot
    return PositionSizeQuote(
        volume_lots=floored_lots,
        risk_budget_account_ccy=risk_budget,
        actual_risk_account_ccy=actual_risk,
        pip_value_per_lot_account_ccy=pip_val,
        required_margin_account_ccy=req_margin,
        stop_pips=stop_pips,
        valid=True,
        reason="OK",
        raw_requested_lots=raw_lots,
    )
