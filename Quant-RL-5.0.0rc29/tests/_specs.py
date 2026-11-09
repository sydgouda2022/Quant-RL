"""Explicit test-only MT5-format specifications; not a production data source."""
from quant_rl.data.mt5_source import SymbolSpec,split_fx_symbol

def default_symbol_spec(symbol: str, reference_price: float | None = None, account_currency: str = "USD") -> SymbolSpec:
    """Construct realistic spot FX specification with exact deposit-currency tick value."""
    base, quote, _ = split_fx_symbol(symbol)
    is_jpy = quote == "JPY"
    point = 0.001 if is_jpy else 0.00001
    contract_size = 100_000.0
    default_prices = {
        "EURUSD": 1.0850,
        "GBPUSD": 1.2700,
        "USDJPY": 150.00,
        "USDCHF": 0.8850,
        "USDCAD": 1.3600,
        "AUDUSD": 0.6550,
        "NZDUSD": 0.6050,
    }
    px = reference_price if (reference_price and reference_price > 0) else default_prices.get(base + quote, 1.0)
    acc = account_currency.upper()
    if quote == acc:
        tick_value = contract_size * point
    elif base == acc:
        tick_value = (contract_size * point) / max(px, 1e-8)
    else:
        tick_value = contract_size * point
    return SymbolSpec(
        name=symbol,
        base_currency=base,
        quote_currency=quote,
        point=point,
        trade_tick_size=point,
        trade_tick_value=float(tick_value),
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        stops_level_points=20.0,
        freeze_level_points=0.0,
        contract_size=contract_size,
        trade_mode=4,
        swap_long_per_lot_per_day=-4.5,
        swap_short_per_lot_per_day=-4.5,
    )

