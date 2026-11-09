"""Data acquisition, UTC time alignment, and historical verification.

Exports are resolved lazily so lightweight schema modules can import the
canonical timeframe map without importing MT5 history and creating a cycle.
"""
from importlib import import_module

_EXPORTS = {
    "DataCoverageManifest": ("quant_rl.data.quality", "DataCoverageManifest"),
    "DataQualityAuditor": ("quant_rl.data.quality", "DataQualityAuditor"),
    "FrozenDataset": ("quant_rl.data.mt5_history", "FrozenDataset"),
    "MT5History": ("quant_rl.data.mt5_history", "MT5History"),
    "MT5DataSource": ("quant_rl.data.mt5_source", "MT5DataSource"),
    "OHLCVBar": ("quant_rl.data.mt5_source", "OHLCVBar"),
    "SymbolSpec": ("quant_rl.data.mt5_source", "SymbolSpec"),
    "pip_size_for_symbol": ("quant_rl.data.mt5_source", "pip_size_for_symbol"),
    "split_fx_symbol": ("quant_rl.data.mt5_source", "split_fx_symbol"),
    "TIMEFRAME_SECONDS": ("quant_rl.data.time_alignment", "TIMEFRAME_SECONDS"),
    "align_completed_bars_asof": ("quant_rl.data.time_alignment", "align_completed_bars_asof"),
    "resample_ohlcv_utc": ("quant_rl.data.time_alignment", "resample_ohlcv_utc"),
    "timeframe_to_seconds": ("quant_rl.data.time_alignment", "timeframe_to_seconds"),
}

__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
