from dataclasses import replace
from pathlib import Path

from quant_rl.config.schema import VALID_TIMEFRAMES
from quant_rl.data.time_alignment import TIMEFRAME_SECONDS
from quant_rl.data.mt5_source import broker_spec_hash
from quant_rl.contracts import ActionProposal, TradeDirection
from tests._specs import default_symbol_spec


def test_config_timeframes_have_one_canonical_source():
    assert VALID_TIMEFRAMES == tuple(TIMEFRAME_SECONDS)


def test_broker_spec_certificate_ignores_transient_trade_permission():
    spec = default_symbol_spec("EURUSD")
    assert broker_spec_hash(spec) == broker_spec_hash(replace(spec, trade_mode=0))


def test_shared_action_contracts_are_available_from_package_contracts():
    assert TradeDirection.NO_TRADE.value == "NO_TRADE"
    assert ActionProposal.__dataclass_params__.frozen


def test_obsolete_duplicate_modules_are_removed():
    root = Path(__file__).resolve().parents[1] / "quant_rl"
    assert not (root / "intelligence" / "action_space.py").exists()
    assert not (root / "data" / "certificates.py").exists()
    assert not (root / "research" / "evaluation.py").exists()
