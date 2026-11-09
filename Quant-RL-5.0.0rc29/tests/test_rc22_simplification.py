"""Simplification must not reintroduce duplicate stores or weaken guardrails."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_rl.config.loader import ConfigLoader
from quant_rl.config.schema import QuantRLConfig
from quant_rl.data.mt5_history import MT5History
from quant_rl.data.mt5_source import OHLCVBar
from quant_rl.execution.examiner import PreTradeExaminer
from tests._specs import default_symbol_spec


def test_freeze_keeps_verified_dataset_in_ram_without_creating_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = MT5History()
    bars = [OHLCVBar(1704067200 + i * 3600, 1.10, 1.12, 1.08, 1.11, 100, 3)
            for i in range(4)]
    frozen = store.freeze('EURUSD', 'H1', bars, default_symbol_spec('EURUSD'),
                          broker_server='MetaQuotes-Demo')
    assert frozen.fingerprint
    assert frozen.manifest.data_origin == 'MT5_TERMINAL'
    assert not frozen.time_s.flags.writeable
    assert not tuple(tmp_path.rglob('*.npz'))
    assert not tuple(tmp_path.rglob('*.manifest.json'))


def test_no_removed_configuration_is_silently_accepted():
    for patch in ({'data': {'storage_dir': './storage'}},
                  {'execution': {'market_correlation_tightening_factor': .8}},
                  {'execution': {'market_portfolio_correlation_trigger': .9}}):
        with pytest.raises(ValidationError):
            QuantRLConfig.model_validate(patch)


def test_operator_yaml_keeps_intended_training_limits_and_mt5_login():
    cfg = ConfigLoader.load(Path(__file__).parents[1] / 'config.yaml')
    assert cfg.ppo.episodes == cfg.ppo.ppo_epochs == cfg.optimization.trials == 1000
    assert cfg.research.lookback_years == 4
    assert cfg.execution.max_open_positions == 1
    assert cfg.broker_connection.mt5_login == 0
    assert len(cfg.research.symbols) == 7


def test_mandatory_examiner_check_identifiers_remain_intact():
    cfg = QuantRLConfig()
    examiner = PreTradeExaminer(cfg.execution)
    assert examiner is not None
    # The deleted correlation extension never implemented any of C1-C26.
    from quant_rl.execution import examiner as mod
    source = Path(mod.__file__).read_text('utf-8')
    assert all(f'"C{i}"' in source for i in range(1, 27))
