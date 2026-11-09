"""Broker time-alignment safeguards. MT5 is mocked; no actual broker orders."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from quant_rl.data.mt5_source import clock_offset_diagnostic,broker_tick_age
from quant_rl.research.evolution import ModelPopulation
from quant_rl.routing import _existing_model
from quant_rl.execution.live import LiveSession
from quant_rl.config.schema import QuantRLConfig


def test_raw_future_quote_diagnostic_is_observation_only():
    now=1_800_000_000
    txt=clock_offset_diagnostic(now,now+3848)
    assert 'age=-3848s' in txt
    assert 'no broker offset is applied' in txt
    assert broker_tick_age(now,now+3848)==-3848
    assert broker_tick_age(now,now-3600)==3600


def test_clock_diagnostic_rejects_extreme_and_missing_data():
    assert 'no MT5 UTC timestamp' in clock_offset_diagnostic(1000,0)
    assert 'age=-46800s' in clock_offset_diagnostic(1_000,1_000+13*3600)


def _pop(tmp_path):
    pop=ModelPopulation(tmp_path)
    snap={'weights':{'x':torch.tensor([1.])},'mean':torch.zeros(1),'std':torch.ones(1)}
    rec={'arch':'test','origin':'MT5_TERMINAL','account_currency':'USD','current_score':.04,'mt5_server_timezone':'UTC'}
    pop.publish({'EURUSD':(snap,rec)}, {}, 'EURUSD', {})
    return pop


def test_model_reused_only_with_matching_server_timezone(tmp_path):
    pop=_pop(tmp_path)
    assert _existing_model(pop,'EURUSD','USD','UTC')
    assert not _existing_model(pop,'EURUSD','USD','Europe/Athens')
    assert ModelPopulation(tmp_path).data['champion']=='EURUSD'
    # Do not overwrite the existing model or wipe the manifest merely because
    # a user is evaluating whether an MT5 broker offset might be present.
    assert (tmp_path/'best_score.json').exists()


def test_publication_never_reuses_different_server_timezone(tmp_path):
    _pop(tmp_path)
    pop=ModelPopulation(tmp_path)
    # This is the final publication step under a different server timezone.
    pop.publish({}, {}, None, {'EURUSD':{'status':'NO_QUALIFIED_MODEL'}})
    reloaded=ModelPopulation(tmp_path)
    assert 'EURUSD' in reloaded.data['symbols']  # retained as audit-only under +2h
    assert not _existing_model(reloaded,'EURUSD','USD','Europe/Athens')
    assert _existing_model(reloaded,'EURUSD','USD','UTC')
    assert reloaded.data['champion'] is None


def test_live_constructor_blocks_unverified_timezone_or_model_basis(tmp_path):
    _pop(tmp_path)
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path),
              'broker_connection':{'mt5_server_timezone':'Europe/Athens'}})
    broker=NS(account=lambda:NS(login=1234,server='MetaQuotes-Demo',currency='USD',trade_mode=0))
    with pytest.raises(ValueError,match='different MT5 server timezone'):
        LiveSession(cfg,broker)
