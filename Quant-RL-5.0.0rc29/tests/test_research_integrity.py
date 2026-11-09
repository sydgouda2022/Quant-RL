"""Current-release domain invariants: no hidden data/source fallback or test-set selection."""
from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from pydantic import ValidationError

from quant_rl.config.schema import QuantRLConfig
from quant_rl.research.evolution import chronological_split, ModelPopulation
from quant_rl.routing import choose_mode
from quant_rl.intelligence.policy import PPOTrader, PPOSettings
from tests._specs import default_symbol_spec
from tests.test_auto_mt5_routing import FakeAPI, FakeSource, FakeStore, FakePopulation, make_cfg


def test_h1_native_acquisition_rejects_sub_hour_primary():
    for tf in ('M1','M5','M15','M30'):
        with pytest.raises(ValidationError,match='H1 or coarser'):
            QuantRLConfig.model_validate({'research':{'timeframe':tf}})


def test_observation_timeframes_validate_at_config_load():
    for tfs in ([],['H1','H1'],['M5','H4'],['FAKE']):
        with pytest.raises(ValidationError):
            QuantRLConfig.model_validate({'data':{'observation_timeframes':tfs}})


def test_calendar_split_uses_elapsed_time_instead_of_row_count():
    # First half contains many more observations than the last half.
    timestamps=np.array([*range(0,200,1),*range(200,400,4)],dtype=np.int64)
    train,validation,test=chronological_split(timestamps,10,window_bounds=(0,400))
    assert timestamps[train[-1]]<200<=timestamps[validation[0]]
    assert timestamps[validation[-1]]<300<=timestamps[test[0]]
    assert len(train)>len(validation)


def test_production_mode_decision_does_not_load_entire_historical_population(tmp_path):
    import time
    now=int(time.time())
    cfg=make_cfg(tmp_path)
    api=FakeAPI(now,supported={'EURUSD','GBPUSD'})
    source=FakeSource(api)
    store=FakeStore()
    decision,datasets,rejected=choose_mode(cfg,source_factory=lambda:source,
                                            history=store,population=FakePopulation(),now_s=now)
    assert decision.mode=='RESEARCH'
    assert decision.targets==('EURUSD','GBPUSD')
    assert store.read==[] and datasets=={} and rejected=={}
    assert source.closed


def test_checkpoint_restores_training_episode_and_adam_state(tmp_path,monkeypatch):
    class Features:
        n_features=10
        def snapshot(self):return {'mean':torch.zeros(2),'std':torch.ones(2)}
        def restore(self,snap):pass
    features=Features()
    settings=PPOSettings(episodes=3,ppo_epochs=1,validation_interval_episodes=1)
    spec=default_symbol_spec('EURUSD')
    cfg=QuantRLConfig()
    path=tmp_path/'resume.pt'
    visited=[]
    def replay(self,*args,**kwargs):
        visited.append('replay')
        return SimpleNamespace(learning_reward_account_ccy=42.),[]
    monkeypatch.setattr(PPOTrader,'replay',replay)
    first=PPOTrader(features,settings,123)
    def interrupt(episode,score,validation):
        if episode==0:raise KeyboardInterrupt('test interruption')
    with pytest.raises(KeyboardInterrupt):
        first.fit(None,spec,np.array([0]),np.array([1]),cfg,123,callback=interrupt,checkpoint_path=path)
    assert path.is_file()
    raw=torch.load(path,weights_only=True)
    assert raw['next_episode']==1 and 'optimizer' in raw and 'rng_numpy' in raw
    visited.clear()
    recovered=PPOTrader(features,settings,123)
    assert recovered.fit(None,spec,np.array([0]),np.array([1]),cfg,123,checkpoint_path=path)==42.
    assert len(visited)==4  # Two remaining training rollouts + two validation evaluations
    assert not path.exists()


def test_population_promotion_does_not_delete_old_immutable_models(tmp_path):
    pop=ModelPopulation(tmp_path)
    first={'weights':{'a':torch.ones(2)},'mean':torch.zeros(1),'std':torch.ones(1)}
    second={'weights':{'a':torch.zeros(2)},'mean':torch.zeros(1),'std':torch.ones(1)}
    pop.publish({'EURUSD':(first,{'arch':'test','current_score':.1})},{},'EURUSD',{})
    a=pop.models/pop.data['symbols']['EURUSD']['file']
    pop.publish({'EURUSD':(second,{'arch':'test','current_score':.2})},{},'EURUSD',{})
    b=pop.models/pop.data['symbols']['EURUSD']['file']
    assert a!=b and a.is_file() and b.is_file()


def test_live_model_certificate_blocks_expired_and_cross_broker_reuse():
    """Broker server, economic policy and evaluation recency are LIVE gates."""
    from copy import deepcopy
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from quant_rl.routing import _existing_model
    from quant_rl.config.schema import QuantRLConfig
    cfg=QuantRLConfig()
    now=1_800_000_000
    record={'origin':'MT5_TERMINAL','account_currency':'USD',
            'mt5_server_timezone':cfg.broker_connection.mt5_server_timezone, 'policy_hash':cfg.policy_hash,
            'broker_server':'MetaQuotes-Demo','current_score':3.0,
            'period':{'test_end_utc':datetime.fromtimestamp(now-3600,timezone.utc).isoformat()},
            'arch':'sha'}
    class Population:
        def __init__(self, rec): self.data={'symbols':{'EURUSD':rec}}
        def verify_artifact(self,rec):return None
        def load(self, rec, arch):
            assert arch=='sha'
            return {}
    def eligible(rec,server='MetaQuotes-Demo'):
        return _existing_model(Population(rec),'EURUSD','USD',None,
                               cfg=cfg,broker_server=server,now_s=now)
    assert eligible(record)
    assert not eligible(record,'OtherBroker-Live')
    outdated=deepcopy(record)
    outdated['period']['test_end_utc']=datetime.fromtimestamp(
        now-(cfg.qualification.max_live_model_age_days+1)*86400,timezone.utc).isoformat()
    assert not eligible(outdated)
    missing_policy=deepcopy(record)
    missing_policy.pop('policy_hash')
    assert not eligible(missing_policy)
    bad_clock=deepcopy(record)
    bad_clock['mt5_server_timezone']='Europe/Athens'
    assert not eligible(bad_clock)
