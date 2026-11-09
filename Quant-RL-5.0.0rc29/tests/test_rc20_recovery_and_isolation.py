"""Recovery and isolation tests: no terminal, broker orders or generated LIVE data."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from quant_rl.intelligence.policy import PPOTrader, PPOSettings
from quant_rl.research.evolution import exclusive_run
from quant_rl.config.schema import QuantRLConfig
from tests._specs import default_symbol_spec


def test_os_lock_released_after_subprocess_crash(tmp_path):
    root=tmp_path/'process_lock'
    script='''import os
from pathlib import Path
from quant_rl.research.evolution import exclusive_run
with exclusive_run(Path(__import__('sys').argv[1])):
    os._exit(0)
'''
    result=subprocess.run([sys.executable,'-c',script,str(root)],cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True, timeout=30)
    assert result.returncode==0, result.stderr
    # No manual cleanup required; the kernel releases a dead process's lock.
    with exclusive_run(root):
        with pytest.raises(RuntimeError,match='Concurrent run lock'):
            with exclusive_run(root):
                pass


def test_trader_does_not_change_process_torch_rng_on_init_or_rollout(monkeypatch):
    class Features:
        n_features=10
        def snapshot(self):return {'mean':torch.zeros(1),'std':torch.ones(1)}
        def restore(self,snapshot):pass
    before=torch.get_rng_state().clone()
    trader=PPOTrader(Features(),PPOSettings(episodes=1,ppo_epochs=1),1234)
    assert torch.equal(before,torch.get_rng_state())
    class FakeSimulation:
        decision_events=[]
    import quant_rl.execution.accounting as acct
    def simulate(self,obs,decide,spec,**kw):
        # Consume randomized rollout values, then verify the outer RNG was not touched.
        torch.rand(5)
        return FakeSimulation()
    monkeypatch.setattr(acct.PortfolioAccountingSimulator,'simulate',simulate)
    monkeypatch.setattr(acct,'CausalMarketContextCache',lambda *args: NS(compatible=lambda *args:True))
    indices=np.array([0])
    # obs and cfg passed only to the stubbed account engine and cache.
    trader.replay(NS(),default_symbol_spec('EURUSD'),indices,QuantRLConfig(),deterministic=False)
    assert torch.equal(before,torch.get_rng_state())
    assert not torch.equal(before,trader._torch_rng_state)


def test_optuna_completed_winner_recovers_without_retraining_prior_trial(tmp_path,monkeypatch):
    import quant_rl.research.optimizer as optimizer
    class Features:
        def __init__(self,obs,train,balance,**kwargs):
            self.n_features=10
            self.feature_group=kwargs['feature_group']
            self.sequence_bars=kwargs['sequence_bars']
    class Trader:
        calls=0
        def __init__(self,features,settings,seed):
            self.features=features
            self.settings=settings
            self.value=None
        def fit(self,*a,**kw):
            self.__class__.calls+=1
            self.value=10.0 if self.__class__.calls==1 else -7.0
            return self.value
        def replay(self,*a,**kw):
            return NS(ending_equity=10000.0,decision_events=[],executed_trades=[],
                      open_positions_at_end=[],penalty_total_account_ccy=0,
                      terminal_halt_reason=None,terminal_halt_bar_index=None),[]
        def snapshot(self):
            return {'weights':{'dummy':torch.tensor([self.value])},
                    'mean':torch.zeros(1),'std':torch.ones(1)}
        def restore(self,snap):self.value=float(snap['weights']['dummy'][0])
    monkeypatch.setattr(optimizer,'EconomicFeatures',Features)
    monkeypatch.setattr(optimizer,'PPOTrader',Trader)
    cfg=QuantRLConfig.model_validate({'project_root':str(tmp_path),'optimization':{
        'trials':1,'enabled':True,'max_session_seconds':60}, 'ppo':{'episodes':1,'ppo_epochs':1}})
    obs=NS(sequence_tensor=np.zeros((10,12,5)))
    path=tmp_path/'study.db'
    args=(cfg,obs,default_symbol_spec('EURUSD'),np.array([0]),np.array([1]),None)
    first=optimizer.optimize_candidate(*args,study_path=path,study_key='test-current-window')
    assert first.validation_reward_account_ccy==10
    second=optimizer.optimize_candidate(*args,study_path=path,study_key='test-current-window')
    assert second.validation_reward_account_ccy==10
    assert second.trader.value==10
    assert Trader.calls==2
    files=list(tmp_path.glob('study.candidate.*.pt'))
    assert files, 'Completed Optuna trials need recoverable tensor snapshots'


def test_uncommitted_optuna_candidate_never_evicts_last_winner(tmp_path,monkeypatch):
    import quant_rl.research.optimizer as optimizer
    class Features:
        def __init__(self,obs,train,balance,**kw):
            self.n_features=10;self.feature_group=kw['feature_group'];self.sequence_bars=kw['sequence_bars']
    class Trader:
        calls=0
        def __init__(self,features,settings,seed):self.features=features;self.settings=settings
        def fit(self,*args,**kwargs):
            type(self).calls+=1
            return 25.0 if type(self).calls==1 else -1.
        def replay(self,*args,**kwargs):
            return NS(ending_equity=10000.,decision_events=[],executed_trades=[],
                      open_positions_at_end=[],penalty_total_account_ccy=0.,
                      terminal_halt_reason=None,terminal_halt_bar_index=None),[]
        def snapshot(self):return {'weights':{'a':torch.tensor([25.])},'mean':torch.zeros(1),'std':torch.ones(1)}
        def restore(self,snap):pass
    monkeypatch.setattr(optimizer,'EconomicFeatures',Features)
    monkeypatch.setattr(optimizer,'PPOTrader',Trader)
    cfg=QuantRLConfig.model_validate({'optimization':{'trials':1,'max_session_seconds':60},'ppo':{'episodes':1,'ppo_epochs':1}})
    obs=NS(sequence_tensor=np.zeros((3,8,3)))
    study=tmp_path/'test.db'
    run=lambda: optimizer.optimize_candidate(cfg,obs,default_symbol_spec('EURUSD'),np.array([0]),np.array([1]),None,study_path=study,study_key='frozen')
    result=run()
    assert result.validation_reward_account_ccy==25
    existing=list(tmp_path.glob('test.candidate.*.pt'))
    assert len(existing)==1
    # A fake artifact naming an uncommitted trial must not supersede COMPLETE.
    malicious=tmp_path/'test.candidate.99.pt'
    outer=torch.load(existing[0],weights_only=True)
    document=torch.load(io.BytesIO(outer['bytes']),weights_only=True)
    document['trial_number']=99
    document['validation_reward']=1e12
    payload=io.BytesIO();torch.save(document,payload)
    import hashlib
    torch.save({'sha256':hashlib.sha256(payload.getvalue()).hexdigest(),'bytes':payload.getvalue()},malicious)
    assert run().validation_reward_account_ccy==25


def test_logger_resume_rotates_all_full_segments(tmp_path,capsys):
    from quant_rl.config.logger import AuditLogger
    logger=AuditLogger()
    logger.max_file_bytes=200
    logger.configure(tmp_path)
    for i in range(15):
        logger.info('TEST',f'sequence={i:02d} extra log evidence')
    logger.shutdown()
    prior=list((tmp_path/'logs').glob('*.log'))
    assert len(prior)>2
    assert any('_001.log' in p.name for p in prior)
    restart=AuditLogger()
    restart.max_file_bytes=200
    restart.configure(tmp_path)
    restart.info('TEST','new process logger instance; previous segments must remain bounded')
    restart.shutdown()
    after=list((tmp_path/'logs').glob('*.log'))
    assert len(after)>=len(prior)
    assert max(p.stat().st_size for p in after)<500


def test_optuna_top_k_candidates_are_persisted_and_recovered_in_rank_order(tmp_path,monkeypatch):
    import quant_rl.research.optimizer as optimizer
    class Features:
        def __init__(self,obs,train,balance,**kw):
            self.n_features=10
            self.feature_group=kw['feature_group']
            self.sequence_bars=kw['sequence_bars']
    class Trader:
        calls=0
        values=[10.,30.,20.,-10.,-20.,-30.]
        def __init__(self,features,settings,seed):
            self.features=features;self.settings=settings;self.value=None
        def fit(self,*args,**kwargs):
            self.value=self.values[type(self).calls]
            type(self).calls+=1
            return self.value
        def replay(self,*args,**kwargs):
            return NS(ending_equity=10000.,decision_events=[],executed_trades=[],
                      open_positions_at_end=[],penalty_total_account_ccy=0.,
                      terminal_halt_reason=None,terminal_halt_bar_index=None),[]
        def snapshot(self):
            return {'weights':{'dummy':torch.tensor([self.value])},
                    'mean':torch.zeros(1),'std':torch.ones(1)}
        def restore(self,snap):self.value=float(snap['weights']['dummy'][0])
    monkeypatch.setattr(optimizer,'EconomicFeatures',Features)
    monkeypatch.setattr(optimizer,'PPOTrader',Trader)
    cfg=QuantRLConfig.model_validate({'optimization':{'trials':3,'enabled':True,
        'max_session_seconds':60},'ppo':{'episodes':1,'ppo_epochs':1}})
    obs=NS(sequence_tensor=np.zeros((10,12,5)))
    study=tmp_path/'topk.db'
    args=(cfg,obs,default_symbol_spec('EURUSD'),np.array([0]),np.array([1]),None)
    first=optimizer.optimize_candidate(*args,study_path=study,study_key='top-k',top_k=2)
    assert [x.validation_reward_account_ccy for x in first]==[30.,20.]
    assert [x.trader.value for x in first]==[30.,20.]
    assert len(list(tmp_path.glob('topk.candidate.*.pt')))==2
    second=optimizer.optimize_candidate(*args,study_path=study,study_key='top-k',top_k=2)
    assert [x.validation_reward_account_ccy for x in second]==[30.,20.]
    assert [x.trader.value for x in second]==[30.,20.]
    assert len(list(tmp_path.glob('topk.candidate.*.pt')))==2


def test_default_abstention_reference_volume_is_one_hundredth_lot():
    cfg=QuantRLConfig.model_validate({})
    assert cfg.ppo.penalty_reference_lots==0.01
