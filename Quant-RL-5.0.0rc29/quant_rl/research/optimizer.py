"""One Optuna study optimizing PPO learning, judged by account equity on validation only."""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import os
import json
import hashlib
import uuid
import time
import io
import math
import torch
from quant_rl.execution.accounting import summarize_veto_context
from quant_rl.config.logger import log
import optuna
from quant_rl.intelligence.policy import PPOSettings, PPOTrader, EconomicFeatures


@dataclass
class OptimizedCandidate:
    trader: PPOTrader
    settings: PPOSettings
    validation_reward_account_ccy: float
    trial_report: list[dict]
    seed_from: str
    features: EconomicFeatures
    trial_number: int | None = None


def trial_decision_diagnostics(sim):
    """Explain validation decisions without altering economic objective or safety rules."""
    events=sim.decision_events
    failed=Counter(code for event in events if event.status=='VETO' for code in event.veto_codes)
    vetoed=[event for event in events if event.status=='VETO']
    per_check=summarize_veto_context(events)
    return {
        'eligible_decisions':len(events),
        'executed_proposals':sum(event.action != 'NO_TRADE' and event.status in ('CLOSED','OPEN') for event in events),
        'no_trade_decisions':sum(event.status=='ABSTAIN' for event in events),
        'vetoed_proposals':len(vetoed),
        'veto_fraction':len(vetoed)/len(events) if events else 0.0,
        'all_failed_check_counts':dict(sorted(failed.items())),
        'veto_context_by_check':per_check,
        'average_veto_stop_units':sum(event.proposed_stop_units for event in vetoed)/len(vetoed) if vetoed else None,
        'average_veto_target_units':sum(event.proposed_target_units for event in vetoed)/len(vetoed) if vetoed else None,
        'completed_trades':len(sim.executed_trades),
        'terminal_censored_positions':len(sim.open_positions_at_end),
        'terminal_censored_learning_mark':sum(
            e.terminal_mark_to_market_account_ccy or 0. for e in events),
        'learning_penalty_account_ccy':sim.penalty_total_account_ccy,
        'terminal_halt_reason':sim.terminal_halt_reason,
        'terminal_halt_bar_index':sim.terminal_halt_bar_index,
    }


def optimize_candidate(cfg,obs,spec,train,val,m1,seed_lookup=None,study_path=None,study_key=None,deadline_monotonic=None,top_k=1):
    """Optimize on search validation and retain the top-K recoverable snapshots.

    The default remains the historical single-candidate return contract. When
    top_k > 1, return a descending-score list for promotion adjudication.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials=[]
    best=[]
    deadline=(time.monotonic()+cfg.optimization.max_session_seconds
              if deadline_monotonic is None else deadline_monotonic)
    # SQLite retains completed trial values, but values alone cannot recover
    # their neural weights. Keep a content-verified, study-specific winner.
    candidate_prefix=(Path(study_path).with_suffix('') if study_path is not None else None)

    def persist_candidate(trial_number, candidate):
        if candidate_prefix is None:
            return
        candidate_path=candidate_prefix.with_name(candidate_prefix.name+f'.candidate.{trial_number}.pt')
        profile={'feature_group':candidate.features.feature_group,
                 'sequence_bars':candidate.features.sequence_bars}
        document={'format':'optuna_winner_v1', 'study_key':study_key or Path(study_path).stem,
                  'trial_number':trial_number,
                  'validation_reward':float(candidate.validation_reward_account_ccy),
                  'settings':candidate.settings.__dict__, 'feature_profile':profile,
                  'feature_count':candidate.features.n_features,
                  'seed_from':candidate.seed_from,
                  'snapshot':candidate.trader.snapshot()}
        stream=io.BytesIO()
        torch.save(document,stream)
        raw=stream.getvalue()
        payload={'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':raw}
        candidate_path.parent.mkdir(parents=True,exist_ok=True)
        temp=candidate_path.with_name(candidate_path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with temp.open('xb') as handle:
                torch.save(payload,handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp,candidate_path)
        finally:
            temp.unlink(missing_ok=True)

    def recover_candidate(study):
        if candidate_prefix is None:
            return []
        from optuna.trial import TrialState
        completed={t.number:t for t in study.trials if t.state==TrialState.COMPLETE and t.value is not None}
        recovered=[]
        # Keep each previous winner in its own artifact. A crash between
        # writing a new candidate and Optuna committing COMPLETE cannot erase
        # the last previously certified winner.
        for candidate_path in candidate_prefix.parent.glob(candidate_prefix.name+'.candidate.*.pt'):
            try:
                outer=torch.load(candidate_path,map_location='cpu',weights_only=True)
                raw=outer['bytes']
                if hashlib.sha256(raw).hexdigest()!=outer['sha256']:
                    raise ValueError('Winner artifact checksum mismatch')
                saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                if (saved['format']!='optuna_winner_v1' or
                        saved['study_key']!=(study_key or Path(study_path).stem)):
                    raise ValueError('Winner study identity mismatch')
                trial=completed.get(saved['trial_number'])
                if trial is None or not math.isclose(float(trial.value),
                           float(saved['validation_reward']),abs_tol=1e-6,rel_tol=1e-9):
                    # Candidate written by an interrupted/uncommitted trial.
                    continue
                settings=PPOSettings(**saved['settings'])
                profile=saved['feature_profile']
                features=EconomicFeatures(obs,train,cfg.execution.starting_balance,
                            symbol_spec=spec,account_currency=cfg.execution.account_currency,
                            min_risk_pct=cfg.intelligence.position_sizing.min_risk_per_trade_pct,
                            max_risk_pct=cfg.intelligence.position_sizing.max_risk_per_trade_pct,
                            **profile)
                if features.n_features!=saved['feature_count']:
                    raise ValueError('Winner observation shape mismatch')
                trader=PPOTrader(features,settings,cfg.research.seed+saved['trial_number'])
                trader.restore(saved['snapshot'])
                recovered.append(OptimizedCandidate(trader,settings,float(saved['validation_reward']),
                                             [],saved['seed_from'],features,int(saved['trial_number'])))
            except (OSError,RuntimeError,ValueError,KeyError,TypeError,EOFError) as exc:
                log.warning('OPTUNA',f'{spec.name}: invalid winner {candidate_path.name}: {exc}')
        recovered.sort(key=lambda item: -item.validation_reward_account_ccy)
        return recovered[:top_k]

    def objective(trial):
        if time.monotonic() >= deadline:
            study.stop()
            raise optuna.TrialPruned('Research budget expired before next trial')
        nonlocal best
        log.info("OPTUNA", f"{spec.name}: trial {trial.number} starting")
        base=cfg.ppo
        if cfg.optimization.enabled and cfg.optimization.trials>1:
            hidden=trial.suggest_categorical('hidden_dim',sorted({base.hidden_dim,max(8,base.hidden_dim//2)}))
            lr=trial.suggest_float('learning_rate',base.learning_rate/2,base.learning_rate*2,log=True)
            entropy=trial.suggest_float('entropy_coef',max(1e-5,base.entropy_coef/4),max(2e-5,base.entropy_coef*4),log=True)
            feature_group=trial.suggest_categorical('feature_group',['all','core'])
            sequence_bars=trial.suggest_categorical('sequence_bars', sorted({max(4,obs.sequence_tensor.shape[1]//2), obs.sequence_tensor.shape[1]}))
            clip=trial.suggest_float('clip_ratio',max(.05,base.clip_ratio*.6),min(.4,base.clip_ratio*1.4))
        else:
            hidden=base.hidden_dim;lr=base.learning_rate;entropy=base.entropy_coef
            feature_group='all';sequence_bars=obs.sequence_tensor.shape[1];clip=base.clip_ratio
        settings=PPOSettings(hidden_dim=hidden,learning_rate=lr,
               gamma=base.gamma,gae_lambda=base.gae_lambda,clip_ratio=clip,
               entropy_coef=entropy,value_coef=base.value_coef,
               ppo_epochs=base.ppo_epochs,episodes=base.episodes,
               target_kl=base.target_kl,validation_interval_episodes=base.validation_interval_episodes)
        features=EconomicFeatures(obs,train,cfg.execution.starting_balance,
                                  symbol_spec=spec,account_currency=cfg.execution.account_currency,
                                  min_risk_pct=cfg.intelligence.position_sizing.min_risk_per_trade_pct,
                                  max_risk_pct=cfg.intelligence.position_sizing.max_risk_per_trade_pct,
                                  feature_group=feature_group,sequence_bars=sequence_bars)
        trial_seed=int(trial.user_attrs.get("resume_seed",cfg.research.seed+trial.number))
        trader=PPOTrader(features,settings,trial_seed)
        source='scratch'
        if seed_lookup is not None:
            initial,source=seed_lookup(settings,features)
            if initial is not None:trader.restore(initial)
        def on_epoch(episode,score,validation):
            # No validation episode is ever used for a gradient update.
            trial.report(score,episode)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Pruned after validation episode {episode+1}")
        try:
            trial_checkpoint=None
            if study_path is not None:
                contract=hashlib.sha256(json.dumps({'params':trial.params,'base':base.model_dump(),
                    'feature_shape':features.n_features},sort_keys=True).encode()).hexdigest()[:16]
                trial_checkpoint=Path(study_path).parent/'checkpoints'/f'{Path(study_path).stem}-{trial_seed}-{contract}.pt'
            score=trader.fit(obs,spec,train,val,cfg,trial_seed,m1=m1,
                             callback=on_epoch,checkpoint_path=trial_checkpoint,
                             deadline_monotonic=deadline)
            best_validation,_=trader.replay(obs,spec,val,cfg,deterministic=True,m1=m1)
            result={'trial':trial.number,'validation_reward_account_ccy':score,
                    'validation_ending_equity':best_validation.ending_equity,
                    'seed_from':source,'decision_diagnostics':trial_decision_diagnostics(best_validation),
                    'parameters':{'hidden_dim':hidden,'learning_rate':lr,'entropy_coef':entropy,
                         'feature_group':feature_group,'sequence_bars':sequence_bars,'clip_ratio':clip}}
            trials.append(result)
            log.info('OPTUNA', f'{spec.name}: trial {trial.number} finished; '
                     f'validation_reward={score:+.2f}; vetoes={result["decision_diagnostics"]["vetoed_proposals"]}')
            trial.set_user_attr('veto_diagnostics',result['decision_diagnostics'])
            candidate=OptimizedCandidate(trader,settings,score,list(trials),source,features,trial.number)
            if len(best)<top_k or score>best[-1].validation_reward_account_ccy:
                persist_candidate(trial.number,candidate)
                best.append(candidate)
                best.sort(key=lambda item: -item.validation_reward_account_ccy)
                best=best[:top_k]
            return score
        except TimeoutError:
            # Preserve the complete-episode checkpoint for resumption and stop
            # the study rather than burning another thousand-trial budget.
            trial.set_user_attr('budget_exhausted', True)
            if trial_checkpoint is not None and trial_checkpoint.is_file() and trial.params:
                # A clean deadline expiry has the same resume contract as a crash.
                # Enqueue identical parameters + seed; the persisted checkpoint
                # restores the complete actor/critic, optimizer and RNG state.
                study.enqueue_trial(dict(trial.params),user_attrs={'resume_seed':trial_seed})
            study.stop()
            raise optuna.TrialPruned('Global research budget expired')
        except optuna.TrialPruned as exc:
            if trial_checkpoint is not None:trial_checkpoint.unlink(missing_ok=True)
            trials.append({"trial":trial.number,"status":"PRUNED","reason":str(exc)})
            log.info("OPTUNA", f"{spec.name}: trial {trial.number} pruned: {exc}")
            raise
        except Exception as exc:
            trials.append({'trial':trial.number,'error':str(exc)})
            log.warning('OPTUNA', f'{spec.name}: trial {trial.number} failed ({type(exc).__name__}): {exc}')
            raise
    kwargs={}
    if study_path is not None:
        db=Path(study_path);db.parent.mkdir(parents=True,exist_ok=True)
        kwargs=dict(storage='sqlite:///'+str(db.resolve()),study_name=study_key or db.stem,load_if_exists=True)
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=cfg.research.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=cfg.optimization.pruner_startup_trials,
            n_warmup_steps=cfg.optimization.pruner_warmup_episodes),**kwargs)
    # Scoreboard and winner weights are recovered together; never treat a
    # historical Optuna score as a reconstructible policy without its tensors.
    best=recover_candidate(study)
    # A power loss leaves Optuna RUNNING and the last complete episode checkpoint
    # on disk. The exclusive research lock guarantees no competing process is
    # legitimately executing this study; mark the interrupted trial failed and
    # enqueue the SAME params and SAME seed. The new trial resumes optimizer,
    # actor, critic and RNG state from its checkpoint, not just model weights.
    if study_path is not None:
        from optuna.trial import TrialState
        for interrupted in list(study.trials):
            if interrupted.state == TrialState.RUNNING:
                study.tell(interrupted.number,state=TrialState.FAIL)
                if interrupted.params:
                    saved_seed=int(interrupted.user_attrs.get('resume_seed',
                                   cfg.research.seed+interrupted.number))
                    study.enqueue_trial(interrupted.params,user_attrs={'resume_seed':saved_seed})
                    log.warning('OPTUNA', f'{spec.name}: resuming interrupted trial '
                                f'{interrupted.number} with seed {saved_seed}')
    # Seed the NEW rolling-window study with previously promising parameters.
    # The trials are retrained and re-scored on CURRENT observations; old rewards
    # are never injected into the current objective or promotion score.
    history_path=(Path(study_path).parent/f'{spec.name}.search-parameters.json') if study_path is not None else None
    if history_path is not None and history_path.is_file() and not study.trials:
        try:
            hist=json.loads(history_path.read_text('utf-8'))
            if hist.get('config_hash')==cfg.config_hash:
                for params in hist.get('top_parameters',[])[:3]:
                    study.enqueue_trial(params)
        except (OSError,ValueError,TypeError):
            log.warning('OPTUNA',f'{spec.name}: ignoring invalid parameter search archive')
    study.optimize(objective,n_trials=cfg.optimization.trials if cfg.optimization.enabled else 1,
                   timeout=max(1,int(deadline-time.monotonic())),
                   catch=(ValueError,RuntimeError,FloatingPointError))
    if history_path is not None:
        from optuna.trial import TrialState
        completed=sorted((t for t in study.trials if t.state==TrialState.COMPLETE
                          and t.value is not None),key=lambda t:-float(t.value))
        saved={'config_hash':cfg.config_hash,
               'top_parameters':[t.params for t in completed[:3]],
               'source_study':study.study_name}
        history_path.parent.mkdir(parents=True,exist_ok=True)
        tmp=history_path.with_name(history_path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with tmp.open('x',encoding='utf-8') as f:
                json.dump(saved,f,sort_keys=True)
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,history_path)
        finally:
            tmp.unlink(missing_ok=True)
    if not best:raise RuntimeError(f'No recoverable completed PPO candidate: {trials}')
    # Keep only the current top-K immutable snapshots for this study. Each file
    # is independently checksummed; SQLite COMPLETE state is still required.
    if candidate_prefix is not None:
        keep_trials={candidate.trial_number for candidate in best}
        for path in candidate_prefix.parent.glob(candidate_prefix.name+'.candidate.*.pt'):
            try:
                outer=torch.load(path,map_location='cpu',weights_only=True)
                raw=outer['bytes']
                if hashlib.sha256(raw).hexdigest()!=outer['sha256']:
                    raise ValueError('Winner artifact checksum mismatch during cleanup')
                saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                if saved.get('trial_number') not in keep_trials:
                    path.unlink(missing_ok=True)
            except (OSError,RuntimeError,ValueError,KeyError,TypeError,EOFError):
                path.unlink(missing_ok=True)
    for candidate in best:
        candidate.trial_report=trials
    return best[0] if top_k==1 else best
