"""Single-symbol sequential PPO research and immutable global population."""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import hashlib
import io
import json
import math
import os
import time
import uuid
import numpy as np
import torch
from quant_rl.config.schema import QuantRLConfig
from quant_rl.config.logger import log
from quant_rl.data.mt5_history import MT5History
from quant_rl.data.mt5_source import broker_spec_hash
from quant_rl.data.rolling_window import select_rolling_bars
from quant_rl.market.state_builder import ObservationBuilder
from quant_rl.intelligence.policy import EconomicFeatures, PPOTrader, PPOSettings
from quant_rl.research.optimizer import optimize_candidate
from quant_rl.reporting.economic_report import write_economic_report


def chronological_split(time_s, warmup, *, window_bounds=None):
    n=len(time_s); usable=n-warmup
    if usable < 120 or np.any(np.diff(time_s)<=0):
        raise ValueError('Insufficient completed chronological candles')
    if window_bounds is not None:
        window_start,window_end=window_bounds
        if window_end <= window_start or time_s[-1] >= window_end:
            raise ValueError('Invalid calendar-time split bounds')
        a=max(warmup+1,int(np.searchsorted(time_s,window_start+(window_end-window_start)*0.5)))
        b=max(a+1,int(np.searchsorted(time_s,window_start+(window_end-window_start)*0.75)))
        if a>=b or b>=n:
            raise ValueError('Insufficient calendar-time split coverage')
    else:
        a=warmup+usable//2;b=a+usable//4
    return (np.arange(warmup,a,dtype=np.int64),
            np.arange(a,b,dtype=np.int64),
            np.arange(b,n,dtype=np.int64))


def select_champion(scores):
    finite={k:v for k,v in scores.items() if math.isfinite(v)}
    return min(finite,key=lambda k:(-finite[k],k)) if finite else None


def architecture_hash(cfg,features,settings):
    payload={'input':features.n_features,'hidden':settings.hidden_dim,
             'geometry':cfg.intelligence.geometry.model_dump(),
             'sizing':cfg.intelligence.position_sizing.model_dump(),
             'timeframes':cfg.data.observation_timeframes,
             'technical':cfg.data.include_technical_indicators,
             'feature_group':features.feature_group,'sequence_bars':features.sequence_bars,
             'bar_price_side':features.obs.bar_price_side,
             'economic_features_contract':'broker_affordability_v1',
             'account_currency':cfg.execution.account_currency}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16]


class ModelPopulation:
    """The only model score authority: one JSON manifest and immutable model weights."""
    FORMAT = 'quant_rl_v5_account_server_certificate_v3'
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest_path = self.root / 'best_score.json'
        self.models = self.root / 'models'
        self.models.mkdir(parents=True,exist_ok=True)
        if self.manifest_path.exists() and self.manifest_path.stat().st_size > 0:
            data = json.loads(self.manifest_path.read_text('utf-8'),
                              parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite scoreboard value')))
            if (not isinstance(data,dict) or set(data) != {'format','symbols','champion','cycles'}
                    or data['format']!=self.FORMAT or not isinstance(data['symbols'],dict)):
                raise ValueError('Invalid global best_score.json; refusing to reuse unverified models')
            self.data=data
        else:
            self.data={'format':self.FORMAT,'symbols':{}, 'champion':None, 'cycles':{}}

    def verify_artifact(self, record: dict) -> None:
        """Verify artifact bytes and saved tensor shape, without trusting a self-asserted arch."""
        self._checked_artifact(record)

    def load(self, record: dict, expected_arch: str) -> dict:
        if record['arch'] != expected_arch:
            raise ValueError('Incompatible model architecture; cannot transfer weights')
        return self._checked_artifact(record)

    def _checked_artifact(self, record: dict) -> dict:
        name=record['file']
        if '/' in name or '\\' in name or '..' in name:
            raise ValueError('Unsafe model artifact reference')
        path=self.models / name
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Model artifact checksum mismatch')
        saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if set(saved) != {'weights','mean','std'}:
            raise ValueError('Invalid model tensor schema')
        return saved

    def publish(self, accepted: dict[str,tuple[dict,dict]], evaluations: dict[str,dict], champion: str | None, completed: dict[str,dict]):
        """Create immutable weight files, then atomically replace the ONLY score authority."""
        # Retain archived records; current broker and model certificates are
        # independently checked before a model enters live routing.
        # Archive all historical symbols, including ones no longer configured.
        records={s:copy.deepcopy(rec) for s,rec in self.data['symbols'].items()}
        for symbol, (snapshot, metadata) in accepted.items():
            buf=io.BytesIO()
            torch.save(snapshot,buf)
            raw=buf.getvalue(); sha=hashlib.sha256(raw).hexdigest()
            safe=''.join(c if c.isalnum() or c in '._-' else '_' for c in symbol)
            name=f'{safe}-{sha[:20]}.pt'
            target=self.models/name
            if not target.exists():
                temp=self.models/(name+'.'+uuid.uuid4().hex+'.tmp')
                with temp.open('wb') as f:
                    f.write(raw);f.flush();os.fsync(f.fileno())
                os.replace(temp,target)
            records[symbol]={**metadata,'file':name,'sha256':sha}
        for symbol,meta in evaluations.items():
            if symbol in records:
                records[symbol].update(meta)
        # Champion must refer to one of the newly current evaluated and eligible models.
        if champion is not None and champion not in records:
            raise ValueError('Champion does not exist in population')
        payload={'format':self.FORMAT,'symbols':records,'champion':champion, 'cycles':{**self.data['cycles'],**completed}}
        temp=self.manifest_path.with_name('.best_score.'+uuid.uuid4().hex+'.tmp')
        with temp.open('w',encoding='utf-8') as f:
            json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(temp,self.manifest_path)
        self.data=payload
        # Retain immutable historical model weights for audit and rollback.
        # Pruning must be an explicit, separate operator action, never promotion side effect.


def append_research_event(root: Path, payload: dict) -> None:
    """Durably record each attempt separately from the current winning scoreboard."""
    path=root/'research_runs.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    event={'recorded_utc':datetime.now(timezone.utc).isoformat(),**payload}
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,sort_keys=True,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive_run(root: Path):
    """Use an OS-released, cross-process file lock for both modes.

    The lock file deliberately remains on disk: deleting a locked inode allows
    a second process to lock a new inode at the same pathname on POSIX. Kernel
    locks are released after process termination, including crashes. The file
    content is diagnostic only; it never confers ownership.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'research.lock'
    token = uuid.uuid4().hex
    with path.open('a+b') as handle:
        locked = False
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                # Windows locking locks bytes relative to the current file
                # pointer. Always use byte zero for every process.
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except (OSError, BlockingIOError) as exc:
            handle.seek(0)
            owner = handle.read(1024).decode('utf-8', errors='replace')
            raise RuntimeError(f'Concurrent run lock at {path}; owner={owner}') from exc
        try:
            owner = {'pid': os.getpid(), 'created_utc': datetime.now(timezone.utc).isoformat(),
                     'token': token, 'purpose': 'exclusive MT5 research/LIVE process'}
            handle.seek(0)
            handle.truncate(0)
            handle.write(json.dumps(owner).encode('utf-8'))
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            if locked:
                if os.name == 'nt':
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)



def run_research_session(config: QuantRLConfig, *, targets=None, preloaded=None, reason='MISSING_MODELS') -> dict:
    """Research one instrument at a time, atomically publish only qualified winners."""
    # Tiny per-decision networks are much faster without large CPU thread
    # pools. This is explicit and configurable for IDE-only research runs.
    torch.set_num_threads(config.ppo.cpu_threads)
    targets=tuple(config.research.symbols if targets is None else targets)
    if not targets or any(s not in config.research.symbols for s in targets):
        raise ValueError("Research targets must be nonempty configured symbols")
    if reason not in ("MISSING_MODELS", "MARKET_CLOSED", "WEEKLY_SCHEDULE"):
        raise ValueError("Invalid automatic research trigger")
    preloaded=preloaded or {}
    root=Path(config.project_root).resolve()
    now_ts=int(time.time())
    global_deadline=time.monotonic()+config.optimization.max_session_seconds
    def require_time(phase_name):
        if time.monotonic() >= global_deadline:
            raise TimeoutError(f'Research session deadline reached during {phase_name}')
    cycle=datetime.fromtimestamp(now_ts,timezone.utc).date().isoformat()
    with exclusive_run(root):
        population=ModelPopulation(root)
        history=MT5History(config)
        builder=ObservationBuilder(list(config.data.observation_timeframes),
                     config.data.sequence_lookback_bars,config.data.include_technical_indicators)
        # Scores enter the champion competition ONLY after re-evaluation in this run.
        # Old windows are not directly comparable with current-window scores.
        results={};accepted={};evaluations={};cycles={};scores={}
        accepted_symbols=[]
        # Input weights from symbol incumbents or previously persisted global champion.
        # Do not use current-cycle final evaluation to warm-start other models this cycle.
        old_global=population.data.get('champion')
        global_record=population.data['symbols'].get(old_global) if old_global else None
        if global_record and (global_record.get('mt5_server_timezone') != config.broker_connection.mt5_server_timezone
                or global_record.get('policy_hash') != config.policy_hash
                or global_record.get('account_currency') != config.execution.account_currency):
            global_record=None
        log.info('RESEARCH', f'Starting research for {len(targets)} symbol(s): {list(targets)}')
        log.warning('RESEARCH', f'Training budget per symbol: {config.optimization.trials} Optuna trials '
                    f'x {config.ppo.episodes} PPO episodes x {config.ppo.ppo_epochs} PPO epochs '
                    f'= {config.optimization.trials * config.ppo.episodes * config.ppo.ppo_epochs:,} '
                    'PPO epoch passes before minibatches at the configured upper bounds. KL stopping '
                    'and trial pruning may reduce work; Optuna timeout is checked between trials.')
        for symbol in targets:
            if time.monotonic() >= global_deadline:
                log.warning('RESEARCH', 'Global session budget exhausted; stopping before next symbol')
                break
            phase='DATA_ACQUISITION'
            try:
                log.info('RESEARCH', f'{symbol}: retrieving/validating freshly acquired MT5 dataset')
                frozen=preloaded.pop(symbol,None)
                if frozen is None:
                    frozen=history.fetch(symbol,base_timeframe='H1',deadline_monotonic=global_deadline)
                require_time('data acquisition')
                if frozen.manifest.data_origin!='MT5_TERMINAL':
                    raise ValueError('Research requires fresh MT5 data')
                if frozen.manifest.verified_spread_fraction < 1.0 and not (
                        frozen.fallback_spread_points is not None and frozen.fallback_spread_points > 0):
                    raise ValueError('MT5 historical spread missing with no broker quote for estimated costs')
                bars,window_start,window_end=select_rolling_bars(
                    frozen.bars,frozen.timeframe,config.research.lookback_years,
                    now_utc_s=now_ts)
                log.info('RESEARCH', f'{symbol}: {config.research.lookback_years}-year rolling '
                         f'window {datetime.fromtimestamp(window_start,timezone.utc).isoformat()} '
                         f'to {datetime.fromtimestamp(window_end,timezone.utc).isoformat()}')
                require_time('dataset preparation')
                obs=builder.build_dataset(base_bars=bars,symbol=symbol,
                    primary_timeframe=config.research.timeframe,
                    volatility_estimator=config.research.volatility_estimator,
                    volatility_period=config.research.volatility_period,spec=frozen.spec,
                    bar_price_side=frozen.manifest.bar_price_side,
                    fallback_spread_points=(frozen.fallback_spread_points
                                            if frozen.fallback_spread_points is not None else float("nan")))
                require_time('observation construction')
                train,val,test=chronological_split(obs.time_s,max(40,config.data.sequence_lookback_bars),
                                                    window_bounds=(window_start,window_end))
                # Search can use only the first validation half; a second untouched
                # validation half adjudicates promotion. Final 25% is for monitoring,
                # never model acceptance. This prevents repeated test-set promotion.
                search_val=val[:len(val)//2]
                promotion_val=val[len(val)//2:]
                if len(search_val)<8 or len(promotion_val)<8:
                    raise ValueError('Insufficient separate search and promotion validation intervals')
                period={'window_start_utc':datetime.fromtimestamp(window_start,timezone.utc).isoformat(),
                        'window_end_utc':datetime.fromtimestamp(window_end,timezone.utc).isoformat(),
                        'lookback_years':config.research.lookback_years,
                        'train_start_utc':datetime.fromtimestamp(int(obs.time_s[train[0]]),timezone.utc).isoformat(),
                        'train_end_utc':datetime.fromtimestamp(int(obs.decision_time_s[train[-1]]),timezone.utc).isoformat(),
                        'validation_start_utc':datetime.fromtimestamp(int(obs.time_s[val[0]]),timezone.utc).isoformat(),
                        'validation_end_utc':datetime.fromtimestamp(int(obs.decision_time_s[val[-1]]),timezone.utc).isoformat(),
                        'test_start_utc':datetime.fromtimestamp(int(obs.time_s[test[0]]),timezone.utc).isoformat(),
                        'test_end_utc':datetime.fromtimestamp(int(obs.decision_time_s[test[-1]]),timezone.utc).isoformat()}
                identity=hashlib.sha256(json.dumps({'fingerprint':frozen.fingerprint,'period':period,
                                                      'config':config.config_hash, 'broker_server':frozen.broker_server, 'policy_hash':config.policy_hash, 'broker_spec':frozen.spec.__dict__},sort_keys=True).encode()).hexdigest()[:20]
                prior_cycle=population.data['cycles'].get(symbol) or {}
                attempt=int(prior_cycle.get('attempt',0))+1
                # Market-closed research explores NEW Optuna trials even if candles are
                # unchanged; never skip an identical history automatically.
                m1=(frozen.m1_time_s,frozen.m1_high,frozen.m1_low,
                    frozen.m1_spread_points,frozen.m1_ordered)
                own_record=population.data['symbols'].get(symbol)
                if own_record and (own_record.get('mt5_server_timezone') != config.broker_connection.mt5_server_timezone
                    or own_record.get('broker_server') != frozen.broker_server
                    or own_record.get('policy_hash') != config.policy_hash
                    or own_record.get('account_currency') != config.execution.account_currency
                    or own_record.get('broker_spec_hash') != broker_spec_hash(frozen.spec)):
                    own_record=None
                compatible_global=(global_record if global_record is not None
                                   and global_record.get('broker_server')==frozen.broker_server
                                   and global_record.get('broker_spec_hash')==broker_spec_hash(frozen.spec) else None)
                def load_seed(settings,features):
                    arch=architecture_hash(config,features,settings)
                    for title,record in [('symbol_incumbent',own_record),('previous_global_champion',compatible_global)]:
                        if record is not None and record.get('arch')==arch:
                            try:
                                saved=population.load(record,arch)
                                # Transfer learned weights but normalize using the TARGET symbol's
                                # training history; never transport another symbol's feature statistics.
                                return {**saved, **features.snapshot()},title
                            except (ValueError,FileNotFoundError,KeyError,RuntimeError):
                                continue
                    return None,'scratch'
                phase='OPTIMIZATION'
                require_time('optimization start')
                candidates=optimize_candidate(config,obs,frozen.spec,train,search_val,m1,load_seed,
                    study_path=root/'optuna_studies'/f'{symbol}-{identity}.db',study_key=f'{symbol}-{identity}',
                    deadline_monotonic=global_deadline,top_k=config.optimization.tournament_top_k)
                if not isinstance(candidates,list):
                    candidates=[candidates]
                require_time('optimization')
                if not candidates:
                    raise RuntimeError('Optimizer returned an empty finalist pool')
                phase='EVALUATION'
                # All finalists are judged on the withheld promotion segment;
                # search_val never participates in promotion or cost-stress replay.
                eval_exec=config.execution.model_copy(update={'starting_balance':10000.0})
                eval_cfg=config.model_copy(update={'execution':eval_exec})
                finalists=[]
                for rank,candidate in enumerate(candidates,1):
                    candidate.trader.features.initial_balance=10000.0
                    require_time('promotion evaluation')
                    promotion_sim=evaluate_candidate(candidate.trader,obs,frozen.spec,promotion_val,eval_cfg,m1,deadline_monotonic=global_deadline)
                    stress=evaluate_candidate(candidate.trader,obs,frozen.spec,promotion_val,eval_cfg,m1,
                                               config.qualification.cost_stress_multiplier,deadline_monotonic=global_deadline)
                    eligible,reasons=is_qualified(
                        promotion_sim, stress, config, frozen.manifest.data_origin,
                        m1_price_coverage_fraction=frozen.manifest.m1_window_coverage_fraction,
                        m1_spread_coverage_fraction=frozen.manifest.m1_spread_coverage_fraction)
                    score=(promotion_sim.ending_equity/10000.-1.) if promotion_sim.executed_trades else -float('inf')
                    finalists.append({'candidate':candidate,'rank':rank,'promotion':promotion_sim,
                                      'stress':stress,'eligible':eligible,'reasons':reasons,
                                      'score':score,'test':None})
                finalists.sort(key=lambda item:(-item['score'],item['rank']))
                # Sequential held-out fallback: only promotion-qualified finalists
                # are tested, in promotion-score order; the first positive net
                # return wins. This consumes the held-out segment for selection.
                selected_finalist=None
                for finalist in finalists:
                    if not finalist['eligible']:
                        continue
                    require_time('held-out evaluation')
                    finalist['test']=evaluate_candidate(finalist['candidate'].trader,obs,frozen.spec,
                                                        test,eval_cfg,m1,deadline_monotonic=global_deadline)
                    if finalist['test'].net_return_pct>0:
                        selected_finalist=finalist
                        break
                # Preserve the strongest promotion candidate for diagnostics if
                # no finalist clears the held-out fallback; it is not publishable.
                diagnostic_finalist=finalists[0]
                if diagnostic_finalist['test'] is None:
                    require_time('diagnostic held-out evaluation')
                    diagnostic_finalist['test']=evaluate_candidate(
                        diagnostic_finalist['candidate'].trader,obs,frozen.spec,test,eval_cfg,m1,deadline_monotonic=global_deadline)
                finalist=selected_finalist or diagnostic_finalist
                candidate=finalist['candidate']
                challenger=candidate.trader
                arch=architecture_hash(config,candidate.features,candidate.settings)
                sim=finalist['test']
                promotion_sim=finalist['promotion']
                eligible=bool(selected_finalist is not None)
                reasons=list(finalist['reasons'])
                if selected_finalist is None:
                    reasons.append('HELD_OUT_TEST_NO_POSITIVE_NET_RETURN')
                score=finalist['score']
                incumbent_score=-float('inf')
                incumbent=None
                if own_record is not None and 'settings' in own_record:
                    try:
                        old_settings=PPOSettings(**own_record['settings'])
                        profile=own_record.get('feature_profile',{'feature_group':'all',
                                                   'sequence_bars':obs.sequence_tensor.shape[1]})
                        old_features=EconomicFeatures(obs,train,10000.,symbol_spec=frozen.spec,
                            account_currency=config.execution.account_currency,
                            min_risk_pct=config.intelligence.position_sizing.min_risk_per_trade_pct,
                            max_risk_pct=config.intelligence.position_sizing.max_risk_per_trade_pct,
                            **profile)
                        old_arch=architecture_hash(config,old_features,old_settings)
                        snap=population.load(own_record,old_arch)
                        incumbent=PPOTrader(old_features,old_settings,config.research.seed)
                        incumbent.restore(snap)
                        require_time('incumbent evaluation')
                        old_sim=evaluate_candidate(incumbent,obs,frozen.spec,promotion_val,eval_cfg,m1,deadline_monotonic=global_deadline)
                        old_stress=evaluate_candidate(incumbent,obs,frozen.spec,promotion_val,eval_cfg,m1,
                                                  config.qualification.cost_stress_multiplier,deadline_monotonic=global_deadline)
                        ok,_=is_qualified(
                            old_sim, old_stress, config, frozen.manifest.data_origin,
                            m1_price_coverage_fraction=frozen.manifest.m1_window_coverage_fraction,
                            m1_spread_coverage_fraction=frozen.manifest.m1_spread_coverage_fraction)
                        if ok:incumbent_score=old_sim.ending_equity/10000.-1.
                    except (RuntimeError,ValueError,FileNotFoundError,KeyError):
                        incumbent=None
                challenger_wins=bool(selected_finalist is not None and score>incumbent_score)
                chosen=challenger if challenger_wins else incumbent if incumbent_score>-float('inf') else challenger
                chosen_sim=(sim if chosen is challenger else
                            evaluate_candidate(incumbent,obs,frozen.spec,test,eval_cfg,m1,deadline_monotonic=global_deadline))
                chosen_eligible=bool(challenger_wins or (chosen is incumbent and incumbent_score>-float('inf')))
                chosen_reasons=(reasons if chosen is challenger else [])
                report_settings=(candidate.settings.__dict__ if chosen is challenger
                                 else own_record['settings'])
                report_profile=({'feature_group':candidate.features.feature_group,
                                 'sequence_bars':candidate.features.sequence_bars}
                                if chosen is challenger else own_record.get('feature_profile',{}))
                require_time('report generation')
                report_path=write_economic_report(root,symbol,chosen_sim,
                      origin=frozen.manifest.data_origin,settings={**report_settings,
                             'feature_profile':report_profile},period=period,
                      qualified=chosen_eligible,reasons=chosen_reasons,
                      trial_report=candidate.trial_report,
                      provenance={'dataset_fingerprint':frozen.fingerprint,
                                  'bar_price_side':frozen.manifest.bar_price_side,
                                  'reported_spread_fraction':frozen.manifest.verified_spread_fraction,
                                  'm1_intrabar_coverage_fraction':frozen.manifest.m1_window_coverage_fraction,
                                  'm1_spread_coverage_fraction':frozen.manifest.m1_spread_coverage_fraction,
                                  'spread_provenance':frozen.manifest.spread_provenance,
                                  'fallback_spread_points':frozen.fallback_spread_points,
                                  'fallback_spread_source':frozen.manifest.fallback_spread_source,
                                  'config_hash':config.config_hash,
                                  'qualification_segment':'PROMOTION_VALIDATION_PLUS_CONSUMED_HELD_OUT_SELECTION',
                                  'held_out_selection_consumed':True,
                                  'tournament_top_k':config.optimization.tournament_top_k,
                                  'finalists_evaluated':len(finalists),
                                  'promotion_qualified_finalists':sum(1 for item in finalists if item['eligible']),
                                  'held_out_positive_finalist_found':selected_finalist is not None,
                                  'model_architecture':
                                     (arch if chosen is challenger else own_record['arch'])})
                require_time('model qualification and publication')
                if chosen is challenger and eligible and score>incumbent_score:
                    historical_best=max(float(own_record.get('historical_best',-float('inf'))) if own_record else -float('inf'),score)
                    accepted[symbol]=(challenger.snapshot(),{
                        'arch':arch,'cycle':cycle,'run_key':identity,'current_score':score,
                        'score':score,'score_basis':'PROMOTION_VALIDATION_WITH_POSITIVE_HELD_OUT_SELECTION',
                        'historical_best':historical_best,'validation_reward_account_ccy':candidate.validation_reward_account_ccy,
                        'seed_from':candidate.seed_from,'origin':frozen.manifest.data_origin,
                        'account_currency':config.execution.account_currency,
                        'broker_server':frozen.broker_server,
                        'broker_spec_hash':broker_spec_hash(frozen.spec),
                    'policy_hash':config.policy_hash,
                        'mt5_server_timezone':config.broker_connection.mt5_server_timezone,
                        'period':period,'settings':candidate.settings.__dict__,
                        'feature_profile':{'feature_group':candidate.features.feature_group,
                                          'sequence_bars':candidate.features.sequence_bars}})
                    scores[symbol]=score
                    accepted_symbols.append(symbol)
                    status='ACCEPTED'
                elif incumbent_score>-float('inf'):
                    scores[symbol]=incumbent_score
                    evaluations[symbol]={'current_score':incumbent_score,'cycle':cycle,'period':period}
                    status='RETAINED_INCUMBENT'
                else:
                    status='DEMO_ONLY' if frozen.manifest.data_origin!='MT5_TERMINAL' else 'NO_QUALIFIED_MODEL'
                cycles[symbol]={'cycle':cycle,'identity':identity,'status':status,'attempt':attempt,'reason':reason}
                results[symbol]={'status':status,'challenger_score':score if math.isfinite(score) else None,
                    'score':scores.get(symbol),'reasons':reasons,'period':period,
                    'report':str(report_path),'origin':frozen.manifest.data_origin,'attempt':attempt}
                # Completed symbols are durable even if a later symbol crashes.
                interim_champion=select_champion(scores)
                phase='PUBLICATION'
                population.publish(accepted,evaluations,interim_champion,cycles)
                accepted.clear()  # Persisted snapshots need not be serialized again for later symbols.
                append_research_event(root,{'symbol':symbol,'cycle':cycle,'status':status,
                                             'study_identity':identity,'dataset_fingerprint':frozen.fingerprint,
                                             'broker_server':frozen.broker_server,'score':scores.get(symbol)})
                log.info('RESEARCH', f'{symbol}: {status}; challenger ending equity=${sim.ending_equity:,.2f} '
                         f'closed={len(sim.executed_trades)} vetoes={sim.veto_count}')
            except (ValueError,ConnectionError,RuntimeError,OSError,TypeError,IndexError) as exc:
                deadline_hit = isinstance(exc, TimeoutError) or time.monotonic() >= global_deadline
                status=('SESSION_DEADLINE_EXCEEDED' if deadline_hit else
                        'DATA_REJECTED' if phase=='DATA_ACQUISITION' else f'{phase}_FAILED')
                log.warning('RESEARCH',f'{symbol}: {status}: {type(exc).__name__}: {exc}')
                results[symbol]={'status':status,'error':str(exc)}
                cycles[symbol]={'cycle':cycle,'status':status,
                                'reason':str(exc),'attempt':int((population.data['cycles'].get(symbol) or {}).get('attempt',0))+1}
                population.publish(accepted,evaluations,select_champion(scores),cycles)
                append_research_event(root,{'symbol':symbol,'cycle':cycle,'status':status,
                                            'error_type':type(exc).__name__,'error':str(exc)})
                if deadline_hit:
                    break
                continue
        champion=select_champion(scores)
        population.publish(accepted,evaluations,champion,cycles)
        log.success('POPULATION', f'champion={champion or "none"}; new models={list(accepted)}')
        return {'champion':champion,'accepted':accepted_symbols,'results':results,
                'score_file':str(population.manifest_path)}


# Candidate replay and promotion qualification live with the evolution policy.
def is_qualified(sim,stress,cfg,origin,*,
                 m1_price_coverage_fraction=None,m1_spread_coverage_fraction=None):
    """Apply the same promotion evidence contract to challengers and incumbents."""
    q=cfg.qualification
    days=(int(sim.time_s_curve[-1])-int(sim.time_s_curve[0]))/86400 if len(sim.time_s_curve)>1 else 0
    reasons=[]
    if not sim.executed_trades:reasons.append('NO_CLOSED_TRADES')
    if days<q.minimum_evaluation_calendar_days:reasons.append('INSUFFICIENT_CALENDAR_EVIDENCE')
    if sim.trades_per_year<q.minimum_trades_per_year:reasons.append('INSUFFICIENT_TRADE_ACTIVITY')
    if sim.net_return_pct<=0:reasons.append('NON_POSITIVE_NET_EQUITY')
    if sim.max_drawdown_pct>q.max_drawdown_pct or stress.max_drawdown_pct>q.max_drawdown_pct:
        reasons.append('DRAWDOWN_LIMIT')
    if q.require_positive_stressed_return and stress.net_return_pct<=0:
        reasons.append('COST_STRESS_FAILED')
    if origin!='MT5_TERMINAL':
        reasons.append('NON_MT5_HISTORY')
    minimum_m1=q.minimum_m1_coverage_fraction
    if m1_price_coverage_fraction is None or m1_price_coverage_fraction < minimum_m1:
        reasons.append('INSUFFICIENT_M1_PRICE_PATH_EVIDENCE')
    if m1_spread_coverage_fraction is None or m1_spread_coverage_fraction < minimum_m1:
        reasons.append('INSUFFICIENT_M1_SPREAD_EVIDENCE')
    return not reasons,reasons


def evaluate_candidate(trader,obs,spec,indices,cfg,m1=None,stress_multiplier=1.,deadline_monotonic=None):
    sim,_=trader.replay(obs,spec,indices,cfg,deterministic=True,m1=m1,
                        cost_multiplier=stress_multiplier,deadline_monotonic=deadline_monotonic)
    return sim
