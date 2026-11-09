"""PPO learning on completed economic decision events with elapsed-bar aware GAE."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import time
import uuid
import copy
import math
import numpy as np
import torch
from quant_rl.intelligence.actions import to_proposal
from quant_rl.intelligence.model import HybridActorCritic
from quant_rl.execution.accounting import PortfolioAccountingSimulator, CausalMarketContextCache
from quant_rl.execution.sizing import compute_pip_value_per_lot
from quant_rl.data.mt5_source import SymbolSpec


@dataclass(frozen=True)
class PPOSettings:
    hidden_dim: int = 64
    learning_rate: float = 0.0003
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    ppo_epochs: int = 1000
    episodes: int = 1000
    target_kl: float = 0.025
    validation_interval_episodes: int = 25


class EconomicFeatures:
    """Frozen train-only normalizer; augments market inputs with contemporaneous account state."""
    def __init__(self, obs, train_indices: np.ndarray, starting_balance: float,
                 *, symbol_spec: SymbolSpec, account_currency: str,
                 min_risk_pct: float, max_risk_pct: float,
                 feature_group: str = 'all', sequence_bars: int | None = None):
        self.obs=obs;self.initial_balance=starting_balance
        if (symbol_spec.symbol != obs.symbol or not 0 < min_risk_pct <= max_risk_pct
                or max_risk_pct > 100 or symbol_spec.volume_min <= 0
                or symbol_spec.pip_size <= 0):
            raise ValueError('Invalid broker affordability observation inputs')
        self.symbol_spec=symbol_spec
        self.account_currency=account_currency
        self.min_risk_fraction=min_risk_pct/100.0
        self.max_risk_fraction=max_risk_pct/100.0
        if feature_group not in ('all','core'):
            raise ValueError('Unknown feature group')
        self.feature_group=feature_group
        self.sequence_bars=int(sequence_bars or obs.sequence_tensor.shape[1])
        if self.sequence_bars<1 or self.sequence_bars>obs.sequence_tensor.shape[1]:
            raise ValueError('Invalid sequence lookback')
        if feature_group == 'core':
            keys=('ret_', 'range', 'body', 'vol_', 'spread', 'hour_', 'dow_')
            columns=[i for i,name in enumerate(obs.feature_names)
                     if name.startswith(keys) and not name.startswith('D1_')]
            if not columns:raise ValueError('Core feature selection is empty')
            seq_columns=[i for i,name in enumerate(obs.sequence_feature_names)
                         if name.startswith(keys)]
        else:
            columns=list(range(obs.feature_matrix.shape[1]))
            seq_columns=list(range(obs.sequence_tensor.shape[2]))
        seq=obs.sequence_tensor[:,-self.sequence_bars:,:][:,:,seq_columns].reshape(obs.num_bars,-1)
        self.raw=np.hstack([obs.feature_matrix[:,columns],seq]).astype(np.float64)
        x=self.raw[train_indices]
        self.mean=np.mean(x,axis=0)
        self.std=np.maximum(np.std(x,axis=0),1e-5)
        self.n_features=self.raw.shape[1]+8

    def __call__(self, bar_index: int, state, fx_rates: dict[str, float] | None = None) -> np.ndarray:
        market=np.clip((self.raw[bar_index]-self.mean)/self.std,-8,8)
        account=np.array([
            math.log(max(state.balance,1.0)/self.initial_balance),
            math.log(max(state.equity,1.0)/self.initial_balance),
            max(0.,(state.peak_equity-state.equity)/max(state.peak_equity,1.)),
            min(1.,state.consecutive_losses/10.),
        ],dtype=np.float64)
        # Point-in-time broker feasibility: the policy must see why a wide
        # stop combined with its chosen risk can fall below minimum lot size.
        # No market future, rejected-proposal outcome, or Examiner approval is
        # provided. The actual position size is still computed by the Examiner.
        spec=self.symbol_spec
        equity=max(float(state.equity),1e-9)
        price=float(self.obs.close[bar_index])
        vol_pips=max(float(self.obs.volatility_price[bar_index])/spec.pip_size,1e-9)
        pip_value=compute_pip_value_per_lot(spec,price,self.account_currency,fx_rates)
        risk_per_vol_unit=max(spec.volume_min*vol_pips*pip_value,1e-9)
        affordable_min=equity*self.min_risk_fraction/risk_per_vol_unit
        affordable_max=equity*self.max_risk_fraction/risk_per_vol_unit
        broker_stop_floor=max(spec.stops_level_points*spec.point,2.*spec.pip_size)
        broker=np.array([
            math.log1p(max(0.,affordable_min)),
            math.log1p(max(0.,affordable_max)),
            min(10.,broker_stop_floor/max(float(self.obs.volatility_price[bar_index]),1e-9)),
            min(10.,max(0.,float(self.obs.spread_price[bar_index]))/max(float(self.obs.volatility_price[bar_index]),1e-9)),
        ],dtype=np.float64)
        return np.concatenate([market,account,broker]).astype(np.float32)

    def snapshot(self):
        return {'mean':torch.as_tensor(self.mean.copy(),dtype=torch.float32),
                'std':torch.as_tensor(self.std.copy(),dtype=torch.float32)}

    def restore(self,data):
        if len(data['mean'])!=len(self.mean):
            raise ValueError('Incompatible observation normalization')
        self.mean=data['mean'].cpu().numpy().astype(np.float64)
        self.std=data['std'].cpu().numpy().astype(np.float64)


def advantage_estimates(rewards: np.ndarray,values: np.ndarray,indices: np.ndarray,
                        gamma: float, lam: float, resolved: np.ndarray):
    """Variable-time semi-Markov GAE: discount per elapsed H4 bar, not per trade."""
    n=len(rewards)
    adv=np.zeros(n,dtype=np.float32)
    running=0.0
    for i in range(n-1,-1,-1):
        if not resolved[i]:
            running=0.0
            continue
        dt=int(indices[i+1]-indices[i]) if i+1<n else 1
        disc=gamma**max(1,dt)
        continuation=(i+1<n and bool(resolved[i+1]))
        bootstrap=values[i+1] if continuation else 0.0
        delta=rewards[i]+disc*bootstrap-values[i]
        running=delta + disc*(lam**max(1,dt))*running if continuation else delta
        adv[i]=running
    return adv,adv+values


class PPOTrader:
    def __init__(self, features:EconomicFeatures, settings:PPOSettings, seed:int):
        self.features=features;self.settings=settings
        # Isolate each trial's actor initialization and rollout RNG from the
        # process-wide Torch RNG (other trials and model inference).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model=HybridActorCritic(features.n_features,settings.hidden_dim)
            self._torch_rng_state=torch.get_rng_state().clone()
        self.opt=torch.optim.Adam(self.model.parameters(),lr=settings.learning_rate)
        # A trader visits the same immutable historical bars many times during
        # PPO rollouts, validation and stress replays. Cache deterministic
        # pre-decision market references, never dynamic account/veto outcomes.
        self._market_context_cache = None

    def snapshot(self):
        return {'weights':{k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
                **self.features.snapshot()}

    def restore(self,snap):
        self.model.load_state_dict(snap['weights'],strict=True)
        self.features.restore(snap)

    def replay(self,obs,spec,indices,cfg,deterministic=True,initial_weights=None,cost_multiplier=1.,m1=None,deadline_monotonic=None):
        if initial_weights is not None:self.restore(initial_weights)
        collected=[]
        def decide(bar,state):
            vec=self.features(bar,state)
            with torch.no_grad():
                act,logp,value,entropy=self.model.act(torch.as_tensor(vec).unsqueeze(0),deterministic)
            collected.append((bar,vec,act,logp.item(),value.item()))
            return to_proposal(act,obs,bar,cfg.intelligence.geometry,cfg.intelligence.position_sizing)
        if (self._market_context_cache is None
                or not self._market_context_cache.compatible(obs, cfg.execution)):
            self._market_context_cache = CausalMarketContextCache(obs, cfg.execution)
        engine=PortfolioAccountingSimulator(cfg.execution,
                    max_risk_per_trade_pct=cfg.intelligence.position_sizing.max_risk_per_trade_pct,
                    max_drawdown_pct=cfg.qualification.max_drawdown_pct)
        m1=m1 or (None,None,None,None)
        def execute_replay():
            return engine.simulate(obs,decide,spec,start_index=int(indices[0]),end_index=int(indices[-1]),
                         cost_multiplier=cost_multiplier,inactivity_penalty_lots=cfg.ppo.penalty_reference_lots,
                         market_context_cache=self._market_context_cache,
                         m1_time_s=m1[0],m1_high=m1[1],m1_low=m1[2],
                         m1_spread_points=m1[3] if len(m1)>3 else None,
                         deadline_monotonic=deadline_monotonic)
        if deterministic:
            sim=execute_replay()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._torch_rng_state)
                try:
                    sim=execute_replay()
                finally:
                    self._torch_rng_state=torch.get_rng_state().clone()
        if len(collected)!=len(sim.decision_events):
            raise AssertionError('Policy decisions and economic ledger desynchronized')
        return sim,collected

    def fit(self,obs,spec,train,val,cfg,seed,m1=None,callback=None,checkpoint_path=None,deadline_monotonic=None):
        best=-float('inf'); best_snapshot=None
        rng=np.random.default_rng(seed)
        start_episode=0
        destination=Path(checkpoint_path) if checkpoint_path is not None else None
        if destination is not None and destination.exists():
            saved=torch.load(destination,map_location='cpu',weights_only=True)
            if (saved.get('format')!='ppo_trial_resume_v1' or
                    saved.get('settings')!=self.settings.__dict__ or
                    int(saved.get('seed',-1))!=seed):
                raise ValueError('Invalid PPO trial checkpoint identity')
            self.model.load_state_dict(saved['weights'],strict=True)
            self.opt.load_state_dict(saved['optimizer'])
            rng.bit_generator.state=saved['rng_numpy']
            self._torch_rng_state=saved['rng_torch'].clone()
            best=float(saved['best_validation_reward'])
            best_snapshot=saved['best_weights']
            start_episode=int(saved['next_episode'])
            if start_episode < 0 or start_episode > self.settings.episodes:
                raise ValueError('Invalid PPO checkpoint episode')
        for episode in range(start_episode,self.settings.episodes):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError('Global training wall-time budget exhausted at episode boundary')
            # A fresh episode starts at the same chronologically bounded training history.
            sim,samples=self.replay(obs,spec,train,cfg,deterministic=False,m1=m1,
                                      deadline_monotonic=deadline_monotonic)
            if samples:
                events=sim.decision_events
                rewards=np.array([e.reward_account_ccy for e in events],dtype=np.float32)
                rewards /= cfg.execution.starting_balance*0.01
                # A trade's terminal cash reward arrives at its CLOSE, not on entry.
                # Semi-Markov PPO discounts that delayed reward by the elapsed bars.
                elapsed=np.array([max(0,(e.resolved_bar_index or e.bar_index)-e.bar_index)
                                  for e in events],dtype=np.float32)
                rewards *= np.power(self.settings.gamma,elapsed)
                old_values=np.array([s[4] for s in samples],dtype=np.float32)
                bars=np.array([s[0] for s in samples],dtype=np.int64)
                # Terminal open positions are right-censored, NOT closed.
                # Their last observable marked-to-market equity is a valid
                # truncated-episode training target. Do not invent an eventual
                # realized outcome, and never access the next data split.
                reward_available=np.array([
                    e.resolved or e.terminal_mark_to_market_account_ccy is not None
                    for e in events],dtype=bool)
                adv,ret=advantage_estimates(rewards,old_values,bars,self.settings.gamma,
                                            self.settings.gae_lambda,reward_available)
                trainable=np.where(reward_available)[0]
                if len(trainable):
                    mean,std=float(adv[trainable].mean()),float(adv[trainable].std())
                    norm=(adv-mean)/(std+1e-6) if len(trainable)>1 else adv
                    states=torch.tensor(np.stack([samples[i][1] for i in trainable]),dtype=torch.float32)
                    dirs=torch.tensor([samples[i][2].direction_index for i in trainable],dtype=torch.long)
                    vals=torch.tensor([[samples[i][2].stop_fraction,samples[i][2].target_fraction,
                                        samples[i][2].risk_fraction] for i in trainable],dtype=torch.float32)
                    old=torch.tensor([samples[i][3] for i in trainable],dtype=torch.float32)
                    targets=torch.tensor(ret[trainable],dtype=torch.float32)
                    advantages=torch.tensor(norm[trainable],dtype=torch.float32)
                    for _ in range(self.settings.ppo_epochs):
                        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                            raise TimeoutError('Global training budget exhausted inside PPO epoch')
                        # On-policy rollout cannot safely be reused once actor drift is large.
                        with torch.no_grad():
                            now_logp, _ = self.model.evaluate_actions(states, dirs, vals)
                            approx_kl = float((old - now_logp).mean().item())
                        if approx_kl > self.settings.target_kl:
                            break
                        order=rng.permutation(len(trainable))
                        for mini in np.array_split(order,max(1,int(np.ceil(len(order)/cfg.research.batch_size)))):
                            mini=torch.as_tensor(mini,dtype=torch.long)
                            logp,ent=self.model.evaluate_actions(states[mini],dirs[mini],vals[mini])
                            _,_,v=self.model.distribution(states[mini])
                            ratio=torch.exp((logp-old[mini]).clamp(-15,15))
                            clipped=torch.clamp(ratio,1-self.settings.clip_ratio,1+self.settings.clip_ratio)
                            loss=-(torch.min(ratio*advantages[mini],clipped*advantages[mini]).mean()
                                   -self.settings.value_coef*torch.nn.functional.mse_loss(v,targets[mini])
                                   +self.settings.entropy_coef*ent.mean())
                            self.opt.zero_grad();loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                            self.opt.step()
            should_validate = (episode == 0 or (episode + 1) % self.settings.validation_interval_episodes == 0
                               or episode + 1 == self.settings.episodes)
            validation = None
            if should_validate:
                validation,_ = self.replay(obs,spec,val,cfg,deterministic=True,m1=m1,
                                              deadline_monotonic=deadline_monotonic)
                # Never backpropagate across validation observations.
                score=validation.learning_reward_account_ccy
                if score > best:
                    best=score
                    best_snapshot=self.snapshot()
                if episode == 0 or (episode+1) % 25 == 0 or episode+1 == self.settings.episodes:
                    from quant_rl.config.logger import log
                    log.info('PPO', f'{spec.name}: episode {episode+1}/{self.settings.episodes}; '
                             f'best_validation_reward={best:+.2f}; PPO epochs/episode={self.settings.ppo_epochs}')
            # Persist after EVERY completed episode, not just validation
            # intervals; a restart cannot silently skip the last 24 updates.
            if destination is not None and episode+1 < self.settings.episodes:
                destination.parent.mkdir(parents=True,exist_ok=True)
                tmp=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
                payload={'format':'ppo_trial_resume_v1','settings':self.settings.__dict__,
                         'seed':seed,'next_episode':episode+1,
                         'weights':self.model.state_dict(),'optimizer':self.opt.state_dict(),
                         'rng_numpy':rng.bit_generator.state,'rng_torch':self._torch_rng_state.clone(),
                         'best_validation_reward':best,'best_weights':best_snapshot}
                try:
                    with tmp.open('xb') as handle:
                        torch.save(payload,handle)
                        handle.flush();os.fsync(handle.fileno())
                    os.replace(tmp,destination)
                finally:
                    tmp.unlink(missing_ok=True)
            if should_validate and callback is not None:
                callback(episode,best,validation)
        if best_snapshot is None:raise RuntimeError('No completed PPO candidate')
        self.restore(best_snapshot)
        if destination is not None:destination.unlink(missing_ok=True)
        return best
