"""A compact actor-critic with a genuine conditional hybrid action distribution."""
from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Beta, Categorical
from quant_rl.intelligence.actions import HybridAction


class HybridActorCritic(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features,hidden_dim),nn.Tanh(),
                                     nn.Linear(hidden_dim,hidden_dim),nn.Tanh())
        self.direction = nn.Linear(hidden_dim,3)
        self.beta_params = nn.Linear(hidden_dim,6)
        self.value = nn.Linear(hidden_dim,1)

    def distribution(self, state: torch.Tensor):
        h=self.encoder(state)
        dirs=Categorical(logits=self.direction(h))
        concentrations=torch.nn.functional.softplus(self.beta_params(h)) + 1.01
        geometry=Beta(concentrations[...,:3],concentrations[...,3:])
        return dirs,geometry,self.value(h).squeeze(-1)

    def act(self, state: torch.Tensor, deterministic: bool = False):
        ds,gs,v=self.distribution(state)
        direction=ds.probs.argmax(-1) if deterministic else ds.sample()
        vals=gs.mean if deterministic else gs.sample()
        # Reuse the same distribution as the sampled action. Re-encoding the
        # state for evaluate_actions here doubles every event-driven inference.
        active=(direction != 0).to(state.dtype)
        logp=ds.log_prob(direction)+active*gs.log_prob(vals).sum(-1)
        entropy=ds.entropy()+active*gs.entropy().sum(-1)
        result=HybridAction(int(direction.item()),*(float(x) for x in vals[0].tolist()))
        return result,logp.squeeze(0),v.squeeze(0),entropy.squeeze(0)

    def evaluate_actions(self,state,directions,geometry):
        d,g,_=self.distribution(state)
        dirs=directions.long().view(-1)
        geom=geometry.view(-1,3).clamp(1e-6,1-1e-6)
        active=(dirs != 0).to(state.dtype)
        logp=d.log_prob(dirs)+active*g.log_prob(geom).sum(-1)
        entropy=d.entropy()+active*g.entropy().sum(-1)
        return logp,entropy
