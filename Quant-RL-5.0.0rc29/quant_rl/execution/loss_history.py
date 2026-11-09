"""Broker-backed, account-bound loss-streak checkpoint for LIVE C14.

The broker deal ledger is the authority. The local checkpoint only avoids
rescanning the complete available MT5 trade history at every decision. Missing
history for a previously witnessed broker deal fails closed.
"""
from __future__ import annotations

from pathlib import Path
import math
from quant_rl.execution.state_io import store_json
import hashlib
import json
import time
from quant_rl.data.mt5_source import broker_timestamp_utc


def consecutive_loss_streak(broker, *, root: Path, login: int, server: str,
                            symbol: str, magic: int,
                            verify_history_anchor: bool = True) -> int:
    suffix=hashlib.sha256(server.encode()).hexdigest()[:10]
    path=Path(root)/'live_state'/f'{login}_{suffix}_{symbol}_loss_history.json'
    identity={'version':3,'timestamp_basis':'MT5_UTC','login':int(login),'server':server,
              'symbol':symbol,'magic':int(magic)}
    if path.is_file():
        prior=json.loads(path.read_text('utf-8'))
        if not isinstance(prior,dict):
            raise RuntimeError('LIVE loss-history certificate identity mismatch; broker verification required')
        if prior.get('version') == 2:
            legacy={
                'version':2,'login':int(login),'server':server,'symbol':symbol,
                'magic':int(magic),'offset_seconds':0,
            }
            if any(prior.get(k)!=v for k,v in legacy.items()):
                raise RuntimeError('Legacy nonzero-offset loss history cannot be migrated safely; reconcile broker history before LIVE')
            # A zero-offset checkpoint already stored raw MT5 UTC epochs. It is
            # safe to upgrade its identity without changing the witnessed data.
            prior={**prior,**identity}
            store_json(path,prior)
        if any(prior.get(k)!=v for k,v in identity.items()):
            raise RuntimeError('LIVE loss-history certificate identity mismatch; broker verification required')
        last_amount=float(prior['last_exit_amount'])
        if not math.isfinite(last_amount):
            raise RuntimeError('Invalid LIVE loss-history cash anchor')
        last_time=int(prior['last_time_s'])
        last_ticket=int(prior['last_ticket'])
        streak=int(prior['streak'])
        last_checked_s=int(prior.get('last_checked_s',0))
        if last_time < 0 or last_ticket < 0 or streak < 0 or last_checked_s < 0:
            raise RuntimeError('Invalid LIVE loss-history checkpoint')
    else:
        last_time=last_ticket=streak=last_checked_s=0
        last_amount=0.0
    # Initial and post-restart calls verify the prior broker exit anchor. In a
    # running session, the verified in-memory checkpoint only needs new deals
    # since its last poll (with a 24-hour late-arrival overlap).
    if verify_history_anchor:
        since=max(0,last_time-120) if last_ticket else 0
    else:
        since=max(0,last_checked_s-86400)
    deals=broker.recent_deals(since)
    m=broker._mt5
    closing={getattr(m,name,None) for name in ('DEAL_ENTRY_OUT','DEAL_ENTRY_OUT_BY','DEAL_ENTRY_INOUT')}
    closing.discard(None)
    execution={getattr(m,name,None) for name in ('DEAL_TYPE_BUY','DEAL_TYPE_SELL')}
    execution.discard(None)
    entries=[]
    for deal in deals:
        if (str(getattr(deal,'symbol',''))!=symbol or
                int(getattr(deal,'magic',-1) or -1)!=magic or
                getattr(deal,'entry',None) not in closing or
                getattr(deal,'type',None) not in execution):
            continue
        ticket=int(getattr(deal,'ticket',0) or 0)
        time_s=broker_timestamp_utc(getattr(deal,'time',0))
        if ticket<=0 or time_s<=0:
            raise RuntimeError('Broker exit has no verifiable ticket/timestamp; C14 cannot be certified')
        amount=sum(float(getattr(deal,k,0.) or 0.) for k in ('profit','commission','swap','fee'))
        if not math.isfinite(amount):
            raise RuntimeError('Nonfinite broker exit amount; C14 cannot be certified')
        entries.append((time_s,ticket,amount))
    entries.sort()
    if verify_history_anchor and last_ticket:
        anchor=next((amount for t,tid,amount in entries if (t,tid)==(last_time,last_ticket)),None)
        if anchor is None:
            raise RuntimeError('Cannot verify previous closing deal in current MT5 history; C14 fails closed')
        if not math.isclose(anchor,last_amount,rel_tol=0.0,abs_tol=1e-7):
            raise RuntimeError('Previously witnessed MT5 exit cash changed; C14 fails closed')
    for time_s,ticket,amount in entries:
        if (time_s,ticket) <= (last_time,last_ticket):
            continue
        if amount<0:
            streak+=1
        elif amount>0:
            streak=0
        # A zero-net exit cannot reset a losing streak.
        last_time,last_ticket=time_s,ticket
        last_amount=amount
    store_json(path,{**identity,'last_time_s':last_time,'last_ticket':last_ticket,
                     'last_exit_amount':last_amount,
                     'last_checked_s':int(time.time()),'streak':streak})
    return streak
