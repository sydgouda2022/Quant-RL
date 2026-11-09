"""LIVE runner: identical MT5 order code for demo and funded accounts.

No learning, no synthetic fallback, no model publication, no new decision while
any broker position/order exists. A write-ahead journal prevents blind retry.
"""
from __future__ import annotations
from quant_rl.config.logger import log
from quant_rl.data.mt5_source import broker_timestamp_utc, broker_tick_age, account_is_demo
from datetime import datetime, timezone
from dataclasses import replace
from quant_rl.execution.operational_safety import (
    kill_switch_engaged, broker_heartbeat_healthy, operator_blackout_active,
)
from pathlib import Path
import json
import hashlib
import math
import time
import numpy as np
import torch
from quant_rl.config.schema import QuantRLConfig
from quant_rl.execution.mt5_broker import MT5Broker
from quant_rl.execution.examiner import PortfolioStateSnapshot,PreTradeExaminer
from quant_rl.execution.market_safety import market_context_asof
from quant_rl.execution.sizing import conversion_rates_asof
from quant_rl.execution.loss_history import consecutive_loss_streak
from quant_rl.intelligence.policy import PPOTrader,PPOSettings,EconomicFeatures
from quant_rl.intelligence.actions import to_proposal
from quant_rl.market.state_builder import ObservationBuilder
from quant_rl.research.evolution import ModelPopulation,architecture_hash,exclusive_run
from quant_rl.data.mt5_source import split_fx_symbol
from quant_rl.data.mt5_source import broker_spec_hash


from quant_rl.execution.state_io import store_json


class LiveSession:
    def __init__(self, cfg:QuantRLConfig, broker:MT5Broker):
        if cfg.execution.max_open_positions != 1:
            raise ValueError('LIVE is single-position; execution.max_open_positions must be 1')
        self.cfg=cfg
        self.broker=broker
        self.population=ModelPopulation(Path(cfg.project_root))
        self.symbol=cfg.live.symbol
        self._loss_anchor_verified=False
        identity=broker.account()
        if getattr(identity, 'trade_mode', None) is None:
            raise PermissionError('MT5 account trading mode unavailable; LIVE is blocked')
        mt5_module=getattr(broker, '_mt5', None)
        if not cfg.live.allow_non_demo_account and not account_is_demo(identity, mt5_module):
            raise PermissionError('LIVE is Demo-only; explicitly enable live.allow_non_demo_account to use a funded account')
        if not cfg.broker_connection.mt5_server_timezone:
            raise ValueError('LIVE requires a verified IANA broker timezone in broker_connection.mt5_server_timezone')
        self.login=int(identity.login)
        self.server=str(getattr(identity,'server',''))
        self.record=self.population.data['symbols'].get(self.symbol)
        if not self.record or self.record.get('origin')!='MT5_TERMINAL' or not math.isfinite(float(self.record.get('current_score',float('-inf')))):
            raise ValueError('LIVE requires an accepted, qualified MT5-trained symbol model')
        if self.record.get('mt5_server_timezone') != cfg.broker_connection.mt5_server_timezone:
            raise ValueError('LIVE model was certified for a different MT5 server timezone; requalify it')
        if self.record.get('account_currency') != str(identity.currency).upper():
            raise ValueError('LIVE account currency differs from trained model')
        if self.record.get('policy_hash') != cfg.policy_hash:
            raise ValueError('LIVE model inference/risk contract changed; retrain or re-certify the model')
        if self.server and self.record.get('broker_server') != self.server:
            raise ValueError('LIVE model trained on a different broker server; re-certification required')
        period=(self.record.get('period') or {})
        try:
            trained_end=datetime.fromisoformat(period['test_end_utc']).timestamp()
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('LIVE model is missing an evaluation timestamp certificate') from exc
        age=time.time()-trained_end
        if not 0 <= age <= cfg.qualification.max_live_model_age_days*86400:
            raise ValueError('LIVE evaluation certificate is stale or future-dated')
        # Validate the weight digest once per session; routine H4 decisions
        # reuse the same immutable trained tensor snapshot.
        self._verified_snapshot=self.population.load(self.record,self.record['arch'])
        self.builder=ObservationBuilder(list(cfg.data.observation_timeframes),
            cfg.data.sequence_lookback_bars,cfg.data.include_technical_indicators)

    def _journal(self, login:int):
        suffix=('_'+hashlib.sha256(self.server.encode()).hexdigest()[:10]) if self.server else ''
        return Path(self.cfg.project_root)/'live_state'/f'{int(login)}{suffix}_{self.symbol}.json'

    def _load_state(self,account):
        path=self._journal(int(account.login))
        if path.exists():
            state=json.loads(path.read_text('utf-8'))
            if (state.get('login')!=int(account.login) or state.get('symbol')!=self.symbol
                    or state.get('server','')!=self.server):
                raise ValueError('LIVE account-state identity mismatch')
            return state
        return {'login':int(account.login),'server':self.server,'symbol':self.symbol,
            'last_bar_time_s':0,'status':'READY','peak_equity':float(account.equity),
            'initial_equity':float(account.equity)}

    def _reconcile_pending(self, state: dict, account) -> dict:
        """Recognize confirmed MT5 outcomes after a crash without guessing absence.

        An unacknowledged order must NEVER be retried automatically. Only an
        authenticated broker order/position/deal with our exact request marker,
        configured magic and symbol can close the pending journal entry.
        """
        marker=str(state.get('request_id') or '')
        if not marker or not marker.startswith('QRL'):
            raise RuntimeError('Pending MT5 request has no valid marker; operator reconciliation required')
        since=max(0,int(state.get('submit_time_s') or state.get('last_bar_time_s') or 0)-120)
        def matches(row):
            return (int(getattr(row,'magic',-1) or -1)==self.cfg.live.magic
                and str(getattr(row,'symbol',''))==self.symbol
                and str(getattr(row,'comment',''))==marker)
        # Broker inquiry failure remains an unresolved pending order; fail closed.
        positions=self.broker.positions()
        orders=self.broker.orders()
        deals=self.broker.recent_deals(since)
        confirmations=[d for d in deals if matches(d)]
        if confirmations:
            state.update(status='FILLED',recovery='MT5_DEAL_HISTORY',
                recovered_deal_ids=[int(getattr(d,'ticket',0) or 0) for d in confirmations])
            store_json(self._journal(account.login),state)
            return {'status':'RECOVERED_FILLED','decision_time_s':int(state['last_bar_time_s'])}
        if any(matches(position) for position in positions):
            state.update(status='FILLED',recovery='MT5_OPEN_POSITION')
            store_json(self._journal(account.login),state)
            return {'status':'RECOVERED_OPEN_POSITION','decision_time_s':int(state['last_bar_time_s'])}
        if any(matches(order) for order in orders):
            # Still outstanding: keep PENDING and re-query next poll.
            return {'status':'BROKER_ORDER_PENDING','decision_time_s':int(state['last_bar_time_s'])}
        raise RuntimeError('Previous order outcome unresolved in MT5 positions/orders/deals; manual reconciliation required')

    def _daily_and_weekly(self,now_s,account):
        # Realized deal cash flows are the broker's recorded profit, swap,
        # commission and fee. No simulated transaction fees in LIVE.
        now=datetime.fromtimestamp(now_s,timezone.utc)
        since_week=int((now.replace(hour=0,minute=0,second=0,microsecond=0)).timestamp())-now.weekday()*86400
        # Deal history used for week/day losses and recent loss streak only.
        # Historical research lookback must not force four-year broker deal scans.
        deals=self.broker.recent_deals(min(now_s - 90*86400,since_week))
        day_s=int(now.replace(hour=0,minute=0,second=0,microsecond=0).timestamp())
        def cash(d):
            return sum(float(getattr(d,k,0.) or 0.) for k in ('profit','commission','swap','fee'))
        # Only deals representing actual trading may enter realized P&L.
        # Deposits, balance transfers, credits and administrative entries are not profits.
        execution_types={getattr(self.broker._mt5,name,None) for name in
                         ('DEAL_TYPE_BUY','DEAL_TYPE_SELL')}
        execution_types.discard(None)
        trading=[d for d in deals if getattr(d,'type',None) in execution_types]
        week=sum(cash(d) for d in trading if broker_timestamp_utc(d.time)>=since_week)
        day=sum(cash(d) for d in trading if broker_timestamp_utc(d.time)>=day_s)
        # With no open positions, equity - realized net gives opening balance
        # for this period, including broker-recorded deposits/withdrawals.
        # The C14 streak is a separate, broker-reconciled lifetime state: a
        # quiet 90-day period must not erase losses prior to that window.
        streak=consecutive_loss_streak(self.broker,root=Path(self.cfg.project_root),
            login=self.login,server=self.server,symbol=self.symbol,
            magic=self.cfg.live.magic,
            verify_history_anchor=not self._loss_anchor_verified)
        self._loss_anchor_verified=True
        return max(0.,float(account.equity)-day),max(0.,float(account.equity)-week),streak

    def _fx_rates(self,spec,account,main_tick):
        base,quote,_=split_fx_symbol(spec.symbol)
        ac=str(account.currency).upper()
        if ac in (base,quote):
            return None
        suffix=spec.symbol[6:]
        prices={spec.symbol:(float(main_tick.bid)+float(main_tick.ask))/2}
        for candidate in (quote+ac,ac+quote,base+ac,ac+base):
            symbol=candidate+suffix
            # MT5 may omit ticks for an unselected Market Watch instrument.
            # Never infer a missing conversion or recycle a stale price.
            if not self.broker._mt5.symbol_select(symbol, True):
                continue
            tick=self.broker._mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
            bid=float(getattr(tick,'bid',0) or 0)
            ask=float(getattr(tick,'ask',0) or 0)
            age=broker_tick_age(int(time.time()),getattr(tick,'time',0))
            if (math.isfinite(bid) and math.isfinite(ask) and bid>0 and ask>=bid
                    and age is not None and 0 <= age <= self.cfg.execution.max_quote_age_seconds):
                prices[symbol]=(bid+ask)/2
        rates=conversion_rates_asof(prices)
        if quote+ac not in rates or base+ac not in rates:
            raise ValueError('Missing broker conversion ticks for live currency accounting')
        return rates

    def step(self):
        """At most one decision per completed bar; safe to call repeatedly."""
        a=self.broker.account()
        if int(a.login)!=self.login or str(getattr(a,'server',''))!=self.server:
            raise ConnectionError('MT5 account changed during LIVE session; restart with the new connection')
        state=self._load_state(a)
        if state.get('status')=='PENDING':
            return self._reconcile_pending(state,a)
        if kill_switch_engaged(Path(self.cfg.project_root)):
            return {'status':'KILL_SWITCH_ENGAGED','decision_time_s':int(state['last_bar_time_s'])}
        # While an order or position is open, do not fetch historical features,
        # run the policy, or produce inactivity penalties.
        if self.broker.positions() or self.broker.orders():
            updated_peak=max(float(state.get('peak_equity',a.equity)),float(a.equity))
            if updated_peak>float(state.get('peak_equity',0)):
                state['peak_equity']=updated_peak
                store_json(self._journal(a.login),state)
            return {'status':'POSITION_OR_ORDER_OPEN',
                    'decision_time_s':int(state['last_bar_time_s'])}
        from quant_rl.data.time_alignment import resample_ohlcv_utc,timeframe_to_seconds
        tf_s=timeframe_to_seconds(self.cfg.research.timeframe)
        quick=self.broker.fetch_bars(self.symbol,'H1',max(16,tf_s//3600*3+4))
        ready=resample_ohlcv_utc(quick,self.cfg.research.timeframe,source_timeframe="H1")
        if not ready:
            raise ConnectionError('Insufficient completed MT5 candles to establish decision time')
        latest_closed=int(ready[-1].time_s)+tf_s
        if latest_closed<=int(state['last_bar_time_s']):
            return {'status':'ALREADY_EVALUATED','decision_time_s':latest_closed}
        bars=self.broker.closed_bars(self.symbol)
        spec=self.broker.get_symbol_spec(self.symbol,str(a.currency))
        if self.record.get('broker_spec_hash') != broker_spec_hash(spec):
            raise ValueError('LIVE broker instrument economics changed; research certification required')
        live_tick=self.broker.tick(self.symbol)
        quote_spread=(float(live_tick.ask)-float(live_tick.bid))/spec.point
        if not math.isfinite(quote_spread) or quote_spread<=0:
            raise ConnectionError('Cannot construct policy features without broker spread evidence')
        obs=self.builder.build_dataset(base_bars=bars,symbol=self.symbol,
            primary_timeframe=self.cfg.research.timeframe,
            volatility_estimator=self.cfg.research.volatility_estimator,
            volatility_period=self.cfg.research.volatility_period,
            spec=spec,bar_price_side='BID',fallback_spread_points=quote_spread)
        index=obs.num_bars-1
        bar_time=int(obs.decision_time_s[index])
        now_s=int(time.time())
        if bar_time>now_s:
            raise ValueError('Incomplete future live observation')
        if now_s-bar_time>self.cfg.live.max_entry_delay_seconds:
            state.update(last_bar_time_s=bar_time,status='AWAITING_NEXT_BAR')
            store_json(self._journal(a.login),state)
            return {'status':'AWAITING_NEXT_BAR','decision_time_s':bar_time}
        if bar_time<=int(state['last_bar_time_s']):
            return {'status':'ALREADY_EVALUATED','decision_time_s':bar_time}
        # A position that was open at candle close but closed afterward must
        # not authorize a retroactive decision on that same candle.
        close_types={getattr(self.broker._mt5,k,None) for k in
                     ('DEAL_ENTRY_OUT','DEAL_ENTRY_OUT_BY','DEAL_ENTRY_INOUT')}
        close_types.discard(None)
        recent=self.broker.recent_deals(bar_time)
        if any(broker_timestamp_utc(getattr(d,'time',0))>=bar_time and
               getattr(d,'entry',None) in close_types for d in recent):
            state.update(last_bar_time_s=bar_time,status='AWAITING_NEXT_BAR_AFTER_EXIT')
            store_json(self._journal(a.login),state)
            return {'status':'AWAITING_NEXT_BAR_AFTER_EXIT','decision_time_s':bar_time}
        tick=self.broker.tick(self.symbol)
        tick_age=broker_tick_age(now_s,getattr(tick,'time',0))
        if tick_age is None or not 0<=tick_age<=self.cfg.execution.max_quote_age_seconds:
            raise ConnectionError('Stale live quote')
        effective_exec=self.cfg.execution.model_copy(update={
            'starting_balance':float(a.balance),'account_currency':str(a.currency),
            'account_leverage':float(a.leverage)})
        effective_cfg=self.cfg.model_copy(update={'execution':effective_exec})
        profile=self.record.get('feature_profile',{})
        settings=PPOSettings(**self.record['settings'])
        torch.set_num_threads(self.cfg.ppo.cpu_threads)
        feature=EconomicFeatures(obs,np.arange(max(0,index-64),index+1),
            max(1.,float(state.get('initial_equity',a.equity))),symbol_spec=spec,account_currency=str(a.currency),
            min_risk_pct=self.cfg.intelligence.position_sizing.min_risk_per_trade_pct,
            max_risk_pct=self.cfg.intelligence.position_sizing.max_risk_per_trade_pct,
            feature_group=profile.get('feature_group','all'),
            sequence_bars=profile.get('sequence_bars',self.cfg.data.sequence_lookback_bars))
        # Learned architecture belongs to the original research configuration;
        # changing the connected account currency must not rewrite its identity.
        arch=architecture_hash(self.cfg,feature,settings)
        if arch != self.record['arch']:
            raise ValueError('LIVE model architecture differs from certified policy')
        snapshot=self._verified_snapshot
        policy=PPOTrader(feature,settings,seed=self.cfg.research.seed)
        policy.restore(snapshot)
        day_start,week_start,streak=self._daily_and_weekly(now_s,a)
        peak=max(float(state.get('peak_equity',a.equity)),float(a.equity))
        state['peak_equity']=peak
        portfolio=PortfolioStateSnapshot(current_time_s=bar_time,
            balance=float(a.balance),equity=float(a.equity),peak_equity=peak,
            day_start_equity=day_start,week_start_equity=week_start,
            consecutive_losses=streak,last_bar_time_s=bar_time,
            timeframe_seconds=obs.timeframe_seconds,
            broker_healthy=broker_heartbeat_healthy(self.broker,login=self.login,server=self.server),
            kill_switch_engaged=kill_switch_engaged(Path(self.cfg.project_root)),
            economic_blackout_active=operator_blackout_active(
                Path(self.cfg.project_root), self.cfg.live.economic_blackout_active),
            execution_time_s=now_s)
        fx_rates=self._fx_rates(spec,a,tick)
        with torch.no_grad():
            action,*_=policy.model.act(torch.as_tensor(feature(index,portfolio,fx_rates)).unsqueeze(0),True)
        proposal=to_proposal(action,obs,index,self.cfg.intelligence.geometry,self.cfg.intelligence.position_sizing)
        state['last_bar_time_s']=bar_time
        state['action']=proposal.action_name
        if proposal.action_name=='NO_TRADE':
            state['status']='NO_TRADE'
            store_json(self._journal(a.login),state)
            return {'status':'NO_TRADE','decision_time_s':bar_time}
        market=market_context_asof(obs,index,lookback_bars=effective_exec.market_history_bars,
            min_history_bars=effective_exec.market_min_history_bars,
            candidate_windows=effective_exec.market_window_candidates,
            regime_shift_ratio=effective_exec.market_regime_shift_ratio)
        mid=(float(tick.bid)+float(tick.ask))/2
        spread_pips=(float(tick.ask)-float(tick.bid))/spec.pip_size
        examiner=PreTradeExaminer(effective_exec,
            max_risk_per_trade_pct=self.cfg.intelligence.position_sizing.max_risk_per_trade_pct,
            max_drawdown_pct=self.cfg.qualification.max_drawdown_pct,
            server_timezone=self.cfg.broker_connection.mt5_server_timezone)
        verdict=examiner.inspect(proposal,portfolio,spec,mid_price=mid,
            spread_pips=spread_pips,volatility_price=float(obs.volatility_price[index]),
            market_context=market,fx_rates=fx_rates)
        if not verdict.approved:
            state['status']='VETOED';state['veto_code']=verdict.veto_code
            state['veto_reasons']=dict(verdict.failed_check_details)
            store_json(self._journal(a.login),state)
            return {'status':'VETOED','veto_code':verdict.veto_code,'decision_time_s':bar_time}
        # Re-read broker status immediately before preparing an order.
        if self.broker.positions() or self.broker.orders():
            state['status']='POSITION_OR_ORDER_OPEN'
            store_json(self._journal(a.login),state)
            return {'status':'POSITION_OR_ORDER_OPEN','decision_time_s':bar_time}
        a2=self.broker.account()
        if (int(a.login)!=int(a2.login) or
                str(getattr(a2,'server',''))!=self.server):
            raise ConnectionError('MT5 account/server changed during decision')
        if (str(a.currency)!=str(a2.currency) or int(a.leverage)!=int(a2.leverage)
                or abs(float(a.equity)-float(a2.equity))>1e-6):
            state['status']='ACCOUNT_STATE_CHANGED'
            store_json(self._journal(a.login),state)
            return {'status':'ACCOUNT_STATE_CHANGED','decision_time_s':bar_time}
        fresh=self.broker.tick(self.symbol)
        fresh_age=broker_tick_age(int(time.time()),getattr(fresh,'time',0))
        if (fresh_age is None or not 0 <= fresh_age <= self.cfg.execution.max_quote_age_seconds or
                abs(float(fresh.ask)-float(tick.ask))>spec.point*self.cfg.execution.max_slippage_points or
                abs(float(fresh.bid)-float(tick.bid))>spec.point*self.cfg.execution.max_slippage_points):
            state['status']='QUOTE_MOVED'
            store_json(self._journal(a.login),state)
            return {'status':'QUOTE_MOVED','decision_time_s':bar_time}
        new_mid=(float(fresh.bid)+float(fresh.ask))/2
        # The FX conversion snapshot is a point-in-time risk input, not a
        # constant for the entire decision. Re-fetch it after the executable
        # quote and account have been refreshed. A missing conversion fails
        # closed rather than using the old pip/margin estimate.
        fresh_fx_rates=self._fx_rates(spec,a2,fresh)
        # Re-read both active safety signals after account/quote refresh and
        # immediately before any broker-side preflight or order submission.
        refreshed_portfolio=replace(portfolio,
            broker_healthy=broker_heartbeat_healthy(self.broker,login=self.login,server=self.server),
            kill_switch_engaged=kill_switch_engaged(Path(self.cfg.project_root)),
            economic_blackout_active=operator_blackout_active(
                Path(self.cfg.project_root), self.cfg.live.economic_blackout_active),
            execution_time_s=now_s)
        fresh_verdict=examiner.inspect(proposal,refreshed_portfolio,spec,mid_price=new_mid,
            spread_pips=(float(fresh.ask)-float(fresh.bid))/spec.pip_size,
            volatility_price=float(obs.volatility_price[index]),
            market_context=market,fx_rates=fresh_fx_rates)
        if not fresh_verdict.approved or fresh_verdict.approved_volume_lots!=verdict.approved_volume_lots:
            state['status']='VETOED_REFRESH'
            state['veto_code']=fresh_verdict.veto_code
            store_json(self._journal(a.login),state)
            return {'status':'VETOED_REFRESH','decision_time_s':bar_time}
        direction=proposal.action_name
        price=float(fresh.ask if direction=='BUY' else fresh.bid)
        # Normalize to the broker's executable tick increment, rounding stops
        # toward entry and targets toward entry (never widening policy risk).
        step=float(spec.trade_tick_size)
        if not math.isfinite(step) or step<=0:
            raise ValueError('Invalid broker trade tick increment')
        def snapped(value,round_up):
            units=value/step
            return round((math.ceil(units-1e-9) if round_up else math.floor(units+1e-9))*step,10)
        sl=snapped(price-proposal.stop_distance_price,direction=='BUY') if direction=='BUY' else snapped(price+proposal.stop_distance_price,False)
        tp=snapped(price+proposal.target_distance_price,False) if direction=='BUY' else snapped(price-proposal.target_distance_price,True)
        volume=fresh_verdict.approved_volume_lots
        m=self.broker._mt5
        side=m.ORDER_TYPE_BUY if direction=='BUY' else m.ORDER_TYPE_SELL
        estimated_loss=m.order_calc_profit(side,self.symbol,volume,price,sl)
        slippage_price=self.cfg.execution.max_slippage_points*spec.point
        # Stress both adverse entry movement and stop execution beyond its trigger.
        # Config validation ensures broker deviation cannot exceed this bound.
        worst_entry=price+slippage_price if direction=='BUY' else price-slippage_price
        worst_sl=sl-slippage_price if direction=='BUY' else sl+slippage_price
        stressed_loss=m.order_calc_profit(side,self.symbol,volume,worst_entry,worst_sl)
        margin=m.order_calc_margin(side,self.symbol,volume,worst_entry)
        estimated_commission=volume*self.cfg.execution.commission_per_lot_round_trip
        if (margin is None or estimated_loss is None or stressed_loss is None
                or not all(math.isfinite(float(v)) for v in (margin,estimated_loss,stressed_loss))
                or estimated_loss>=0 or stressed_loss>=0
                or -float(stressed_loss)+estimated_commission>float(a2.equity)*proposal.requested_risk_fraction+1e-6
                or float(margin)>float(a2.margin_free)):
            state['status']='VETOED_BROKER_CALC'
            store_json(self._journal(a.login),state)
            return {'status':'VETOED_BROKER_CALC','decision_time_s':bar_time}
        comment=f'QRL{bar_time}'
        request=self.broker.order_request(symbol=self.symbol,direction=direction,
            volume=volume,price=price,sl=sl,tp=tp,
            deviation=self.cfg.live.order_deviation_points,
            magic=self.cfg.live.magic,comment=comment)
        # Persist the intent BEFORE calling order_send; a crash or timeout
        # never causes an automatic duplicate submission on restart.
        state.update(status='PENDING',request_id=comment,submit_time_s=int(time.time()))
        store_json(self._journal(a.login),state)
        if (kill_switch_engaged(Path(self.cfg.project_root)) or
                operator_blackout_active(Path(self.cfg.project_root),
                                         self.cfg.live.economic_blackout_active) or
                not broker_heartbeat_healthy(
                    self.broker,login=self.login,server=self.server)):
            # Intent is already journaled as PENDING; no broker order was sent.
            state.update(status='HALTED_BEFORE_SEND')
            store_json(self._journal(a.login),state)
            return {'status':'HALTED_BEFORE_SEND','decision_time_s':bar_time}
        outcome=self.broker.submit(request)
        state.update(status=outcome['status'] if outcome['status'] in ('FILLED','CHECK_REJECTED') else 'PENDING',result=outcome)
        store_json(self._journal(a.login),state)
        return {'decision_time_s':bar_time,**outcome}


def run_live_session(cfg:QuantRLConfig):
    """Runs at completed-bar cadence in an IDE; Ctrl+C terminates cleanly."""
    broker=MT5Broker(cfg.broker_connection,
                     allow_non_demo_account=cfg.live.allow_non_demo_account)
    try:
        broker.connect()
        account=broker.account()
        # Exactly one Quant-RL LIVE process per connected account. Separate
        # demo/funded accounts naturally receive independent locks.
        with exclusive_run(Path(cfg.project_root)):
            session=LiveSession(cfg,broker)
            while True:
                result=session.step()
                if result['status']!='ALREADY_EVALUATED':
                    log.info('MT5 LIVE', str(result))
                time.sleep(cfg.live.poll_seconds)
    except KeyboardInterrupt:
        log.warning('MT5 LIVE', 'Stopped by operator')
    finally:
        broker.close()
