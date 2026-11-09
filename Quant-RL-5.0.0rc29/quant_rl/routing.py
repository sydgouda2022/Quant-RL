"""Automatically select LIVE or RESEARCH from actual MT5 market/account state.

There are only two operating modes. Unavailable MT5/data is an error, not a
silent fallback to simulations or stale files.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import time
from quant_rl.data.mt5_source import MT5DataSource, broker_tick_age, clock_offset_diagnostic, account_is_demo
from quant_rl.data.mt5_history import MT5History
from quant_rl.data.rolling_window import select_rolling_bars
from quant_rl.market.state_builder import ObservationBuilder
from quant_rl.research.evolution import ModelPopulation, chronological_split
from quant_rl.config.logger import log
from quant_rl.execution.trading_calendar import weekly_fx_entry_blocked


@dataclass(frozen=True)
class RoutingDecision:
    mode: str
    reason: str
    targets: tuple[str,...]
    market_open: bool
    live_symbol: str | None = None
    account_currency: str = "USD"


def _existing_model(pop, symbol, account_currency=None, mt5_server_timezone=None,
                    *, cfg=None, broker_server="", now_s=None):
    """A manifest entry alone is not a model: check provenance and checksum."""
    rec=pop.data['symbols'].get(symbol)
    if not rec or rec.get('origin') != 'MT5_TERMINAL':
        return False
    if account_currency is not None and rec.get('account_currency') != account_currency:
        return False
    # Changing the certified broker-clock basis requires re-evaluation.
    if rec.get('mt5_server_timezone') != mt5_server_timezone:
        return False
    if cfg is not None:
        if rec.get('policy_hash') != cfg.policy_hash:
            return False
        if broker_server and rec.get('broker_server') != broker_server:
            return False
        try:
            end=datetime.fromisoformat(rec['period']['test_end_utc']).timestamp()
            age=(time.time() if now_s is None else now_s)-end
            if not 0 <= age <= cfg.qualification.max_live_model_age_days*86400:
                return False
        except (KeyError,TypeError,ValueError):
            return False
    try:
        if not math.isfinite(float(rec.get('current_score',float('-inf')))):
            return False
        pop.verify_artifact(rec)
        return True
    except (OSError,ValueError,KeyError,RuntimeError):
        return False


def _qualified_history(cfg, symbol, frozen):
    """Qualification before PPO starts; only actual freshly requested MT5 data."""
    if frozen.manifest.data_origin != 'MT5_TERMINAL':
        return False,'Non-MT5 data'
    if (frozen.manifest.verified_spread_fraction < 1.0 and
            not (getattr(frozen,'fallback_spread_points',None) is not None and frozen.fallback_spread_points > 0)):
        return False,'Missing MT5 historical spread and valid current broker spread quote'
    try:
        bars,start,end=select_rolling_bars(frozen.bars,frozen.timeframe,cfg.research.lookback_years,
                                    now_utc_s=int(time.time()))
        builder=ObservationBuilder(list(cfg.data.observation_timeframes),
                cfg.data.sequence_lookback_bars,cfg.data.include_technical_indicators)
        obs=builder.build_dataset(base_bars=bars,symbol=symbol,
                primary_timeframe=cfg.research.timeframe,
                volatility_estimator=cfg.research.volatility_estimator,
                volatility_period=cfg.research.volatility_period,spec=frozen.spec,
                bar_price_side=frozen.manifest.bar_price_side,
                fallback_spread_points=(frozen.fallback_spread_points
                                        if getattr(frozen,"fallback_spread_points",None) is not None else float("nan")))
        _,_,final=chronological_split(obs.time_s,max(40,cfg.data.sequence_lookback_bars),
                                      window_bounds=(start,end))
        test_days=(int(obs.decision_time_s[final[-1]]) - int(obs.time_s[final[0]]))/86400
        if test_days < cfg.qualification.minimum_evaluation_calendar_days:
            return False,'Insufficient final evaluation history'
        return True,'Qualified'
    except (ValueError,IndexError,AttributeError) as exc:
        return False,f'Insufficient completed history: {exc}'


def choose_mode(cfg, *, source_factory=None, history=None, population=None, now_s=None,
                preload_research=False):
    """Inspect current MT5 state; fetch *only needed* research histories.

    Routing does not open or submit an order. The caller selects a session
    and can pass `datasets` directly to prevent duplicate MT5 downloads.
    """
    ts=int(time.time() if now_s is None else now_s)
    root=Path(cfg.project_root).resolve()
    pop=population or ModelPopulation(root)
    store=history or MT5History(cfg)
    if source_factory is None:
        source_factory=lambda:MT5DataSource(login=cfg.broker_connection.mt5_login,
            password=cfg.broker_connection.mt5_password,
            server=cfg.broker_connection.mt5_server,path=cfg.broker_connection.mt5_path)
    source=source_factory()
    log.info('ROUTER', 'Connecting to MT5 and inspecting configured symbols')
    try:
        source.connect()
        m=source._mt5
        term=m.terminal_info()
        account=m.account_info()
        if term is None or not bool(term.connected) or account is None:
            raise ConnectionError('MT5 terminal/account unavailable; no historical fallback permitted')
        if getattr(account, 'trade_mode', None) is None:
            raise PermissionError('MT5 account trading mode unavailable; LIVE is blocked')
        if not cfg.live.allow_non_demo_account and not account_is_demo(account, m):
            raise PermissionError('LIVE is Demo-only; explicitly enable live.allow_non_demo_account to use a funded account')
        # The configured set is authoritative; do not discover unrelated assets.
        supported=[]; live_quotes=[]; can_open=[]
        clock_warning_emitted=False
        for symbol in cfg.research.symbols:
            info=m.symbol_info(symbol)
            if info is None:
                continue
            try:
                if not bool(m.symbol_select(symbol,True)):
                    continue
                # A symbol can be supported on weekends although its current
                # trade_mode temporarily forbids order opening.
                if float(info.point)<=0 or float(info.trade_contract_size)<=0:
                    continue
            except (AttributeError,ValueError,TypeError):
                continue
            supported.append(symbol)
            trade_mode=int(getattr(info,'trade_mode',0))
            if trade_mode in (1,2,4):
                can_open.append(symbol)
            tick=m.symbol_info_tick(symbol)
            tick_timestamp=int(getattr(tick,'time',0) or 0) if tick is not None else 0
            raw_age=(ts-tick_timestamp) if tick_timestamp>0 else None
            tick_age=broker_tick_age(ts,tick_timestamp)
            bid=float(getattr(tick,'bid',0) or 0) if tick is not None else 0.0
            ask=float(getattr(tick,'ask',0) or 0) if tick is not None else 0.0
            quote_valid=math.isfinite(bid) and math.isfinite(ask) and bid>0 and ask>=bid
            is_fresh=(tick_age is not None and 0<=tick_age<=cfg.execution.max_quote_age_seconds and quote_valid)
            if raw_age is not None and raw_age < -5 and not clock_warning_emitted:
                log.warning('MT5 CLOCK', f'{symbol}: {clock_offset_diagnostic(ts,tick_timestamp)}. '
                            'A future-dated MT5 UTC timestamp remains ineligible for LIVE.')
                clock_warning_emitted=True
            log.info('ROUTER', f'{symbol}: supported; trade_mode={trade_mode}; '
                     f'tick_age={tick_age if tick_age is not None else "unavailable"}s; '
                     f'raw_tick_age={raw_age if raw_age is not None else "unavailable"}s; '
                     f'server_timezone={cfg.broker_connection.mt5_server_timezone or "UNCONFIGURED"}; '
                     f'quote={"valid" if quote_valid else "missing/invalid"}; '
                     f'fresh={is_fresh}')
            if trade_mode in (1,2,4) and is_fresh:
                live_quotes.append(symbol)
        if not supported:
            raise RuntimeError('No configured symbols supported by the connected MT5 broker')
        account_currency=str(account.currency).upper()
        if not account_currency:
            raise ConnectionError("Connected MT5 account currency missing")
        # A broker account may not use the configuration's default currency.
        # All policy-certificate comparisons must use the connected account.
        effective_cfg=cfg.model_copy(update={'execution':cfg.execution.model_copy(
            update={'account_currency':account_currency})})
        live_quote_available=bool(live_quotes)
        server_timezone=cfg.broker_connection.mt5_server_timezone
        weekly_block=(weekly_fx_entry_blocked(
            ts, server_timezone=server_timezone,
            friday_cutoff_hour=cfg.execution.friday_cutoff_server_hour,
            sunday_open_hour=cfg.execution.sunday_open_server_hour,
        ) if server_timezone else False)
        # The generic Friday/Saturday/Sunday schedule takes precedence even
        # if a broker keeps streaming quotes. It blocks entries, not positions.
        market_open=live_quote_available and bool(server_timezone) and not weekly_block
        # The configured weekly FX schedule is the only calendar-based closure
        # classification. A weekday/holiday quote outage remains UNKNOWN.
        confirmed_closed=weekly_block
    finally:
        source.close()

    missing=tuple(symbol for symbol in can_open
                  if not _existing_model(pop,symbol,account_currency,
                                         cfg.broker_connection.mt5_server_timezone,
                                         cfg=effective_cfg,broker_server=str(getattr(account,'server','') or ''),now_s=ts))
    # Missing-model research does NOT require a streaming quote. It requires
    # fresh historical retrieval directly from MT5 and adequate usable bars.
    # A weekday quote outage must never be silently classified as a closure.
    if weekly_block:
        targets=tuple(supported)
    elif market_open:
        targets=missing
    elif confirmed_closed:
        targets=tuple(supported)
    elif missing:
        targets=missing
        log.warning('ROUTER',
                    'No fresh live quotes; market closure not confirmed. '
                    'Attempting missing-model RESEARCH using freshly acquired MT5 history only; LIVE is blocked.')
    else:
        if not cfg.broker_connection.mt5_server_timezone:
            raise ConnectionError('LIVE blocked: set the verified IANA broker timezone in '
                                  'broker_connection.mt5_server_timezone before enabling LIVE')
        raise ConnectionError('No fresh MT5 ticks; market closure unconfirmed and no missing-model '
                              'research is required. Check MT5 terminal quotes, Market Watch, '
                              'account/server connection, and UTC tick timestamps; LIVE is blocked.')
    route_reason=('WEEKLY_SCHEDULE' if weekly_block else
                  'MARKET_CLOSED' if confirmed_closed else 'MISSING_MODELS')
    log.info('ROUTER', f'live_quote_available={live_quote_available}; weekly_entry_block={weekly_block}; '
             f'market_closed={confirmed_closed}; supported={supported}; '
             f'missing_models={list(missing)}; research_targets={list(targets)}')
    # Production routing does not hold seven four-year M1 datasets in RAM.
    # Research fetches and qualifies each symbol immediately before learning.
    if targets and not preload_research:
        reason=route_reason
        log.info('ROUTER',f'RESEARCH: {reason}; candidates={list(targets)}; '
                 'historical qualification deferred per symbol')
        return RoutingDecision('RESEARCH',reason,tuple(targets),market_open,
                               account_currency=account_currency),{},{}
    datasets={}; qualified=[]; rejected={}
    for symbol in targets:
        try:
            # Every attempted research read is fresh from MT5, never a saved NPZ.
            frozen=store.fetch(symbol)
            ok,reason=_qualified_history(cfg,symbol,frozen)
            if ok:
                datasets[symbol]=frozen;qualified.append(symbol)
                coverage=frozen.manifest.verified_spread_fraction
                if coverage < cfg.qualification.historical_spread_warning_fraction:
                    log.warning('MT5 DATA', f'{symbol}: historical spread only {coverage:.1%} reported; '
                                f'using MT5-derived {frozen.manifest.fallback_spread_source} '
                                f'({frozen.fallback_spread_points} points) for missing observations; '
                                'historical execution costs are estimated')
                log.success('ROUTER', f'{symbol}: qualified for RESEARCH')
            else:
                rejected[symbol]=reason
                log.warning('ROUTER', f'{symbol}: skipped: {reason}')
        except (ValueError,ConnectionError,RuntimeError,TypeError) as exc:
            rejected[symbol]=str(exc)
            log.warning('ROUTER', f'{symbol}: {type(exc).__name__}: {exc}')
    if qualified:
        why=route_reason
        return RoutingDecision('RESEARCH',why,tuple(qualified),market_open,account_currency=account_currency),datasets,rejected
    if not market_open:
        raise RuntimeError(('Market closed' if confirmed_closed else 'LIVE unavailable (stale MT5 ticks)') +
                           '; no selected symbols have sufficient freshly retrieved MT5 history: '+str(rejected))
    ordered=[]
    champion=pop.data.get('champion')
    for item in (champion,cfg.live.symbol,*live_quotes):
        if item is not None and item not in ordered:
            ordered.append(item)
    live_symbol=next((item for item in ordered if item in live_quotes and
                      _existing_model(pop,item,account_currency,
                                      cfg.broker_connection.mt5_server_timezone,
                                      cfg=effective_cfg,broker_server=str(getattr(account,'server','') or ''),now_s=ts)),None)
    if live_symbol is None:
        raise RuntimeError('No qualified LIVE model with a fresh MT5 quote; '+str(rejected))
    return RoutingDecision('LIVE','MARKET_OPEN_MODELS_READY',(),True,live_symbol,account_currency),{},rejected


def run_automatic_session(cfg):
    decision,datasets,rejected=choose_mode(cfg)
    log.info('ROUTER', f'{decision.mode}: {decision.reason}; qualified={list(decision.targets)}; skipped={rejected}')
    active_execution=cfg.execution.model_copy(update={'account_currency':decision.account_currency})
    active_cfg=cfg.model_copy(update={'execution':active_execution})
    if decision.mode=='RESEARCH':
        from quant_rl.research.evolution import run_research_session
        return run_research_session(active_cfg,targets=decision.targets,
                                    preloaded=datasets,reason=decision.reason)
    from quant_rl.execution.live import run_live_session
    selected=active_cfg.live.model_copy(update={'symbol':decision.live_symbol})
    return run_live_session(active_cfg.model_copy(update={'live':selected}))
