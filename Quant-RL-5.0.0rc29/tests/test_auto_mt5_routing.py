"""No terminal required: broker protocol fixtures validate routing; no live orders."""
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from quant_rl.config.schema import QuantRLConfig
from quant_rl.routing import choose_mode, _qualified_history, _existing_model
from quant_rl.research.evolution import ModelPopulation


WEEKDAY_NOW = int(datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc).timestamp())


class FakeAPI:
    def __init__(self,now,*,fresh=True,disabled=False,supported=None):
        self.now=now
        self.fresh=fresh
        self.disabled=disabled
        self.supported={'EURUSD','GBPUSD'} if supported is None else set(supported)
    def terminal_info(self):return NS(connected=True)
    ACCOUNT_TRADE_MODE_DEMO=0
    def account_info(self):return NS(login=123,currency='USD',trade_mode=0,trade_allowed=True,server='MetaQuotes-Demo')
    def symbol_info(self,s):
        if s not in self.supported:return None
        return NS(point=.00001,trade_contract_size=100000,trade_mode=(0 if self.disabled else 4))
    def symbol_select(self,s,visible):return s in self.supported
    def symbol_info_tick(self,s):
        if s not in self.supported:return None
        return NS(bid=1.1,ask=1.1001,time=self.now-(100000 if not self.fresh else 10))


class FakeSource:
    def __init__(self,api):self._mt5=api;self.connected=False;self.closed=False
    def connect(self):self.connected=True
    def close(self):self.closed=True


class FakeStore:
    def __init__(self,missing=()):self.read=[];self.missing=set(missing)
    def fetch(self,s):
        self.read.append(s)
        if s in self.missing:raise ConnectionError('Missing MT5 history')
        return NS(manifest=NS(data_origin='MT5_TERMINAL',
                  verified_spread_fraction=1.),symbol=s)


class FakePopulation:
    def __init__(self,models=(),now_s=None,*,cfg=None,server_timezone='UTC'):
        from datetime import datetime,timezone
        now_s=int(time.time() if now_s is None else now_s)
        certificate=datetime.fromtimestamp(now_s-3600,timezone.utc).isoformat()
        policy_hash=(cfg or QuantRLConfig()).policy_hash
        self.data={'symbols':{s:dict(origin='MT5_TERMINAL',account_currency='USD',current_score=.2,arch='test',file='weights.pt',sha256='test',
                            policy_hash=policy_hash,mt5_server_timezone=server_timezone,broker_server='MetaQuotes-Demo',
                            period={'test_end_utc':certificate}) for s in models}}
    def load(self,record,arch):return {'weights':{},'mean':0,'std':1}
    def verify_artifact(self, record):return None


def make_cfg(tmp_path):
    return QuantRLConfig.model_validate({'project_root':str(tmp_path),
        'research':{'symbols':['EURUSD','GBPUSD','USDJPY']},'live':{'symbol':'EURUSD'},'broker_connection':{'mt5_server_timezone':'UTC'}})


def route(tmp_path,now,*,model=(),fresh=True,disabled=False,missing=(),supported=None):
    cfg=make_cfg(tmp_path);src=FakeSource(FakeAPI(now,fresh=fresh,disabled=disabled,supported=supported));store=FakeStore(missing)
    import quant_rl.routing as routing
    original=routing._qualified_history
    routing._qualified_history=lambda cfg,s,f:(True,'Qualified')
    try:
        result=choose_mode(cfg,source_factory=lambda:src,history=store,
                           population=FakePopulation(model,now,cfg=cfg),now_s=now,preload_research=True)
    finally:
        routing._qualified_history=original
    assert src.connected and src.closed
    return result,store.read


def test_market_open_only_missing_qualified_symbols(tmp_path):
    now=WEEKDAY_NOW
    (r,data,errors),read=route(tmp_path,now,model=('EURUSD',),missing=('GBPUSD',))
    assert r.mode=='LIVE' and r.live_symbol=='EURUSD'
    assert read==['GBPUSD']
    assert 'GBPUSD' in errors


def test_missing_model_runs_research_for_only_missing_supported(tmp_path):
    now=WEEKDAY_NOW
    (r,data,errors),read=route(tmp_path,now,model=('EURUSD',))
    assert r.mode=='RESEARCH' and r.reason=='MISSING_MODELS'
    assert r.targets==('GBPUSD',) and read==['GBPUSD']
    assert set(data)=={'GBPUSD'}


def test_market_closed_runs_all_supported_sequentially(tmp_path):
    from datetime import datetime, timezone, timedelta
    now_dt=datetime.now(timezone.utc)
    days_until_saturday=(5-now_dt.weekday()) % 7
    saturday=(now_dt+timedelta(days=days_until_saturday)).replace(hour=12,minute=0,second=0,microsecond=0)
    now=int(saturday.timestamp())
    (r,data,errors),read=route(tmp_path,now,model=('EURUSD',),disabled=True,fresh=False)
    assert r.mode=='RESEARCH' and r.reason=='WEEKLY_SCHEDULE'
    assert r.targets==('EURUSD','GBPUSD')
    assert read==['EURUSD','GBPUSD']
    assert 'USDJPY' not in data



def test_weekday_broker_disabled_is_not_assumed_to_be_market_closure(tmp_path):
    from datetime import datetime, timezone
    now_dt=datetime.now(timezone.utc)
    if now_dt.weekday() >= 5:
        # Pick a recent weekday timestamp; no fresh quotes are generated by
        # the fixture because every symbol is broker-disabled.
        from datetime import timedelta
        now_dt=now_dt-timedelta(days=(now_dt.weekday()-4))
    now=int(now_dt.timestamp())
    cfg=make_cfg(tmp_path)
    src=FakeSource(FakeAPI(now,disabled=True))
    with pytest.raises(ConnectionError,match='market closure unconfirmed'):
        choose_mode(cfg,source_factory=lambda:src,history=FakeStore(),
                    population=FakePopulation(('EURUSD','GBPUSD'),now,cfg=cfg),now_s=now,preload_research=True)
    assert src.closed

def test_live_default_with_complete_models(tmp_path):
    now=WEEKDAY_NOW
    (r,data,errors),read=route(tmp_path,now,model=('EURUSD','GBPUSD'))
    assert r.mode=='LIVE' and r.targets==() and read==[]


def test_no_qualified_symbol_never_trains_or_trades(tmp_path):
    now=WEEKDAY_NOW
    with pytest.raises(RuntimeError,match='No configured symbols supported'):
        route(tmp_path,now,supported=set())


def test_live_selects_another_eligible_model_when_preferred_symbol_unavailable(tmp_path):
    now=WEEKDAY_NOW
    (decision,datasets,rejected),read=route(tmp_path,now,model=('GBPUSD',),missing=('EURUSD',))
    assert decision.mode=='LIVE' and decision.live_symbol=='GBPUSD'
    assert rejected['EURUSD'] and read==['EURUSD']


def test_invalid_model_snapshot_is_treated_as_missing(tmp_path):
    pop=ModelPopulation(tmp_path)
    pop.data['symbols']['EURUSD']=dict(origin='MT5_TERMINAL',current_score=1.,arch='bad',file='absent.pt',sha256='x')
    assert not _existing_model(pop,'EURUSD')


def test_bad_data_evidence_not_qualified(tmp_path):
    cfg=make_cfg(tmp_path)
    bad=NS(manifest=NS(data_origin='UNVERIFIED',verified_spread_fraction=1.))
    assert not _qualified_history(cfg,'EURUSD',bad)[0]
    bad.manifest.data_origin='MT5_TERMINAL'
    bad.manifest.verified_spread_fraction=.2
    assert not _qualified_history(cfg,'EURUSD',bad)[0]


def test_model_population_study_safely_persists_accepted_symbols(tmp_path):
    pop=ModelPopulation(tmp_path)
    pop.publish({}, {}, None, {})
    assert pop.data['format']==ModelPopulation.FORMAT


def test_short_quote_outage_is_not_misclassified_as_market_closed(tmp_path):
    cfg=make_cfg(tmp_path)
    now=WEEKDAY_NOW
    api=FakeAPI(now)
    api.symbol_info_tick=lambda symbol:NS(bid=1.1,ask=1.1001,time=now-420)
    src=FakeSource(api)
    with pytest.raises(ConnectionError,match='market closure unconfirmed'):
        choose_mode(cfg,source_factory=lambda:src,history=FakeStore(),population=FakePopulation(('EURUSD','GBPUSD'),now,cfg=cfg),now_s=now,preload_research=True)
    assert src.closed


def test_live_and_research_use_broker_account_currency(tmp_path, monkeypatch):
    from quant_rl.routing import run_automatic_session, RoutingDecision
    cfg=make_cfg(tmp_path)
    frozen=NS(manifest=NS(data_origin='MT5_TERMINAL'))
    monkeypatch.setattr('quant_rl.routing.choose_mode',lambda cfg:
        (RoutingDecision('RESEARCH','MISSING_MODELS',('EURUSD',),True,account_currency='EUR'),{'EURUSD':frozen},{}))
    called=[]
    monkeypatch.setattr('quant_rl.research.evolution.run_research_session',lambda c,**kw:
        called.append((c.execution.account_currency,kw)) or {'ok':True})
    assert run_automatic_session(cfg)=={'ok':True}
    assert called[0][0]=='EUR' and called[0][1]['targets']==('EURUSD',)
    monkeypatch.setattr('quant_rl.routing.choose_mode',lambda cfg:
        (RoutingDecision('LIVE','MARKET_OPEN_MODELS_READY',(),True,'EURUSD','JPY'),{},{}))
    monkeypatch.setattr('quant_rl.execution.live.run_live_session',lambda c:
        called.append(('LIVE',c.execution.account_currency)) or 'LIVE_STARTED')
    assert run_automatic_session(cfg)=='LIVE_STARTED'
    assert called[-1]==('LIVE','JPY')


def test_non_mt5_runtime_history_is_unavailable(tmp_path):
    from quant_rl.data.mt5_history import MT5History
    store=MT5History()
    with pytest.raises(ValueError,match='MT5 connection'):
        store.fetch('EURUSD')


def test_fresh_mt5_acquisition_called_every_time_without_offline_fallback(tmp_path,monkeypatch):
    from quant_rl.data.mt5_history import MT5History
    import quant_rl.data.mt5_history as h
    from quant_rl.data.mt5_source import OHLCVBar
    from tests._specs import default_symbol_spec
    cfg=make_cfg(tmp_path)
    api=NS(account_info=lambda:NS(currency='USD'))
    hits=[]
    class Source:
        def __init__(self,**kwargs):self._mt5=api
        def connect(self):hits.append('connect')
        def get_symbol_spec(self,s,c):return default_symbol_spec(s)
        def fetch_bars_since(self,s,tf,start):
            hits.append('fetch')
            return [OHLCVBar(1704067200+3600*i,1.1,1.1001,1.0999,1.1,150,3.) for i in range(250)]
        def fetch_m1_range(self,s,start,end):return []
        def close(self):hits.append('close')
    monkeypatch.setattr(h,'MT5DataSource',Source)
    # This test verifies fresh acquisitions, not rolling coverage qualification.
    monkeypatch.setattr(h,'select_rolling_bars',lambda bars,tf,years,**kw:
                        (bars,bars[0].time_s,bars[-1].time_s+3600))
    store=MT5History(cfg)
    a=store.fetch('EURUSD');b=store.fetch('EURUSD')
    assert a.fingerprint==b.fingerprint and hits.count('fetch')==2
    class Unavailable(Source):
        def connect(self):raise ConnectionError('terminal unavailable')
    monkeypatch.setattr(h,'MT5DataSource',Unavailable)
    with pytest.raises(ConnectionError,match='terminal unavailable'):
        store.fetch('EURUSD')


def test_market_open_does_not_train_broker_disabled_symbol(tmp_path):
    now=WEEKDAY_NOW; cfg=make_cfg(tmp_path)
    api=FakeAPI(now)
    original=api.symbol_info
    def info(symbol):
        out=original(symbol)
        if symbol=='GBPUSD' and out is not None:out.trade_mode=0
        return out
    api.symbol_info=info
    import quant_rl.routing as routing
    original_qualification=routing._qualified_history
    routing._qualified_history=lambda cfg,s,f:(True,'Qualified')
    try:
        decision,data,errors=choose_mode(cfg,source_factory=lambda:FakeSource(api),
                        history=FakeStore(),population=FakePopulation(('EURUSD',),now,cfg=cfg),now_s=now,preload_research=True)
        assert decision.mode=='LIVE' and data=={}
    finally:
        routing._qualified_history=original_qualification


def test_account_currency_change_triggers_research_instead_of_reusing_model(tmp_path):
    now=WEEKDAY_NOW
    cfg=make_cfg(tmp_path)
    api=FakeAPI(now)
    api.account_info=lambda:NS(login=123,currency='EUR',trade_mode=0,trade_allowed=True,server='MetaQuotes-Demo')
    import quant_rl.routing as routing
    previous=routing._qualified_history
    routing._qualified_history=lambda cfg,s,f:(True,'Qualified')
    try:
        decision,data,_=choose_mode(cfg,source_factory=lambda:FakeSource(api),
            history=FakeStore(),population=FakePopulation(('EURUSD','GBPUSD'),now,cfg=cfg),now_s=now,preload_research=True)
        assert decision.mode=='RESEARCH'
        assert decision.account_currency=='EUR'
        assert decision.targets==('EURUSD','GBPUSD')
    finally:
        routing._qualified_history=previous


def test_partial_publication_preserves_other_symbol_models_and_champion(tmp_path):
    from quant_rl.research.evolution import select_champion
    import torch
    pop=ModelPopulation(tmp_path)
    baseline={'weights':{'a':torch.tensor([1.])},'mean':torch.zeros(1),'std':torch.ones(1)}
    rec=dict(arch='test',origin='MT5_TERMINAL',account_currency='USD',
             current_score=.1,settings={},feature_profile={},mt5_server_timezone='UTC')
    pop.publish({'EURUSD':(baseline,rec)}, {},'EURUSD',{})
    pop=ModelPopulation(tmp_path)
    assert _existing_model(pop,'EURUSD','USD','UTC')
    assert select_champion({'EURUSD':.1})=='EURUSD'
    pop.publish({}, {},'EURUSD',{'GBPUSD':{'status':'NO_QUALIFIED_MODEL'}})
    reloaded=ModelPopulation(tmp_path)
    assert reloaded.data['champion']=='EURUSD'
    assert _existing_model(reloaded,'EURUSD','USD','UTC')


def test_router_refuses_funded_account_by_default(tmp_path):
    cfg=make_cfg(tmp_path)
    api=FakeAPI(WEEKDAY_NOW)
    api.account_info=lambda:NS(login=123,currency='USD',trade_mode=2,trade_allowed=True,server='Broker-Live')
    source=FakeSource(api)
    with pytest.raises(PermissionError,match='Demo-only'):
        choose_mode(cfg,source_factory=lambda:source,history=FakeStore(),
                    population=FakePopulation(('EURUSD','GBPUSD'),WEEKDAY_NOW),now_s=WEEKDAY_NOW)
    assert source.closed


def test_router_refuses_unknown_account_mode(tmp_path):
    cfg=make_cfg(tmp_path)
    api=FakeAPI(WEEKDAY_NOW)
    api.account_info=lambda:NS(login=123,currency='USD',trade_allowed=True,server='MetaQuotes-Demo')
    source=FakeSource(api)
    with pytest.raises(PermissionError,match='trading mode unavailable'):
        choose_mode(cfg,source_factory=lambda:source,history=FakeStore(),
                    population=FakePopulation(('EURUSD','GBPUSD'),WEEKDAY_NOW),now_s=WEEKDAY_NOW)


def test_invalid_iana_server_timezone_rejected():
    with pytest.raises(Exception,match='Unknown IANA MT5 server timezone'):
        QuantRLConfig.model_validate({'broker_connection':{'mt5_server_timezone':'Not/A_Real_Zone'}})
