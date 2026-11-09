"""Current research configuration, symbols, chronology, and risk constraints."""
from __future__ import annotations
import hashlib
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from quant_rl.data.time_alignment import TIMEFRAME_SECONDS

VALID_TIMEFRAMES = tuple(TIMEFRAME_SECONDS)

class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    symbols: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD")
    lookback_years: int = Field(default=4, ge=1, le=50)
    timeframe: str = "H4"
    volatility_estimator: str = "atr"
    volatility_period: int = Field(default=14, ge=2, le=300)
    batch_size: int = Field(default=128, ge=8, le=8192)
    seed: int = 42

    @model_validator(mode="after")
    def verify(self):
        if not self.symbols or len(set(self.symbols)) != len(self.symbols) or any(not x or x != x.strip() for x in self.symbols):
            raise ValueError("symbols must be nonempty and distinct")
        if self.timeframe not in VALID_TIMEFRAMES:
            raise ValueError("unsupported timeframe")
        if self.timeframe in ("M1", "M5", "M15", "M30"):
            raise ValueError("Primary timeframe must be H1 or coarser: MT5 research acquisition uses native H1")
        if self.volatility_estimator not in ("atr", "realized_vol", "parkinson_vol"):
            raise ValueError("unsupported volatility estimator")
        return self

class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    observation_timeframes: tuple[str,...] = ("H1", "H4", "H8", "D1")
    sequence_lookback_bars: int = Field(default=32, ge=4, le=512)
    include_technical_indicators: bool = True

    @model_validator(mode="after")
    def verify(self):
        if not self.observation_timeframes or len(set(self.observation_timeframes)) != len(self.observation_timeframes):
            raise ValueError("Observation timeframes must be nonempty and distinct")
        if any(tf not in VALID_TIMEFRAMES or tf in ("M1", "M5", "M15", "M30") for tf in self.observation_timeframes):
            raise ValueError("Observation timeframes must be H1 or coarser")
        return self

class GeometryModeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_stop_units: float = Field(default=0.25, gt=0.05, le=5.)
    max_stop_units: float = Field(default=6., gt=0.5, le=20.)
    min_target_units: float = Field(default=0.35, gt=0.05, le=10.)
    max_target_units: float = Field(default=12., gt=0.5, le=50.)

    @model_validator(mode="after")
    def bounds(self):
        if self.max_stop_units <= self.min_stop_units or self.max_target_units <= self.min_target_units:
            raise ValueError('Invalid trade geometry bounds')
        return self

class PositionSizingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    min_risk_per_trade_pct: float = Field(default=0.1, gt=0.01, le=5.)
    max_risk_per_trade_pct: float = Field(default=1., gt=0.05, le=5.)

    @model_validator(mode="after")
    def bounds(self):
        if self.max_risk_per_trade_pct < self.min_risk_per_trade_pct:
            raise ValueError('Invalid position risk bounds')
        return self

class IntelligenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    geometry: GeometryModeConfig = Field(default_factory=GeometryModeConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)

class PPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    episodes: int = Field(default=1000, ge=1, le=100000)
    ppo_epochs: int = Field(default=1000, ge=1, le=100000)
    gamma: float = Field(default=0.995, gt=0, le=1.)
    gae_lambda: float = Field(default=0.95, ge=0, le=1.)
    clip_ratio: float = Field(default=0.2, gt=0, le=0.5)
    entropy_coef: float = Field(default=0.005, ge=0, le=1)
    value_coef: float = Field(default=0.5, ge=0, le=10)
    hidden_dim: int = Field(default=64, ge=8, le=512)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.1)
    penalty_reference_lots: float = Field(default=0.01, gt=0, le=10)
    cpu_threads: int = Field(default=1, ge=1, le=128)
    validation_interval_episodes: int = Field(default=25, ge=1, le=10000)
    target_kl: float = Field(default=0.025, gt=0.001, le=1.0)

class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    trials: int = Field(default=1000, ge=1, le=100000)
    tournament_top_k: int = Field(default=9, ge=1, le=100)
    enabled: bool = True
    max_session_seconds: int = Field(default=7200, ge=60, le=604800)
    pruner_startup_trials: int = Field(default=5, ge=0, le=1000)
    pruner_warmup_episodes: int = Field(default=100, ge=0, le=100000)

class QualificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_drawdown_pct: float = Field(default=20., gt=0.5, le=80.)
    minimum_trades_per_year: int = Field(default=12, ge=0, le=5000)
    minimum_evaluation_calendar_days: int = Field(default=90, ge=0, le=3650)
    cost_stress_multiplier: float = Field(default=1.5, ge=1, le=5)
    require_positive_stressed_return: bool = True
    historical_spread_warning_fraction: float = Field(default=0.95, ge=0., le=1.)
    minimum_m1_coverage_fraction: float = Field(default=0.80, ge=0., le=1.)
    max_live_model_age_days: int = Field(default=14, ge=1, le=365)

class ExecutionConfig(BaseModel):
    """Execution invariants and C1-C26 safety ceilings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    starting_balance: float = Field(default=10000.0, gt=100.0)
    account_currency: str = "USD"
    account_leverage: float = Field(default=30.0, gt=0.0, le=1000.0)
    maximum_leverage: float = Field(default=30.0, gt=0.0, le=500.0)
    max_open_positions: int = Field(default=1, ge=1, le=50)
    max_open_risk_multiplier: float = Field(default=3.0, gt=0.5, le=20.0)
    max_currency_concentration: int = Field(default=3, ge=1, le=20)
    max_consecutive_losses: int = Field(default=8, ge=1, le=100)
    daily_drawdown_limit_pct: float = Field(default=3.0, gt=0.1, le=50.0)
    weekly_drawdown_limit_pct: float = Field(default=6.0, gt=0.2, le=80.0)
    # Market-conditioned checks are always active with valid prior history.
    market_history_bars: int = Field(default=128, ge=20, le=4096)
    market_min_history_bars: int = Field(default=20, ge=10, le=256)
    market_window_candidates: tuple[int, ...] = (32, 64, 128)
    market_regime_shift_ratio: float = Field(default=1.5, gt=1.0, le=5.0)
    market_portfolio_risk_tightening_floor: float = Field(default=0.5, gt=0.0, le=1.0)
    market_geometry_volatility_floor_fraction: float = Field(default=0.10, ge=0.0, le=0.5)
    market_geometry_spread_floor_multiplier: float = Field(default=1.0, ge=0.0, le=3.0)
    market_min_net_rr_stress_uplift: float = Field(default=0.25, ge=0.0, le=2.0)
    market_spread_shock_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)
    market_volatility_shock_multiplier: float = Field(default=4.0, ge=1.0, le=10.0)
    max_round_trip_cost_pips: float = Field(default=3.0, gt=0.1, le=50.0)
    max_cost_to_stop_fraction: float = Field(default=0.35, gt=0.01, le=0.95)
    min_net_reward_to_risk_ratio: float = Field(default=0.25, ge=0.0, le=5.0)
    max_quote_age_seconds: int = Field(default=120, ge=1, le=3600)
    max_slippage_points: int = Field(default=20, ge=0, le=500)
    simulated_slippage_points_per_side: float = Field(default=0.5, ge=0.0, le=20.0)

    commission_per_lot_round_trip: float = Field(default=7.0, ge=0.0, le=100.0)
    # Explicit, signed account-currency cash per lot per normal UTC rollover.
    # Negative = debit; positive = credit. Defaults are MODEL ASSUMPTIONS,
    # not verified broker swap charges or implicit MT5 swap points.
    financing_long_cash_per_lot_rollover: float = Field(default=-4.5, ge=-500., le=500., allow_inf_nan=False)
    financing_short_cash_per_lot_rollover: float = Field(default=-4.5, ge=-500., le=500., allow_inf_nan=False)
    financing_rollover_utc_hour: int = Field(default=21, ge=0, le=23)
    financing_triple_weekday_utc: int = Field(default=2, ge=0, le=4)
    # Generic FX weekend entry schedule in configured MT5 server wall time.
    # It blocks new entries only; existing broker positions remain managed.
    friday_cutoff_server_hour: int = Field(default=23, ge=0, le=23)
    sunday_open_server_hour: int = Field(default=21, ge=0, le=23)

    @model_validator(mode="after")
    def validate_market_history(self) -> ExecutionConfig:
        if self.simulated_slippage_points_per_side > self.max_slippage_points:
            raise ValueError('Simulated slippage cannot exceed safety slippage bound')
        if self.market_min_history_bars > self.market_history_bars:
            raise ValueError("market_min_history_bars exceeds market_history_bars")
        if (not self.market_window_candidates or any(not isinstance(w, int) or isinstance(w, bool) or w < self.market_min_history_bars for w in self.market_window_candidates)
                or len(set(self.market_window_candidates)) != len(self.market_window_candidates)
                or max(self.market_window_candidates) > self.market_history_bars):
            raise ValueError("market_window_candidates must be unique, valid history lengths")
        return self


class BrokerConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_path: str = ""
    mt5_server_timezone: str | None = None

    @model_validator(mode="after")
    def validate_timezone(self):
        timezone_name = self.mt5_server_timezone
        if timezone_name is not None:
            if not timezone_name.strip():
                raise ValueError("mt5_server_timezone must be an IANA timezone name or null")
            try:
                ZoneInfo(timezone_name)
            except (ZoneInfoNotFoundError, ValueError) as exc:
                raise ValueError(f"Unknown IANA MT5 server timezone: {timezone_name!r}") from exc
        return self


class LiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    symbol: str = "EURUSD"
    poll_seconds: float = Field(default=10.0, ge=1.0, le=300.0)
    max_entry_delay_seconds: int = Field(default=300, ge=10, le=3600)
    economic_blackout_active: bool = False
    order_deviation_points: int = Field(default=20, ge=0, le=500)
    magic: int = Field(default=520052, ge=1, le=2147483647)
    allow_non_demo_account: bool = False

class QuantRLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    live: LiveConfig = Field(default_factory=LiveConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    intelligence: IntelligenceConfig = Field(default_factory=IntelligenceConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    qualification: QualificationConfig = Field(default_factory=QualificationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    broker_connection: BrokerConnectionConfig = Field(default_factory=BrokerConnectionConfig)
    project_root: str = "."

    @model_validator(mode="after")
    def validate_live_execution_bounds(self) -> QuantRLConfig:
        # Broker deviation must fit inside the slippage bound used by risk checks.
        if self.live.order_deviation_points > self.execution.max_slippage_points:
            raise ValueError(
                "live.order_deviation_points cannot exceed execution.max_slippage_points"
            )
        return self

    @property
    def policy_hash(self) -> str:
        # This is the deployability certificate, not a hash of every research
        # knob. It covers everything that can change the policy's observation,
        # learned economic objective, qualification standard, or live execution
        # semantics. Search budgets and presentation/identity fields are omitted.
        payload={
            'research_timeframe':self.research.timeframe,
            'research_lookback_years':self.research.lookback_years,
            'volatility_estimator':self.research.volatility_estimator,
            'volatility_period':self.research.volatility_period,
            'data':self.data.model_dump(),
            'intelligence':self.intelligence.model_dump(),
            'execution':self.execution.model_dump(exclude={'starting_balance'}),
            'qualification':self.qualification.model_dump(),
            # The inactivity penalty changes the learned reward objective.
            'training_economics':{
                'penalty_reference_lots':self.ppo.penalty_reference_lots,
            },
            # Symbol and magic identify deployment, not policy semantics.
            # These fields do affect when/how an otherwise valid action executes.
            'live_execution':{
                'poll_seconds':self.live.poll_seconds,
                'max_entry_delay_seconds':self.live.max_entry_delay_seconds,
                'economic_blackout_active':self.live.economic_blackout_active,
                'order_deviation_points':self.live.order_deviation_points,
            },
            'mt5_server_timezone':self.broker_connection.mt5_server_timezone,
            'contract_version':'economic_deployability_v4',
        }
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:20]

    @property
    def config_hash(self) -> str:
        payload = self.model_dump(exclude={"project_root", "broker_connection", "live"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
