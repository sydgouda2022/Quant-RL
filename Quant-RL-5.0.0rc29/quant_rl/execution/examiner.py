"""Authoritative C1-C26 Pre-Trade Safety Firewall (Examiner)."""

from __future__ import annotations

from dataclasses import dataclass
import math

from quant_rl.config.schema import ExecutionConfig
from quant_rl.data.mt5_source import SymbolSpec, split_fx_symbol
from quant_rl.execution.sizing import PositionSizeQuote, calculate_broker_position_size
from quant_rl.execution.market_safety import MarketSafetyContext
from quant_rl.contracts import ActionProposal, TradeDirection
from quant_rl.execution.trading_calendar import weekly_fx_entry_blocked


@dataclass(frozen=True)
class OpenPositionSnapshot:
    """Point-in-time snapshot of an open position in the portfolio."""

    position_id: int
    symbol: str
    direction: TradeDirection
    volume_lots: float
    entry_price: float
    stop_price: float
    target_price: float
    entry_time_s: int
    open_risk_account_ccy: float
    margin_used_account_ccy: float
    unrealized_pnl_account_ccy: float = 0.0


@dataclass(frozen=True)
class PortfolioStateSnapshot:
    """Point-in-time portfolio state inspected by the C1-C26 Examiner."""

    current_time_s: int
    balance: float
    equity: float
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    open_positions: tuple[OpenPositionSnapshot, ...] = ()
    consecutive_losses: int = 0
    last_bar_time_s: int = 0
    timeframe_seconds: int = 14400
    broker_healthy: bool = True
    kill_switch_engaged: bool = False
    economic_blackout_active: bool = False
    # Actual UTC wall-clock instant for execution-time gates. Historical
    # simulations default to the observation timestamp above.
    execution_time_s: int | None = None


@dataclass(frozen=True)
class ExaminerVerdict:
    """Result of evaluating all 26 safety checks on a proposed action."""

    approved: bool
    is_no_trade: bool
    veto_code: str | None
    veto_reason: str
    checks_passed: dict[str, bool]
    approved_volume_lots: float
    entry_price: float
    stop_price: float
    target_price: float
    actual_risk_account_ccy: float
    required_margin_account_ccy: float
    round_trip_cost_pips: float
    round_trip_cost_account_ccy: float
    safety_mode: str = "conservative_cold_start"
    allowed_cost_pips: float = 0.0
    effective_risk_ceiling_fraction: float = 0.0
    adaptive_limits: tuple[tuple[str, float], ...] = ()
    failed_check_details: tuple[tuple[str, str], ...] = ()
    raw_requested_lots: float = 0.0


class PreTradeExaminer:
    """
    Authoritative C1-C26 Safety Firewall.
    Evaluates all 26 independent checks before any order can reach the simulator or live broker.
    """

    def __init__(self, execution_cfg: ExecutionConfig, max_risk_per_trade_pct: float = 1.0,
                 max_drawdown_pct: float = 20.0, server_timezone: str = "UTC") -> None:
        self.cfg = execution_cfg
        self.max_drawdown_pct = max_drawdown_pct
        self.max_risk_fraction = max_risk_per_trade_pct / 100.0
        self.server_timezone = server_timezone

    def inspect(
        self,
        proposal: ActionProposal,
        portfolio: PortfolioStateSnapshot,
        symbol_spec: SymbolSpec,
        mid_price: float,
        spread_pips: float,
        volatility_price: float,
        cost_multiplier: float = 1.0,
        fx_rates: dict[str, float] | None = None,
        market_context: MarketSafetyContext | None = None,
    ) -> ExaminerVerdict:
        if proposal.direction == TradeDirection.NO_TRADE:
            return ExaminerVerdict(
                approved=False,
                is_no_trade=True,
                veto_code=None,
                veto_reason="Intelligence selected NO_TRADE",
                checks_passed={f"C{i}": True for i in range(1, 27)},
                approved_volume_lots=0.0,
                entry_price=mid_price,
                stop_price=0.0,
                target_price=0.0,
                actual_risk_account_ccy=0.0,
                required_margin_account_ccy=0.0,
                round_trip_cost_pips=0.0,
                round_trip_cost_account_ccy=0.0,
            )

        if (not math.isfinite(mid_price) or mid_price <= 0
                or not math.isfinite(spread_pips) or spread_pips < 0
                or not math.isfinite(volatility_price) or volatility_price <= 0
                or not math.isfinite(cost_multiplier) or cost_multiplier < 1):
            raise ValueError("Invalid market price, spread, volatility or stress multiplier")

        adaptive = bool(market_context is not None)
        if adaptive and market_context is not None:
            if market_context.observation_time_s != portfolio.current_time_s:
                raise ValueError("Market safety context does not match decision timestamp")
            if market_context.sample_count < self.cfg.market_min_history_bars:
                raise ValueError("Insufficient history for adaptive safety; do not trust supplied context")
            median_vol = market_context.median_volatility_price
            median_spread = market_context.median_spread_pips
            # Volatility and spread shocks can only REDUCE allowable risk;
            # they never increase risk, leverage, drawdown, or position ceilings.
            risk_scale = min(1.0, median_vol / volatility_price,
                             max(median_spread, 0.1) / max(spread_pips, 0.1))
            risk_scale = max(0.0, risk_scale)
            effective_risk_ceiling = self.max_risk_fraction * risk_scale
        else:
            effective_risk_ceiling = self.max_risk_fraction
        # Single-position application: risk adapts only to the selected market.
        # C7-C10 still enforce absolute portfolio limits independently.
        risk_tightening = (max(self.cfg.market_portfolio_risk_tightening_floor,
            min(1.0, effective_risk_ceiling / max(self.max_risk_fraction, 1e-12)))
            if adaptive else 1.0)
        portfolio_tightening = risk_tightening
        dynamic_position_cap = max(1, min(self.cfg.max_open_positions,
            math.floor(self.cfg.max_open_positions * portfolio_tightening + 1e-9)))
        dynamic_risk_budget = (portfolio.equity * self.cfg.max_open_risk_multiplier
                               * self.max_risk_fraction * portfolio_tightening)
        dynamic_margin_fraction = 0.80 * risk_tightening
        dynamic_leverage_cap = self.cfg.maximum_leverage * risk_tightening
        # These are geometry feasibility floors; model-owned stops/targets are
        # never modified, and broker stops_level is always authoritative.
        market_geometry_floor = (max(
            self.cfg.market_geometry_spread_floor_multiplier * spread_pips * symbol_spec.pip_size,
            self.cfg.market_geometry_volatility_floor_fraction * market_context.median_volatility_price,
        ) if adaptive and market_context is not None else 0.0)
        effective_spread_pips = spread_pips * cost_multiplier
        half_spread_price = 0.5 * effective_spread_pips * symbol_spec.pip_size
        if proposal.direction == TradeDirection.BUY:
            entry_price = mid_price + half_spread_price
            stop_price = entry_price - proposal.stop_distance_price
            target_price = entry_price + proposal.target_distance_price
        else:
            entry_price = mid_price - half_spread_price
            stop_price = entry_price + proposal.stop_distance_price
            target_price = entry_price - proposal.target_distance_price

        size_quote: PositionSizeQuote = calculate_broker_position_size(
            equity=portfolio.equity,
            requested_risk_fraction=min(proposal.requested_risk_fraction, effective_risk_ceiling),
            max_risk_fraction=effective_risk_ceiling,
            stop_distance_price=proposal.stop_distance_price,
            entry_price=entry_price,
            symbol_spec=symbol_spec,
            account_leverage=self.cfg.account_leverage,
            account_currency=self.cfg.account_currency,
            fx_rates=fx_rates,
        )

        commission_pips_equiv = (
            (self.cfg.commission_per_lot_round_trip * cost_multiplier)
            / max(size_quote.pip_value_per_lot_account_ccy, 1e-6)
        )
        slippage_pips_equiv = (self.cfg.max_slippage_points * symbol_spec.point
                               / symbol_spec.pip_size) * cost_multiplier
        round_trip_cost_pips = effective_spread_pips + commission_pips_equiv + slippage_pips_equiv
        round_trip_cost_acct = (
            round_trip_cost_pips * size_quote.pip_value_per_lot_account_ccy * size_quote.volume_lots
        )

        # The market-conditioned cost budget uses the actual stop size, and
        # contemporaneous spread is screened against *prior* spread history.
        # The conservative cold-start pip ceiling applies if
        # there are not enough completed observations. Worst-case slippage is
        # still included; an expected-cost estimate may not bypass the veto.
        allowed_cost_pips = (
            size_quote.stop_pips * self.cfg.max_cost_to_stop_fraction
            if adaptive else self.cfg.max_round_trip_cost_pips
        )
        if adaptive and market_context is not None:
            spread_shock_limit = max(market_context.upper_spread_pips,
                                     market_context.median_spread_pips, 0.1) * self.cfg.market_spread_shock_multiplier
            spread_shock_ok = spread_pips <= spread_shock_limit
            volatility_shock_ok = (volatility_price <= max(
                market_context.upper_volatility_price,
                market_context.median_volatility_price) * self.cfg.market_volatility_shock_multiplier)
        else:
            spread_shock_ok = True
            volatility_shock_ok = True

        calendar_time_s=(portfolio.execution_time_s
                         if portfolio.execution_time_s is not None
                         else portfolio.current_time_s)
        is_weekend = weekly_fx_entry_blocked(
            calendar_time_s,
            server_timezone=self.server_timezone,
            friday_cutoff_hour=self.cfg.friday_cutoff_server_hour,
            sunday_open_hour=self.cfg.sunday_open_server_hour,
        )

        existing_same_symbol = [p for p in portfolio.open_positions if p.symbol == proposal.symbol]
        total_open_risk = sum(p.open_risk_account_ccy for p in portfolio.open_positions) + size_quote.actual_risk_account_ccy
        total_margin = sum(p.margin_used_account_ccy for p in portfolio.open_positions) + size_quote.required_margin_account_ccy

        new_base, new_quote, _ = split_fx_symbol(proposal.symbol)
        # Signed exposure in risk-equivalent account currency. This is not an
        # exact FX delta or a replacement for a notional/delta margin engine.
        signed_ccy_risk: dict[str, float] = {}
        def add_ccy(symbol: str, direction: TradeDirection, risk: float) -> None:
            b, q, _ = split_fx_symbol(symbol)
            sign = 1 if direction == TradeDirection.BUY else -1
            signed_ccy_risk[b] = signed_ccy_risk.get(b, 0.0) + sign * risk
            signed_ccy_risk[q] = signed_ccy_risk.get(q, 0.0) - sign * risk
        for pos in portfolio.open_positions:
            add_ccy(pos.symbol, pos.direction, pos.open_risk_account_ccy)
        add_ccy(proposal.symbol, proposal.direction, size_quote.actual_risk_account_ccy)
        currency_risk_budget = (portfolio.equity * self.cfg.max_currency_concentration
                                * self.max_risk_fraction * portfolio_tightening)
        max_net_currency_risk = max(map(abs, signed_ccy_risk.values()), default=0.0)
        same_ccy_count = 0
        for p in portfolio.open_positions:
            b, q, _ = split_fx_symbol(p.symbol)
            if new_base in (b, q) or new_quote in (b, q):
                same_ccy_count += 1

        daily_dd_pct = 100.0 * max(0.0, portfolio.day_start_equity - portfolio.equity) / max(portfolio.day_start_equity, 1.0)
        weekly_dd_pct = 100.0 * max(0.0, portfolio.week_start_equity - portfolio.equity) / max(portfolio.week_start_equity, 1.0)
        total_dd_pct = 100.0 * max(0.0, portfolio.peak_equity - portfolio.equity) / max(portfolio.peak_equity, 1.0)

        min_broker_dist_price = max(symbol_spec.stops_level_points * symbol_spec.point, 2.0 * symbol_spec.pip_size)
        effective_geometry_floor = max(min_broker_dist_price, market_geometry_floor)
        # C26 remains a *mechanical* payoff filter until a separately
        # calibrated probability model is independently validated.
        dynamic_min_net_rr = (self.cfg.min_net_reward_to_risk_ratio
            + (1.0 - risk_tightening) * self.cfg.market_min_net_rr_stress_uplift if adaptive
            else self.cfg.min_net_reward_to_risk_ratio)
        adaptive_limits = tuple(sorted({
            "C8_max_positions": float(dynamic_position_cap),
            "C9_open_risk_budget": float(dynamic_risk_budget),
            "C10_currency_risk_budget": float(currency_risk_budget),
            "C10_max_net_currency_risk": float(max_net_currency_risk),
            "C18_margin_fraction": float(dynamic_margin_fraction),
            "C19_max_leverage": float(dynamic_leverage_cap),
            "C20_C21_min_distance": float(effective_geometry_floor),
            "C26_min_net_ratio": float(dynamic_min_net_rr),
            "rolling_spread_window_bars": float(market_context.spread_window_bars) if adaptive and market_context else 0.0,
            "rolling_volatility_window_bars": float(market_context.volatility_window_bars) if adaptive and market_context else 0.0,
        }.items()))
        effective_leverage = (
            (total_margin * self.cfg.account_leverage) / max(portfolio.equity, 1.0)
        )

        checks: dict[str, tuple[bool, str]] = {
            "C1": (not portfolio.kill_switch_engaged, "Kill-switch engaged"),
            "C2": (portfolio.broker_healthy, "Broker connection or heartbeat unhealthy"),
            "C3": (
                abs(portfolio.current_time_s - portfolio.last_bar_time_s) <= 2 * portfolio.timeframe_seconds,
                "Market observation timestamp is stale",
            ),
            "C4": (
                ((symbol_spec.trade_mode == 4)
                 or (symbol_spec.trade_mode == 1 and proposal.direction == TradeDirection.BUY)
                 or (symbol_spec.trade_mode == 2 and proposal.direction == TradeDirection.SELL))
                and mid_price > 0.0,
                "Symbol not tradable in proposed direction or invalid quote",
            ),
            "C5": (not is_weekend, "FX market closed for weekend"),
            "C6": (not portfolio.economic_blackout_active, "High-impact macroeconomic event blackout"),
            "C7": (len(existing_same_symbol) == 0, "Position already open on this symbol"),
            "C8": (len(portfolio.open_positions) < dynamic_position_cap, "Market-conditioned concurrent-position capacity exceeded"),
            "C9": (
                total_open_risk <= dynamic_risk_budget,
                "Total portfolio open risk ceiling exceeded",
            ),
            "C10": (same_ccy_count < self.cfg.max_currency_concentration
                    and (not adaptive or max_net_currency_risk <= currency_risk_budget + 1e-7),
                    "Currency-count or market-conditioned signed risk concentration exceeded"),
            "C11": (daily_dd_pct < self.cfg.daily_drawdown_limit_pct, f"Daily drawdown {daily_dd_pct:.2f}% >= limit"),
            "C12": (weekly_dd_pct < self.cfg.weekly_drawdown_limit_pct, f"Weekly drawdown {weekly_dd_pct:.2f}% >= limit"),
            "C13": (total_dd_pct < self.max_drawdown_pct, f"Peak-to-valley drawdown {total_dd_pct:.2f}% >= {self.max_drawdown_pct}% ceiling"),
            "C14": (portfolio.consecutive_losses < self.cfg.max_consecutive_losses, "Consecutive loss circuit breaker triggered"),
            "C15": (
                0.0 < proposal.requested_risk_fraction <= self.max_risk_fraction + 1e-9,
                "Requested risk fraction exceeds per-trade risk ceiling",
            ),
            "C16": (
                size_quote.valid and size_quote.volume_lots >= symbol_spec.volume_min,
                size_quote.reason or "Volume below broker minimum lot",
            ),
            "C17": (
                size_quote.raw_requested_lots <= symbol_spec.volume_max + 1e-9,
                "Uncapped requested volume exceeds broker maximum lot",
            ),
            "C18": (
                total_margin < portfolio.equity * dynamic_margin_fraction,
                "Insufficient free margin after position reservation",
            ),
            "C19": (
                effective_leverage <= dynamic_leverage_cap + 1e-6,
                f"Effective leverage {effective_leverage:.2f}x exceeds maximum {self.cfg.maximum_leverage:.2f}x",
            ),
            "C20": (
                proposal.stop_distance_price >= effective_geometry_floor and stop_price > 0.0,
                "Stop-loss distance violates broker stops_level minimum",
            ),
            "C21": (
                proposal.target_distance_price >= effective_geometry_floor and target_price > 0.0,
                "Take-profit distance violates broker stops_level minimum",
            ),
            "C22": (
                spread_shock_ok
                and round_trip_cost_pips <= allowed_cost_pips
                and (round_trip_cost_pips / max(size_quote.stop_pips, 1e-6)) <= self.cfg.max_cost_to_stop_fraction,
                (f"C22 cost={round_trip_cost_pips:.2f} pips / budget={allowed_cost_pips:.2f} "
                 f"pips; spread regime acceptable={spread_shock_ok}; "
                 f"mode={'market_adaptive' if adaptive else 'conservative_cold_start'}"),
            ),
            "C23": (
                volatility_shock_ok
                and volatility_price > (0 if adaptive else symbol_spec.pip_size * 1.5)
                and proposal.stop_distance_price <= volatility_price * 6.0,
                "Abnormal volatility regime or invalid stop geometry",
            ),
            "C24": (
                size_quote.actual_risk_account_ccy <= portfolio.equity * effective_risk_ceiling * 1.001,
                "Post-rounding monetary risk exceeds account risk ceiling",
            ),
            "C25": (
                (proposal.direction == TradeDirection.BUY and stop_price < entry_price < target_price)
                or (proposal.direction == TradeDirection.SELL and target_price < entry_price < stop_price),
                "Stop/Entry/Target directional ordering invariant violated",
            ),
            "C26": (
                (proposal.target_distance_price / max(proposal.stop_distance_price, 1e-9))
                - (round_trip_cost_pips / max(size_quote.stop_pips, 1e-9)) >= dynamic_min_net_rr,
                "Mechanical net target payoff after estimated costs is below feasibility threshold",
            ),
        }

        checks_bool = {k: v[0] for k, v in checks.items()}
        for code, (passed, reason) in checks.items():
            if not passed:
                return ExaminerVerdict(
                    approved=False,
                    is_no_trade=False,
                    veto_code=code,
                    veto_reason=reason,
                    checks_passed=checks_bool,
                    failed_check_details=tuple((k, explanation) for k,(ok,explanation) in checks.items() if not ok),
                    approved_volume_lots=0.0,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    actual_risk_account_ccy=size_quote.actual_risk_account_ccy,
                    required_margin_account_ccy=size_quote.required_margin_account_ccy,
                    round_trip_cost_pips=round_trip_cost_pips,
                    round_trip_cost_account_ccy=round_trip_cost_acct,
                    safety_mode="market_adaptive" if adaptive else "conservative_cold_start",
                    allowed_cost_pips=allowed_cost_pips,
                    raw_requested_lots=size_quote.raw_requested_lots,
                    effective_risk_ceiling_fraction=effective_risk_ceiling,
                    adaptive_limits=adaptive_limits,
                )

        return ExaminerVerdict(
            approved=True,
            is_no_trade=False,
            veto_code=None,
            veto_reason="APPROVED",
            checks_passed=checks_bool,
            approved_volume_lots=size_quote.volume_lots,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            actual_risk_account_ccy=size_quote.actual_risk_account_ccy,
            required_margin_account_ccy=size_quote.required_margin_account_ccy,
            round_trip_cost_pips=round_trip_cost_pips,
            round_trip_cost_account_ccy=round_trip_cost_acct,
            safety_mode="market_adaptive" if adaptive else "conservative_cold_start",
            allowed_cost_pips=allowed_cost_pips,
            raw_requested_lots=size_quote.raw_requested_lots,
            effective_risk_ceiling_fraction=effective_risk_ceiling,
            adaptive_limits=adaptive_limits,
        )
