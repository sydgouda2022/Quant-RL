"""Sequential Chronological Portfolio Simulator and Economic Accounting Engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
from datetime import datetime, timezone
import math
import time
import numpy as np

from quant_rl.config.schema import ExecutionConfig
from quant_rl.data.mt5_source import SymbolSpec
from quant_rl.execution.sizing import compute_pip_value_per_lot
from quant_rl.execution.financing import financing_cost
from quant_rl.execution.market_safety import market_context_asof
from quant_rl.execution.examiner import (
    ExaminerVerdict,
    OpenPositionSnapshot,
    PortfolioStateSnapshot,
    PreTradeExaminer,
)
from quant_rl.contracts import ActionProposal, TradeDirection
from quant_rl.market.observations import MarketObservationDataset


@dataclass(frozen=True)
class ExecutedTradeRecord:
    """Complete economic record of an executed trade from entry to exit."""

    trade_id: int
    symbol: str
    timeframe: str
    direction: TradeDirection
    action_index: int
    action_name: str
    entry_bar_index: int
    exit_bar_index: int
    entry_time_s: int
    exit_time_s: int
    holding_bars: int
    holding_days: float
    volume_lots: float
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_reason: str  # "TP" | "SL" | "SL_GAP" | "AMBIGUOUS_SL"
    requested_risk_fraction: float
    initial_risk_account_ccy: float
    gross_pnl_account_ccy: float
    spread_cost_account_ccy: float
    commission_cost_account_ccy: float
    slippage_cost_account_ccy: float
    financing_cost_account_ccy: float
    total_cost_account_ccy: float
    net_pnl_account_ccy: float
    net_r_multiple: float
    mfe_r: float
    mae_r: float
    entry_pip_value_per_lot_account_ccy: float = 0.0
    exit_pip_value_per_lot_account_ccy: float = 0.0
    financing_rollover_units: int = 0
    financing_source: str = "MODELED_CONFIG"


@dataclass(frozen=True)
class DecisionEvent:
    """One eligible policy decision and its final economic learning signal."""
    bar_index: int
    decision_time_s: int
    action: str
    status: str
    reward_account_ccy: float
    resolved_bar_index: int | None
    balance_at_decision: float
    equity_at_decision: float
    spread_penalty_account_ccy: float = 0.0
    veto_codes: tuple[str, ...] = ()
    veto_reason: str = ""
    failed_check_details: tuple[tuple[str, str], ...] = ()
    measured: tuple[tuple[str, float], ...] = ()
    observed_features: tuple[tuple[str, float], ...] = ()
    observed_spread_pips: float = 0.0
    observed_volatility_price: float = 0.0
    proposed_stop_units: float = 0.0
    proposed_target_units: float = 0.0
    proposed_risk_fraction: float = 0.0
    # True only if the trade exited before the evaluation boundary.
    resolved: bool = True
    # Available, marked-to-market reward at the research boundary for a
    # censored position. This is NOT an executed close or realized cash P&L.
    # None means no terminal valuation has been observed for this decision.
    terminal_mark_to_market_account_ccy: float | None = None


def summarize_veto_context(events) -> dict[str, dict]:
    """One reusable, purely observed diagnostic for research and reporting.

    All failed checks are counted (not just the first veto). Historical
    context describes why proposals failed; it does not relax the Examiner.
    """
    by_code: dict[str, list[DecisionEvent]] = {}
    for event in events:
        if event.status == "VETO":
            for code in event.veto_codes:
                by_code.setdefault(code, []).append(event)
    results = {}
    for code, failed in sorted(by_code.items()):
        n = len(failed)
        def average(field):
            return sum(float(getattr(e, field)) for e in failed) / n
        stats = {
            'failure_count':n,
            'mean_observed_spread_pips':average('observed_spread_pips'),
            'mean_observed_volatility_price':average('observed_volatility_price'),
            'mean_proposed_stop_units':average('proposed_stop_units'),
            'mean_proposed_target_units':average('proposed_target_units'),
            'mean_proposed_risk_fraction':average('proposed_risk_fraction'),
        }
        observations = [dict(e.measured) for e in failed]
        for field in ('round_trip_cost_pips', 'allowed_cost_pips',
                      'raw_requested_lots', 'broker_min_volume_lots',
                      'broker_volume_step_lots', 'requested_stop_pips',
                      'max_feasible_stop_pips_at_requested_risk',
                      'required_risk_fraction_for_min_lot_at_requested_stop',
                      'volume_shortfall_lots'):
            values = [float(row[field]) for row in observations if field in row]
            if values:
                stats['mean_' + field] = sum(values) / len(values)
        results[code] = stats
    return results


@dataclass(frozen=True)
class PortfolioSimulationResult:
    """Chronological portfolio simulation output and business scorecard metrics."""

    starting_balance: float
    ending_balance: float
    ending_equity: float
    net_return_pct: float
    compounded_annual_growth_pct: float
    max_drawdown_pct: float
    calmar_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    mean_net_r: float
    trades_per_year: float
    capital_utilization_pct: float
    abstention_rate: float
    veto_count: int
    veto_breakdown: dict[str, int]
    executed_trades: tuple[ExecutedTradeRecord, ...]
    equity_curve: np.ndarray
    time_s_curve: np.ndarray
    bar_decisions: tuple[tuple[int, str, str], ...]  # (bar_idx, action_name, status)
    open_positions_at_end: tuple[OpenPositionSnapshot, ...] = ()
    decision_events: tuple[DecisionEvent, ...] = ()
    learning_reward_account_ccy: float = 0.0
    penalty_total_account_ccy: float = 0.0
    terminal_halt_reason: str | None = None
    terminal_halt_bar_index: int | None = None



class CausalMarketContextCache:
    """Dataset-scoped market references shared across immutable PPO replays.

    Only historical market state is memoized: never account state, proposal
    outcomes, execution costs, or Examiner verdicts. Wrong-source use fails.
    """
    def __init__(self, observations: MarketObservationDataset, cfg: ExecutionConfig):
        self.dataset = observations
        self.parameters = self._parameters(cfg)
        self.values: dict[int, object] = {}

    @staticmethod
    def _parameters(cfg):
        return (cfg.market_history_bars, cfg.market_min_history_bars,
                cfg.market_window_candidates, cfg.market_regime_shift_ratio)

    def compatible(self, observations: MarketObservationDataset, cfg: ExecutionConfig) -> bool:
        return self.dataset is observations and self.parameters == self._parameters(cfg)

class PortfolioAccountingSimulator:
    """
    Simulates chronological bar-by-bar execution with:
    - Fixed post-entry TP/SL levels (no time-based expiry)
    - Intrabar M1 resolution when available (and conservative SL tie-breaking otherwise)
    - Explicit spread, commission, and multi-day rollover financing charges
    - Right-censoring at chronological split boundaries so evaluation windows never leak future bars
    """

    def __init__(self, execution_cfg: ExecutionConfig, max_risk_per_trade_pct: float = 1.0, max_drawdown_pct: float = 20.0) -> None:
        self.cfg = execution_cfg
        self.examiner = PreTradeExaminer(
            execution_cfg, max_risk_per_trade_pct=max_risk_per_trade_pct,
            max_drawdown_pct=max_drawdown_pct,
        )

    def simulate(
        self,
        observations: MarketObservationDataset,
        proposals: list[ActionProposal] | Callable[[int, PortfolioStateSnapshot], ActionProposal],
        symbol_spec: SymbolSpec,
        *,
        cost_multiplier: float = 1.0,
        m1_time_s: np.ndarray | None = None,
        m1_high: np.ndarray | None = None,
        m1_low: np.ndarray | None = None,
        m1_spread_points: np.ndarray | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        inactivity_penalty_lots: float = 0.01,
        market_context_cache: CausalMarketContextCache | None = None,
        deadline_monotonic: float | None = None,
    ) -> PortfolioSimulationResult:
        dynamic_policy = callable(proposals)
        proposal_by_bar: dict[int, ActionProposal] = {}
        if dynamic_policy:
            first_bar = int(start_index) if start_index is not None else 0
            last_bar = int(end_index) if end_index is not None else observations.num_bars - 1
        else:
            for p in proposals:
                if p.symbol != observations.symbol or p.bar_index < 0 or p.bar_index >= observations.num_bars:
                    raise ValueError("Proposal symbol/index mismatch")
                if p.decision_time_s != int(observations.decision_time_s[p.bar_index]):
                    raise ValueError("Proposal must use completed-bar decision timestamp")
                if p.bar_index in proposal_by_bar:
                    raise ValueError(f"Duplicate proposal on bar {p.bar_index}")
                proposal_by_bar[p.bar_index] = p
            if proposal_by_bar:
                first_bar = min(proposal_by_bar)
                last_bar = max(proposal_by_bar)
            else:
                first_bar = int(start_index) if start_index is not None else 0
                last_bar = int(end_index) if end_index is not None else observations.num_bars - 1
        if first_bar < 0 or last_bar >= observations.num_bars or first_bar > last_bar:
            raise ValueError("Invalid chronological simulation bounds")
        if observations.bar_price_side not in ("MID", "BID"):
            raise ValueError("Historical quote side must be MID or BID")
        if market_context_cache is not None and not market_context_cache.compatible(observations, self.cfg):
            raise ValueError("Market context cache belongs to a different dataset or safety configuration")
        # Validate M1 ordering once per replay. Rechecking np.diff over the
        # complete multi-year minute array for every dual-touch bar is O(N)
        # per event and can dominate the replay runtime.
        m1_ordered = (
            m1_time_s is not None and m1_high is not None and m1_low is not None
            and len(m1_time_s) > 0
            and len(m1_time_s) == len(m1_high) == len(m1_low)
            and (m1_spread_points is None or len(m1_spread_points) == len(m1_time_s))
            and bool(np.all(np.diff(m1_time_s) > 0))
        )
        sorted_bars = range(first_bar, last_bar + 1)
        # Convert the full source array once; accessing spread_pips repeatedly
        # constructs a fresh array and scales poorly with long historical windows.
        spreads = observations.spread_pips
        # Tick value captured in broker metadata may reflect an older exchange
        # rate. Derive a point-in-time account-currency pip value at each decision.
        if inactivity_penalty_lots <= 0 or not math.isfinite(inactivity_penalty_lots):
            raise ValueError("Penalty reference volume must be strictly positive")
        decision_events: list[DecisionEvent] = []
        pending_event_idx: int | None = None
        penalty_total = 0.0
        terminal_halt_reason: str | None = None
        terminal_halt_bar_index: int | None = None

        balance = float(self.cfg.starting_balance)
        equity = balance
        peak_equity = balance
        day_start_equity = balance
        week_start_equity = balance
        current_day = -1
        current_week = -1

        tf_sec = 14400
        if len(observations.time_s) >= 2:
            diffs = np.diff(observations.time_s)
            tf_sec = int(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 14400

        open_pos: dict | None = None
        consecutive_losses = 0
        trade_seq = 0
        executed_trades: list[ExecutedTradeRecord] = []
        veto_breakdown: dict[str, int] = {}
        veto_count = 0
        abstain_count = 0
        occupied_bars = 0
        bar_decisions: list[tuple[int, str, str]] = []

        equity_curve = np.zeros(len(sorted_bars), dtype=np.float64)
        time_curve = np.zeros(len(sorted_bars), dtype=np.int64)

        for step_i, bar_idx in enumerate(sorted_bars):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError('Research session deadline reached during economic replay')
            closed_this_bar = False
            t_s = int(observations.time_s[bar_idx])
            decision_time_s = t_s + tf_sec
            dt = datetime.fromtimestamp(decision_time_s, tz=timezone.utc)
            day_key = dt.timetuple().tm_yday + dt.year * 1000
            iso = dt.isocalendar()
            week_key = iso.week + iso.year * 100
            if day_key != current_day:
                current_day = day_key
                day_start_equity = equity
            if week_key != current_week:
                current_week = week_key
                week_start_equity = equity

            bar_open = float(observations.open[bar_idx])
            bar_high = float(observations.high[bar_idx])
            bar_low = float(observations.low[bar_idx])
            bar_close = float(observations.close[bar_idx])
            bar_spread_pips = float(spreads[bar_idx]) * cost_multiplier
            half_spread = 0.5 * bar_spread_pips * symbol_spec.pip_size
            # MT5 OHLC values are BID quotes. Tests may exercise a separate MID
            # contract, but production research accepts only authenticated BID.
            bid_offset = 0.0 if observations.bar_price_side == "BID" else -half_spread
            ask_offset = 2.0 * half_spread if observations.bar_price_side == "BID" else half_spread
            current_mid = bar_close + (half_spread if observations.bar_price_side == "BID" else 0.0)

            # 1. Monitor existing open position on current bar
            if open_pos is not None:
                occupied_bars += 1
                direction: TradeDirection = open_pos["direction"]
                entry_price = open_pos["entry_price"]
                stop_price = open_pos["stop_price"]
                target_price = open_pos["target_price"]
                stop_dist = abs(entry_price - stop_price)
                prior_mfe_r = open_pos["mfe_r"]
                prior_mae_r = open_pos["mae_r"]

                if direction == TradeDirection.BUY:
                    # Bid prices trigger BUY exits
                    bid_high = bar_high + bid_offset
                    bid_low = bar_low + bid_offset
                    bid_close = bar_close + bid_offset
                    bid_open = bar_open + bid_offset
                    open_pos["mfe_r"] = max(open_pos["mfe_r"], (bid_high - entry_price) / max(stop_dist, 1e-9))
                    open_pos["mae_r"] = max(open_pos["mae_r"], (entry_price - bid_low) / max(stop_dist, 1e-9))
                    hit_sl = bid_low <= stop_price
                    hit_tp = bid_high >= target_price
                    mark_price = bid_close
                else:
                    # Ask prices trigger SELL exits
                    ask_high = bar_high + ask_offset
                    ask_low = bar_low + ask_offset
                    ask_close = bar_close + ask_offset
                    ask_open = bar_open + ask_offset
                    open_pos["mfe_r"] = max(open_pos["mfe_r"], (entry_price - ask_low) / max(stop_dist, 1e-9))
                    open_pos["mae_r"] = max(open_pos["mae_r"], (ask_high - entry_price) / max(stop_dist, 1e-9))
                    hit_sl = ask_high >= stop_price
                    hit_tp = ask_low <= target_price
                    mark_price = ask_close

                exit_reason: str | None = None
                exit_price = mark_price
                if hit_sl and ((direction == TradeDirection.BUY and bid_open <= stop_price)
                               or (direction == TradeDirection.SELL and ask_open >= stop_price)):
                    # Adverse gaps fill at the first executable quote beyond the stop.
                    exit_reason = "SL_GAP"
                    exit_price = bid_open if direction == TradeDirection.BUY else ask_open
                elif hit_tp and ((direction == TradeDirection.BUY and bid_open >= target_price)
                                 or (direction == TradeDirection.SELL and ask_open <= target_price)):
                    # A favorable opening gap reaches the take-profit limit
                    # BEFORE any later intrabar stop touch. Do not infer a
                    # later loss from the candle's unordered high/low.
                    exit_reason = "TP"
                    exit_price = target_price  # Conservative: no positive gap improvement.
                elif hit_sl and hit_tp:
                    resolved = self._resolve_dual_touch_m1(
                        direction=direction,
                        stop_price=stop_price,
                        target_price=target_price,
                        half_spread=half_spread,
                        bar_price_side=observations.bar_price_side,
                        bar_start_s=t_s,
                        bar_end_s=t_s + tf_sec,
                        m1_time_s=m1_time_s,
                        m1_high=m1_high,
                        m1_low=m1_low,
                        m1_spread_points=m1_spread_points,
                        point=symbol_spec.point,
                        cost_multiplier=cost_multiplier,
                        m1_ordered=m1_ordered,
                    )
                    exit_reason = resolved
                    exit_price = target_price if resolved == "TP" else stop_price
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = stop_price
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = target_price

                if exit_reason is not None:
                    if exit_reason in ("SL", "AMBIGUOUS_SL", "SL_GAP"):
                        # Do not use extremes from after the exit within the bar.
                        open_pos["mfe_r"] = prior_mfe_r
                        open_pos["mae_r"] = max(prior_mae_r, abs(entry_price - exit_price) / max(stop_dist, 1e-9))
                    elif exit_reason == "TP":
                        open_pos["mae_r"] = prior_mae_r
                        open_pos["mfe_r"] = max(prior_mfe_r, abs(exit_price - entry_price) / max(stop_dist, 1e-9))
                    record = self._close_position(
                        open_pos=open_pos,
                        exit_bar_index=bar_idx,
                        # A bar-level OHLC observation does not establish the actual
                        # intrabar execution minute; book at completed-bar time,
                        # consistently for the stateful simulator and learning replay.
                        exit_time_s=decision_time_s,
                        exit_price=exit_price,
                        exit_spread_pips=bar_spread_pips,
                        exit_reason=exit_reason,
                        symbol_spec=symbol_spec,
                        cost_multiplier=cost_multiplier,
                    )
                    executed_trades.append(record)
                    closed_this_bar = True
                    if pending_event_idx is not None:
                        decision_events[pending_event_idx] = replace(
                            decision_events[pending_event_idx],
                            status="CLOSED", reward_account_ccy=record.net_pnl_account_ccy,
                            resolved_bar_index=bar_idx, resolved=True,
                        )
                        pending_event_idx = None
                    balance += record.net_pnl_account_ccy
                    equity = balance
                    peak_equity = max(peak_equity, equity)
                    if record.net_pnl_account_ccy < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    open_pos = None
                    # C13's peak-to-valley drawdown and C14's consecutive-loss
                    # breaker cannot clear without another executed position or
                    # an explicit operator reset. A trading policy has no valid
                    # further decision while these hard locks are engaged.
                    # Preserve the remaining marked-equity curve but do not
                    # fabricate repeated vetoes and spread penalties.
                    total_dd = 100.0 * (peak_equity-equity)/max(peak_equity,1.)
                    if total_dd >= self.examiner.max_drawdown_pct:
                        terminal_halt_reason = 'C13_MAX_DRAWDOWN'
                    elif consecutive_losses >= self.cfg.max_consecutive_losses:
                        terminal_halt_reason = 'C14_CONSECUTIVE_LOSSES'
                    if terminal_halt_reason is not None:
                        terminal_halt_bar_index = bar_idx
                else:
                    # Mark-to-market unrealized PnL
                    price_diff = (
                        (mark_price - entry_price)
                        if direction == TradeDirection.BUY
                        else (entry_price - mark_price)
                    )
                    pips_unrealized = price_diff / symbol_spec.pip_size
                    mark_pip_value = compute_pip_value_per_lot(
                        symbol_spec, mark_price, self.cfg.account_currency)
                    unrealized = (
                        pips_unrealized * mark_pip_value * open_pos["volume_lots"]
                        - open_pos["commission_cost"]
                        - (self.cfg.simulated_slippage_points_per_side*symbol_spec.point /
                           symbol_spec.pip_size*mark_pip_value*
                           open_pos['volume_lots']*cost_multiplier)
                        - financing_cost(open_pos['entry_time_s'], decision_time_s,
                                         open_pos['volume_lots'], direction,
                                         symbol_spec, self.cfg,
                                         cost_multiplier=cost_multiplier)[0]
                    )
                    equity = balance + unrealized
                    peak_equity = max(peak_equity, equity)

            # 2. Evaluate new proposal at bar close if not at the very last bar of the split
            if (dynamic_policy and open_pos is None and not closed_this_bar
                    and terminal_halt_reason is None and bar_idx < last_bar):
                policy_state = PortfolioStateSnapshot(
                    current_time_s=decision_time_s, balance=balance, equity=equity,
                    peak_equity=peak_equity, day_start_equity=day_start_equity,
                    week_start_equity=week_start_equity, open_positions=(),
                    consecutive_losses=consecutive_losses, last_bar_time_s=decision_time_s,
                    timeframe_seconds=tf_sec,
                )
                proposal = proposals(bar_idx, policy_state)
                if proposal is None:
                    raise ValueError("Eligible policy callback must return a decision")
                if proposal.symbol != observations.symbol or proposal.bar_index != bar_idx or proposal.decision_time_s != decision_time_s:
                    raise ValueError("Policy proposal symbol/index/timestamp mismatch")
            else:
                proposal = proposal_by_bar.get(bar_idx) if not dynamic_policy else None
            if dynamic_policy and (open_pos is not None or closed_this_bar or terminal_halt_reason is not None):
                assert proposal is None, "Decision generated while position occupied"
            # No eligible decision while a position is open, after a close on
            # this candle, or after a hard lockout. Do not materialize a large
            # diagnostic feature tuple when there is no proposed action.
            if proposal is None:
                equity_curve[step_i] = equity
                time_curve[step_i] = t_s
                continue
            penalty_pip_value = compute_pip_value_per_lot(
                symbol_spec, current_mid, self.cfg.account_currency)
            penalty = (float(spreads[bar_idx])
                       * penalty_pip_value * inactivity_penalty_lots)
            observed_features = tuple(zip(observations.feature_names,
                                      (float(v) for v in observations.feature_matrix[bar_idx])))
            observed_spread = float(spreads[bar_idx])
            observed_volatility = float(observations.volatility_price[bar_idx])
            if proposal.direction == TradeDirection.NO_TRADE:
                abstain_count += 1
                penalty_total += penalty
                decision_events.append(DecisionEvent(
                    bar_index=bar_idx, decision_time_s=decision_time_s,
                    action="NO_TRADE", status="ABSTAIN", reward_account_ccy=-penalty,
                    resolved_bar_index=bar_idx, balance_at_decision=balance,
                    equity_at_decision=equity, spread_penalty_account_ccy=penalty,
                    observed_features=observed_features,
                    observed_spread_pips=observed_spread,
                    observed_volatility_price=observed_volatility))
                bar_decisions.append((bar_idx, "NO_TRADE", "ABSTAIN"))
            elif bar_idx < last_bar:
                open_snapshots: tuple[OpenPositionSnapshot, ...] = ()
                if open_pos is not None:
                    open_snapshots = (
                        OpenPositionSnapshot(
                            position_id=open_pos["trade_id"],
                            symbol=open_pos["symbol"],
                            direction=open_pos["direction"],
                            volume_lots=open_pos["volume_lots"],
                            entry_price=open_pos["entry_price"],
                            stop_price=open_pos["stop_price"],
                            target_price=open_pos["target_price"],
                            entry_time_s=open_pos["entry_time_s"],
                            open_risk_account_ccy=open_pos["initial_risk_account_ccy"],
                            margin_used_account_ccy=open_pos["margin_used"],
                            unrealized_pnl_account_ccy=equity - balance,
                        ),
                    )
                portfolio_snap = PortfolioStateSnapshot(
                    current_time_s=decision_time_s,
                    balance=balance,
                    equity=equity,
                    peak_equity=peak_equity,
                    day_start_equity=day_start_equity,
                    week_start_equity=week_start_equity,
                    open_positions=open_snapshots,
                    consecutive_losses=consecutive_losses,
                    last_bar_time_s=decision_time_s,
                    timeframe_seconds=tf_sec,
                )
                verdict: ExaminerVerdict = self.examiner.inspect(
                    proposal=proposal,
                    portfolio=portfolio_snap,
                    symbol_spec=symbol_spec,
                    mid_price=current_mid,
                    spread_pips=float(spreads[bar_idx]),
                    volatility_price=float(observations.volatility_price[bar_idx]),
                    cost_multiplier=cost_multiplier,
                    market_context=self._market_context_cached(
                        observations, bar_idx, spreads, market_context_cache),
                )
                if verdict.approved:
                    trade_seq += 1
                    pending_event_idx = len(decision_events)
                    decision_events.append(DecisionEvent(
                        bar_index=bar_idx, decision_time_s=decision_time_s,
                        action=proposal.action_name, status="OPEN", reward_account_ccy=0.0,
                        resolved_bar_index=None, balance_at_decision=balance,
                        equity_at_decision=equity, resolved=False,
                        observed_features=observed_features,
                        observed_spread_pips=observed_spread,
                        observed_volatility_price=observed_volatility,
                        proposed_stop_units=proposal.stop_units,
                        proposed_target_units=proposal.target_units,
                        proposed_risk_fraction=proposal.requested_risk_fraction))
                    pip_val = (
                        verdict.actual_risk_account_ccy
                        / max(
                            (proposal.stop_distance_price / symbol_spec.pip_size)
                            * verdict.approved_volume_lots,
                            1e-9,
                        )
                    )
                    spread_cost = (
                        float(spreads[bar_idx])
                        * cost_multiplier
                        * pip_val
                        * verdict.approved_volume_lots
                    )
                    comm_cost = (
                        self.cfg.commission_per_lot_round_trip
                        * cost_multiplier
                        * verdict.approved_volume_lots
                    )
                    open_pos = {
                        "trade_id": trade_seq,
                        "symbol": proposal.symbol,
                        "timeframe": proposal.timeframe,
                        "direction": proposal.direction,
                        "action_index": proposal.action_index,
                        "action_name": proposal.action_name,
                        "entry_bar_index": bar_idx,
                        "entry_time_s": decision_time_s,
                        "volume_lots": verdict.approved_volume_lots,
                        "entry_price": verdict.entry_price + (1 if proposal.direction==TradeDirection.BUY else -1)
                            *self.cfg.simulated_slippage_points_per_side*symbol_spec.point*cost_multiplier,
                        "stop_price": verdict.stop_price + (1 if proposal.direction==TradeDirection.BUY else -1)
                            *self.cfg.simulated_slippage_points_per_side*symbol_spec.point*cost_multiplier,
                        "target_price": verdict.target_price + (1 if proposal.direction==TradeDirection.BUY else -1)
                            *self.cfg.simulated_slippage_points_per_side*symbol_spec.point*cost_multiplier,
                        "half_entry_spread_pips": .5*float(spreads[bar_idx])*cost_multiplier,

                        "requested_risk_fraction": proposal.requested_risk_fraction,
                        "initial_risk_account_ccy": verdict.actual_risk_account_ccy,
                        "margin_used": verdict.required_margin_account_ccy,
                        "pip_value_per_lot": pip_val,
                        "spread_cost": spread_cost,
                        "commission_cost": comm_cost,
                        "mfe_r": 0.0,
                        "mae_r": 0.0,
                    }
                    # Include execution friction in equity immediately. Entry is
                    # recorded at the ask/bid and exit marking at the opposite quote.
                    slippage_entry_cost=(self.cfg.simulated_slippage_points_per_side*symbol_spec.point/
                                         symbol_spec.pip_size*pip_val*verdict.approved_volume_lots*cost_multiplier)
                    equity = balance - spread_cost - comm_cost - slippage_entry_cost
                    peak_equity = max(peak_equity, equity)
                    bar_decisions.append((bar_idx, proposal.action_name, "EXECUTED"))
                else:
                    veto_count += 1
                    code = verdict.veto_code or "UNKNOWN"
                    all_failed = tuple(k for k, passed in verdict.checks_passed.items() if not passed)
                    penalty_total += penalty
                    decision_events.append(DecisionEvent(
                        bar_index=bar_idx, decision_time_s=decision_time_s,
                        action=proposal.action_name, status="VETO", reward_account_ccy=-penalty,
                        resolved_bar_index=bar_idx, balance_at_decision=balance,
                        equity_at_decision=equity, spread_penalty_account_ccy=penalty,
                        veto_codes=all_failed, veto_reason=verdict.veto_reason,
                        failed_check_details=verdict.failed_check_details,
                        observed_features=observed_features,
                        observed_spread_pips=observed_spread,
                        observed_volatility_price=observed_volatility,
                        proposed_stop_units=proposal.stop_units,
                        proposed_target_units=proposal.target_units,
                        proposed_risk_fraction=proposal.requested_risk_fraction,
                        measured=(("round_trip_cost_pips",verdict.round_trip_cost_pips),
                                  ("allowed_cost_pips",verdict.allowed_cost_pips),
                                  ("raw_requested_lots",verdict.raw_requested_lots),
                                  ("broker_min_volume_lots",symbol_spec.volume_min),
                                  ("broker_volume_step_lots",symbol_spec.volume_step),
                                  ("requested_stop_pips",proposal.stop_distance_price / symbol_spec.pip_size),
                                  # C16 counterfactual affordability is diagnostic
                                  # only; neither broker limits nor the chosen
                                  # trading action are silently adjusted.
                                  ("max_feasible_stop_pips_at_requested_risk",
                                   (proposal.stop_distance_price / symbol_spec.pip_size)
                                   * verdict.raw_requested_lots / symbol_spec.volume_min),
                                  ("required_risk_fraction_for_min_lot_at_requested_stop",
                                   verdict.actual_risk_account_ccy / max(equity,1e-9)),
                                  ("volume_shortfall_lots",
                                   max(0.,symbol_spec.volume_min-verdict.raw_requested_lots)),
                                  ("requested_stop_price",proposal.stop_distance_price),
                                  ("requested_target_price",proposal.target_distance_price),
                                  ("requested_risk_fraction",proposal.requested_risk_fraction),
                                  *verdict.adaptive_limits)))
                    veto_breakdown[code] = veto_breakdown.get(code, 0) + 1
                    bar_decisions.append((bar_idx, proposal.action_name, f"VETO_{code}"))

            equity_curve[step_i] = equity
            time_curve[step_i] = t_s

        # A position held at the right-censored history boundary has a real
        # account-equity effect. Make this *observable* terminal valuation
        # available to learning without fabricating TP/SL execution, changing
        # balance, or marking the trade as resolved.  Previously the learner
        # silently ignored these open losses/profits and optimizers could
        # favor an unresolved position over an otherwise costly abstention.
        if open_pos is not None:
            if pending_event_idx is None:
                raise AssertionError('Open position missing its entry decision')
            mark = equity - balance
            decision_events[pending_event_idx] = replace(
                decision_events[pending_event_idx],
                reward_account_ccy=mark,
                resolved_bar_index=last_bar,
                terminal_mark_to_market_account_ccy=mark,
            )

        return self._summarize(
            starting_balance=self.cfg.starting_balance,
            ending_balance=balance,
            ending_equity=equity,
            equity_curve=equity_curve,
            time_curve=time_curve,
            executed_trades=executed_trades,
            total_bars=len(sorted_bars),
            occupied_bars=occupied_bars,
            abstain_count=abstain_count,
            veto_count=veto_count,
            veto_breakdown=veto_breakdown,
            bar_decisions=bar_decisions,
            decision_events=tuple(decision_events),
            learning_reward_account_ccy=(equity - float(self.cfg.starting_balance)) - penalty_total,
            penalty_total_account_ccy=penalty_total,
            terminal_halt_reason=terminal_halt_reason,
            terminal_halt_bar_index=terminal_halt_bar_index,
            open_positions_at_end=(
                (OpenPositionSnapshot(
                    position_id=open_pos["trade_id"], symbol=open_pos["symbol"],
                    direction=open_pos["direction"], volume_lots=open_pos["volume_lots"],
                    entry_price=open_pos["entry_price"], stop_price=open_pos["stop_price"],
                    target_price=open_pos["target_price"], entry_time_s=open_pos["entry_time_s"],
                    open_risk_account_ccy=open_pos["initial_risk_account_ccy"],
                    margin_used_account_ccy=open_pos["margin_used"],
                    unrealized_pnl_account_ccy=equity - balance,
                ),) if open_pos is not None else ()
            ),
        )

    @staticmethod
    def _resolve_dual_touch_m1(
        *,
        direction: TradeDirection,
        stop_price: float,
        target_price: float,
        half_spread: float,
        bar_price_side: str = "MID",
        bar_start_s: int,
        bar_end_s: int,
        m1_time_s: np.ndarray | None,
        m1_high: np.ndarray | None,
        m1_low: np.ndarray | None,
        m1_spread_points: np.ndarray | None = None,
        point: float = 0.00001,
        cost_multiplier: float = 1.0,
        m1_ordered: bool | None = None,
    ) -> str:
        """Resolve same-bar TP & SL touch using M1 sub-bars if available; otherwise conservative AMBIGUOUS_SL."""
        if m1_spread_points is not None and (m1_time_s is None or
                len(m1_spread_points) != len(m1_time_s)):
            return "AMBIGUOUS_SL"
        if (m1_time_s is not None and m1_high is not None and m1_low is not None
                and len(m1_time_s) == len(m1_high) == len(m1_low)
                and len(m1_time_s) > 0
                and (m1_ordered if m1_ordered is not None else np.all(np.diff(m1_time_s) > 0))
                and bar_end_s > bar_start_s):
            i0 = int(np.searchsorted(m1_time_s, bar_start_s, side="left"))
            i1 = int(np.searchsorted(m1_time_s, bar_end_s, side="left"))
            expected_t = bar_start_s
            for idx in range(i0, i1):
                # Missing any preceding minute makes the first-hit sequence unknowable.
                # Only a contiguous prefix up to the first hit can justify TP.
                if int(m1_time_s[idx]) != expected_t:
                    return "AMBIGUOUS_SL"
                expected_t += 60
                if not (np.isfinite(m1_high[idx]) and np.isfinite(m1_low[idx])
                        and m1_high[idx] >= m1_low[idx]):
                    return "AMBIGUOUS_SL"
                # When supplied, use the spread actually reported at this M1
                # minute, not the *later* H4 decision spread. Absent/invalid
                # minute spreads cannot establish a favorable TP-first result.
                minute_half_spread = half_spread
                if m1_spread_points is not None:
                    spread_points = float(m1_spread_points[idx])
                    if not np.isfinite(spread_points) or spread_points <= 0:
                        return "AMBIGUOUS_SL"
                    minute_half_spread = 0.5 * spread_points * point * cost_multiplier
                if direction == TradeDirection.BUY:
                    sub_low = float(m1_low[idx]) + (0.0 if bar_price_side == "BID" else -minute_half_spread)
                    sub_high = float(m1_high[idx]) + (0.0 if bar_price_side == "BID" else -minute_half_spread)
                    sl_hit = sub_low <= stop_price
                    tp_hit = sub_high >= target_price
                else:
                    sub_high = float(m1_high[idx]) + (2.0 * minute_half_spread if bar_price_side == "BID" else minute_half_spread)
                    sub_low = float(m1_low[idx]) + (2.0 * minute_half_spread if bar_price_side == "BID" else minute_half_spread)
                    sl_hit = sub_high >= stop_price
                    tp_hit = sub_low <= target_price
                if sl_hit and tp_hit:
                    return "AMBIGUOUS_SL"
                if sl_hit:
                    return "SL"
                if tp_hit:
                    return "TP"
        return "AMBIGUOUS_SL"

    def _market_context_cached(self, observations, bar_idx, spreads, cache):
        """Memoize only historical, pre-decision safety references.

        The caller owns cache scope and must not share it across distinct
        datasets or market-safety configurations. No future bars are read.
        """
        if cache is not None and bar_idx in cache.values:
            return cache.values[bar_idx]
        context = market_context_asof(
            observations, bar_idx, lookback_bars=self.cfg.market_history_bars,
            min_history_bars=self.cfg.market_min_history_bars,
            candidate_windows=self.cfg.market_window_candidates,
            regime_shift_ratio=self.cfg.market_regime_shift_ratio,
            spread_values=spreads,
        )
        if cache is not None:
            cache.values[bar_idx] = context
        return context

    def _close_position(
        self,
        *,
        open_pos: dict,
        exit_bar_index: int,
        exit_time_s: int,
        exit_price: float,
        exit_spread_pips: float,
        exit_reason: str,
        symbol_spec: SymbolSpec,
        cost_multiplier: float,
    ) -> ExecutedTradeRecord:
        direction: TradeDirection = open_pos["direction"]
        entry_price = float(open_pos["entry_price"])
        volume_lots = float(open_pos["volume_lots"])
        entry_pip_val = float(open_pos["pip_value_per_lot"])

        exit_slip = self.cfg.simulated_slippage_points_per_side*symbol_spec.point*cost_multiplier
        # Adverse slippage on both entry and exit; financial ledger never assumes favorable fills.
        exit_price = exit_price + (-exit_slip if direction==TradeDirection.BUY else exit_slip)
        price_diff = (exit_price-entry_price) if direction==TradeDirection.BUY else (entry_price-exit_price)
        pips_gained = price_diff / symbol_spec.pip_size
        # Convert the actual closing quote-currency cash flow into account
        # currency at the execution price, rather than using the entry-date
        # conversion rate. The entry pip value is still the correct basis for
        # the initial stop-risk budget recorded in the Examiner.
        exit_pip_val = compute_pip_value_per_lot(
            symbol_spec, exit_price, self.cfg.account_currency)
        # Note: entry_price already includes half-spread and exit_price includes the other half-spread,
        # so price_diff * pip_val * lots already incorporates the bid-ask spread.
        # To report transparent gross vs cost components:
        spread_cost=(float(open_pos['half_entry_spread_pips'])+.5*exit_spread_pips)*exit_pip_val*volume_lots
        slippage_cost=2*self.cfg.simulated_slippage_points_per_side*symbol_spec.point / symbol_spec.pip_size*exit_pip_val*volume_lots*cost_multiplier
        gross_pnl = pips_gained * exit_pip_val * volume_lots + spread_cost + slippage_cost
        commission_cost = float(open_pos["commission_cost"])
        holding_days = max(0.0, (exit_time_s - int(open_pos["entry_time_s"])) / 86400.0)
        accrued_financing, rollover_units, financing_source = financing_cost(
            int(open_pos["entry_time_s"]), exit_time_s, volume_lots,
            direction, symbol_spec, self.cfg, cost_multiplier=cost_multiplier)

        total_cost = spread_cost + slippage_cost + commission_cost + accrued_financing
        net_pnl = gross_pnl - total_cost
        init_risk = max(float(open_pos["initial_risk_account_ccy"]), 1e-6)
        net_r = net_pnl / init_risk

        return ExecutedTradeRecord(
            trade_id=int(open_pos["trade_id"]),
            symbol=str(open_pos["symbol"]),
            timeframe=str(open_pos["timeframe"]),
            direction=direction,
            action_index=int(open_pos["action_index"]),
            action_name=str(open_pos["action_name"]),
            entry_bar_index=int(open_pos["entry_bar_index"]),
            exit_bar_index=exit_bar_index,
            entry_time_s=int(open_pos["entry_time_s"]),
            exit_time_s=exit_time_s,
            holding_bars=max(1, exit_bar_index - int(open_pos["entry_bar_index"])),
            holding_days=holding_days,
            volume_lots=volume_lots,
            entry_price=entry_price,
            stop_price=float(open_pos["stop_price"]),
            target_price=float(open_pos["target_price"]),
            exit_price=exit_price,
            exit_reason=exit_reason,
            requested_risk_fraction=float(open_pos["requested_risk_fraction"]),
            initial_risk_account_ccy=init_risk,
            gross_pnl_account_ccy=gross_pnl,
            spread_cost_account_ccy=spread_cost,
            commission_cost_account_ccy=commission_cost,
            slippage_cost_account_ccy=slippage_cost,
            financing_cost_account_ccy=accrued_financing,
            total_cost_account_ccy=total_cost,
            net_pnl_account_ccy=net_pnl,
            net_r_multiple=net_r,
            mfe_r=float(open_pos["mfe_r"]),
            mae_r=float(open_pos["mae_r"]),
            entry_pip_value_per_lot_account_ccy=entry_pip_val,
            exit_pip_value_per_lot_account_ccy=exit_pip_val,
            financing_rollover_units=rollover_units,
            financing_source=financing_source,
        )

    @staticmethod
    def _summarize(
        *,
        starting_balance: float,
        ending_balance: float,
        ending_equity: float,
        equity_curve: np.ndarray,
        time_curve: np.ndarray,
        executed_trades: list[ExecutedTradeRecord],
        total_bars: int,
        occupied_bars: int,
        abstain_count: int,
        veto_count: int,
        veto_breakdown: dict[str, int],
        bar_decisions: list[tuple[int, str, str]],
        open_positions_at_end: tuple[OpenPositionSnapshot, ...] = (),
        decision_events: tuple[DecisionEvent, ...] = (),
        learning_reward_account_ccy: float = 0.0,
        penalty_total_account_ccy: float = 0.0,
        terminal_halt_reason: str | None = None,
        terminal_halt_bar_index: int | None = None,
    ) -> PortfolioSimulationResult:
        net_return_pct = 100.0 * (ending_equity - starting_balance) / max(starting_balance, 1.0)
        elapsed_years = max((float(time_curve[-1] - time_curve[0]) / (365.25 * 86400.0)), 1.0 / 52.0)
        ratio = max(ending_equity / max(starting_balance, 1.0), 1e-6)
        cagr_pct = 100.0 * (ratio ** (1.0 / min(max(elapsed_years, 0.1), 50.0)) - 1.0)

        # The initial funded balance is the first high-water mark; otherwise a
        # loss in the very first recorded bar is silently erased from drawdown.
        peaks = np.maximum.accumulate(np.maximum(equity_curve, starting_balance))
        dd_series = 100.0 * (peaks - equity_curve) / np.maximum(peaks, 1.0)
        max_dd_pct = float(np.max(dd_series)) if len(dd_series) > 0 else 0.0
        calmar = cagr_pct / max(max_dd_pct, 0.5)

        if len(equity_curve) > 2:
            rets = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1.0)
            bars_per_year = max(len(equity_curve) / elapsed_years, 1.0)
            std_r = float(np.std(rets))
            downside = rets[rets < 0]
            down_std = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 1e-9
            sharpe = float(np.mean(rets) / std_r * math.sqrt(bars_per_year)) if std_r > 1e-12 else 0.0
            sortino = float(np.mean(rets) / max(down_std, 1e-9) * math.sqrt(bars_per_year))
        else:
            sharpe = 0.0
            sortino = 0.0

        n_trades = len(executed_trades)
        if n_trades > 0:
            wins = [t for t in executed_trades if t.net_pnl_account_ccy > 0]
            losses = [t for t in executed_trades if t.net_pnl_account_ccy <= 0]
            win_rate = len(wins) / n_trades
            gross_win = sum(t.net_pnl_account_ccy for t in wins)
            gross_loss = abs(sum(t.net_pnl_account_ccy for t in losses))
            profit_factor = gross_win / max(gross_loss, 1e-6)
            mean_net_r = float(np.mean([t.net_r_multiple for t in executed_trades]))
        else:
            win_rate = 0.0
            profit_factor = 0.0
            mean_net_r = 0.0

        trades_per_year = n_trades / elapsed_years
        utilization_pct = 100.0 * occupied_bars / max(total_bars, 1)
        abstention_rate = abstain_count / max(total_bars, 1)

        return PortfolioSimulationResult(
            starting_balance=starting_balance,
            ending_balance=ending_balance,
            ending_equity=ending_equity,
            net_return_pct=net_return_pct,
            compounded_annual_growth_pct=cagr_pct,
            max_drawdown_pct=max_dd_pct,
            calmar_ratio=calmar,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            win_rate=win_rate,
            profit_factor=profit_factor,
            mean_net_r=mean_net_r,
            trades_per_year=trades_per_year,
            capital_utilization_pct=utilization_pct,
            abstention_rate=abstention_rate,
            veto_count=veto_count,
            veto_breakdown=veto_breakdown,
            executed_trades=tuple(executed_trades),
            equity_curve=equity_curve,
            time_s_curve=time_curve,
            bar_decisions=tuple(bar_decisions),
            open_positions_at_end=open_positions_at_end,
            decision_events=decision_events,
            learning_reward_account_ccy=learning_reward_account_ccy,
            penalty_total_account_ccy=penalty_total_account_ccy,
            terminal_halt_reason=terminal_halt_reason,
            terminal_halt_bar_index=terminal_halt_bar_index,
        )
