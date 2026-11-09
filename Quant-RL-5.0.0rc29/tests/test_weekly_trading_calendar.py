from datetime import datetime, timezone

from quant_rl.execution.trading_calendar import weekly_fx_entry_blocked


def stamp(y, m, d, hour, minute=0):
    return int(datetime(y, m, d, hour, minute, tzinfo=timezone.utc).timestamp())


def test_friday_cutoff_is_inclusive_in_europe_athens_server_time():
    # Athens is UTC+2 on this November date: 20:59 UTC = 22:59 server time.
    assert not weekly_fx_entry_blocked(stamp(2026, 11, 6, 20, 59), server_timezone='Europe/Athens')
    assert weekly_fx_entry_blocked(stamp(2026, 11, 6, 21, 0), server_timezone='Europe/Athens')


def test_saturday_is_blocked_and_sunday_reopens_at_configured_server_hour():
    assert weekly_fx_entry_blocked(stamp(2026, 11, 7, 12), server_timezone='Europe/Athens')
    assert weekly_fx_entry_blocked(stamp(2026, 11, 8, 18, 59), server_timezone='Europe/Athens')
    assert not weekly_fx_entry_blocked(stamp(2026, 11, 8, 19, 0), server_timezone='Europe/Athens')


def test_dst_transition_is_applied_by_iana_timezone():
    # Athens is UTC+3 in July and UTC+2 in November. The same 20:00 UTC
    # instant maps to 23:00 in July and 22:00 in November.
    assert weekly_fx_entry_blocked(stamp(2026, 7, 3, 20), server_timezone='Europe/Athens')
    assert not weekly_fx_entry_blocked(stamp(2026, 11, 6, 20), server_timezone='Europe/Athens')


def test_weekdays_are_not_blocked_by_weekly_calendar():
    assert not weekly_fx_entry_blocked(stamp(2026, 11, 9, 12), server_timezone='Europe/Athens')


def test_examiner_c5_uses_the_same_server_timezone_cutoff():
    from quant_rl.config.schema import QuantRLConfig
    from quant_rl.execution.examiner import PortfolioStateSnapshot, PreTradeExaminer
    from tests._specs import default_symbol_spec
    from tests.helpers_current import obs_for_path, proposal_for

    obs = obs_for_path([(1.1, 1.101, 1.099, 1.1)] * 8)
    proposal = proposal_for(obs, idx=1, action_index=2, risk=.005)
    now_utc = stamp(2026, 11, 6, 21)  # 23:00 Athens server time
    prior_bar = stamp(2026, 11, 6, 20)
    portfolio = PortfolioStateSnapshot(
        current_time_s=prior_bar, balance=10000, equity=10000, peak_equity=10000,
        day_start_equity=10000, week_start_equity=10000,
        last_bar_time_s=prior_bar, timeframe_seconds=3600, execution_time_s=now_utc,
    )
    examiner = PreTradeExaminer(QuantRLConfig().execution, server_timezone='Europe/Athens')
    verdict = examiner.inspect(
        proposal, portfolio, default_symbol_spec('EURUSD'),
        mid_price=1.1, spread_pips=1., volatility_price=.002,
    )
    assert verdict.checks_passed['C5'] is False
