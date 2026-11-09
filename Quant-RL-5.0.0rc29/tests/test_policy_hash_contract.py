from quant_rl.config.schema import QuantRLConfig


def test_policy_hash_covers_all_behavior_and_qualification_contracts():
    base = QuantRLConfig()
    changes = [
        {'research': {'lookback_years': 5}},
        {'ppo': {'penalty_reference_lots': 0.02}},
        {'qualification': {'cost_stress_multiplier': 1.75}},
        {'qualification': {'max_drawdown_pct': 15.0}},
        {'live': {'max_entry_delay_seconds': 120}},
        {'live': {'economic_blackout_active': True}},
        {'live': {'order_deviation_points': 10}},
        {'live': {'poll_seconds': 5.0}},
        {'execution': {'daily_drawdown_limit_pct': 2.0}},
        {'data': {'observation_timeframes': ['H1', 'H4', 'D1']}},
        {'broker_connection': {'mt5_server_timezone': 'Europe/Athens'}},
    ]
    for change in changes:
        assert QuantRLConfig.model_validate(change).policy_hash != base.policy_hash, change


def test_policy_hash_ignores_search_budget_and_deployment_identity():
    base = QuantRLConfig()
    changed = QuantRLConfig.model_validate({
        'optimization': {'trials': 250, 'tournament_top_k': 5},
        'ppo': {'episodes': 500, 'ppo_epochs': 400},
        'live': {'symbol': 'GBPUSD', 'magic': 12345},
        'broker_connection': {'mt5_login': 987654},
    })
    assert changed.policy_hash == base.policy_hash


def test_router_rejects_model_after_policy_contract_changes(tmp_path):
    from datetime import datetime, timezone
    import time
    import torch
    from quant_rl.research.evolution import ModelPopulation
    from quant_rl.routing import _existing_model

    cfg = QuantRLConfig.model_validate({'project_root': str(tmp_path)})
    pop = ModelPopulation(tmp_path)
    snapshot = {'weights': {'x': torch.tensor([1.])},
                'mean': torch.zeros(1), 'std': torch.ones(1)}
    end = datetime.fromtimestamp(time.time() - 60, timezone.utc).isoformat()
    record = {
        'arch': 'test', 'origin': 'MT5_TERMINAL', 'account_currency': 'USD',
        'current_score': 0.1, 'policy_hash': cfg.policy_hash,
        'broker_server': 'Demo', 'mt5_server_timezone': None,
        'period': {'test_end_utc': end},
    }
    pop.publish({'EURUSD': (snapshot, record)}, {}, 'EURUSD', {})
    assert _existing_model(pop, 'EURUSD', 'USD', None, cfg=cfg,
                           broker_server='Demo')
    changed = cfg.model_copy(update={
        'live': cfg.live.model_copy(update={'economic_blackout_active': True})
    })
    assert not _existing_model(pop, 'EURUSD', 'USD', None, cfg=changed,
                                broker_server='Demo')
