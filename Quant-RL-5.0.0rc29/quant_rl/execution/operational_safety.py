"""Operator-controlled fail-closed trading halt and live MT5 health evidence.

A kill switch can be engaged by QUANT_RL_KILL_SWITCH=1 or a global
`live_state/kill_switch.json` containing {"engaged": true}. A missing switch
means disengaged; corrupted/unreadable state is treated as engaged.
"""
from __future__ import annotations
import json
import os
from pathlib import Path


def kill_switch_engaged(root: Path) -> bool:
    env = os.getenv("QUANT_RL_KILL_SWITCH", "0").strip().lower()
    if env not in ("0", "1", "true", "false", "on", "off"):
        return True
    if env in ("1", "true", "on"):
        return True
    path = Path(root) / "live_state" / "kill_switch.json"
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text("utf-8"))
        if not isinstance(state, dict) or type(state.get("engaged")) is not bool:
            return True
        return state["engaged"]
    except (OSError, ValueError, UnicodeError):
        return True


def broker_heartbeat_healthy(broker, *, login: int, server: str) -> bool:
    """Independently poll MT5 terminal/account via its authoritative account()."""
    try:
        a = broker.account()  # MT5Broker.account checks terminal and trade permissions.
        return int(a.login) == login and str(getattr(a, "server", "")) == server
    except (ConnectionError, PermissionError, ValueError, AttributeError, TypeError):
        return False


def operator_blackout_active(root: Path, configured: bool = False) -> bool:
    """Operator-managed economic-event veto, re-read before each order.

    MT5 provides no authoritative economic-event calendar to this code. An
    operator can set `live_state/economic_blackout.json` with
    {"active": true} while the process is running. Malformed/unreadable state
    blocks orders rather than allowing a trade through ambiguity.
    """
    if configured:
        return True
    env = os.getenv('QUANT_RL_ECONOMIC_BLACKOUT', '0').strip().lower()
    if env not in ('0', '1', 'true', 'false', 'on', 'off'):
        return True
    if env in ('1','true','on'):
        return True
    path = Path(root)/'live_state'/'economic_blackout.json'
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text('utf-8'))
        return (state['active'] if isinstance(state,dict) and
                type(state.get('active')) is bool else True)
    except (OSError, ValueError, KeyError, UnicodeError):
        return True
