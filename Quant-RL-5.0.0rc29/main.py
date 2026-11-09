"""IDE-only startup: MT5-driven automatic RESEARCH/LIVE routing."""
from pathlib import Path
import os
from quant_rl import __version__
from quant_rl.config.loader import ConfigLoader
from quant_rl.routing import run_automatic_session
from quant_rl.config.logger import log


def main():
    cfg=ConfigLoader.load(Path(__file__).resolve().parent/'config.yaml')
    log.configure(cfg.project_root, secrets=[cfg.broker_connection.mt5_password, os.environ.get('QUANT_MT5_PASSWORD', '')])
    log.info('STARTUP', f'Quant-RL {__version__} | MT5-only | symbols={list(cfg.research.symbols)}')
    try:
        return run_automatic_session(cfg)
    except Exception as exc:
        log.critical('STARTUP', f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    main()
