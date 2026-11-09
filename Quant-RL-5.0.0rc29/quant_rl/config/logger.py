"""Small local audit logger: JSONL, console diagnostics, rotation and secret redaction."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import atexit
import json
import os
import sys
import threading


class LogLevel(Enum):
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class AuditLogger:
    _COLORS = {LogLevel.INFO: "\033[32m", LogLevel.SUCCESS: "\033[34m",
               LogLevel.WARNING: "\033[33m", LogLevel.ERROR: "\033[31m",
               LogLevel.CRITICAL: "\033[35m"}
    _MARKERS = {LogLevel.INFO: "[inf]", LogLevel.SUCCESS: "[ok ]",
                LogLevel.WARNING: "[wrn]", LogLevel.ERROR: "[err]",
                LogLevel.CRITICAL: "[ftl]"}

    def __init__(self):
        self._lock = threading.RLock()
        self._directory: Path | None = None
        self._file = None
        self._date = None
        self._seq = 0
        self._part = 0
        self.max_file_bytes = 16 * 1024 * 1024
        self.retention_days = 30
        self._secrets: tuple[str, ...] = ()
        self._console_color = bool(getattr(sys.stdout, 'isatty', lambda: False)()) and os.getenv('NO_COLOR') is None

    def configure(self, root: Path | str, *, secrets=()):
        with self._lock:
            folder = Path(root).resolve() / 'logs'
            if folder != self._directory:
                self.shutdown()
                self._directory = folder
            self._secrets = tuple(sorted({str(s) for s in secrets if s}, key=len, reverse=True))
            folder.mkdir(parents=True, exist_ok=True)
            oldest = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            for existing in folder.glob('quant_rl_*.log'):
                try:
                    if datetime.fromtimestamp(existing.stat().st_mtime,timezone.utc)<oldest:
                        existing.unlink(missing_ok=True)
                except OSError:
                    pass

    def shutdown(self):
        with self._lock:
            if self._file:
                self._file.close()
            self._file = None
            self._date = None
            self._part = 0

    def _scrub(self, message: str) -> str:
        for secret in self._secrets:
            message = message.replace(secret, '[REDACTED]')
        return message

    def _open_segment(self, day: str) -> None:
        """Resume the next writable segment without growing an old full part.

        After a process restart, the prior `_*_001.log` may already be full.
        A fresh logger must not blindly append to it forever.
        """
        assert self._directory is not None
        pid=os.getpid()
        part=0
        while True:
            tail=f'_{part:03d}' if part else ''
            path=self._directory/f'quant_rl_{day}_{pid}{tail}.log'
            if not path.exists() or path.stat().st_size < self.max_file_bytes:
                self._file=path.open('a',encoding='utf-8')
                self._part=part
                return
            part+=1

    def _write(self, level: LogLevel, component: str, message: str):
        with self._lock:
            if self._directory is None:
                # IDE startup failures should still be visible before configuration.
                self.configure(Path(__file__).resolve().parents[2])
            self._seq += 1
            now = datetime.now().astimezone()
            day = now.strftime('%Y-%m-%d')
            if self._file is None or self._date != day:
                if self._file:
                    self._file.close()
                self._part=0
                self._date=day
                self._open_segment(day)
            if self._file.tell() >= self.max_file_bytes:
                self._file.close()
                self._open_segment(day)
            msg = self._scrub(str(message))
            component = self._scrub(str(component))
            stamp = now.isoformat(timespec='seconds')
            human = f'{stamp} sequence={self._seq:06d} {self._MARKERS[level]} {component:<12} {msg}'
            if self._console_color:
                human = self._COLORS[level] + human + '\033[0m'
            print(human, flush=True)
            record = {'timestamp': stamp,
                      'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                      'sequence': self._seq, 'pid': os.getpid(),
                      'level': level.name, 'component': component, 'message': msg}
            self._file.write(json.dumps(record, ensure_ascii=False) + '\n')
            if level.value >= LogLevel.WARNING.value or self._seq % 32 == 0:
                self._file.flush()

    def info(self, component, message): self._write(LogLevel.INFO, component, message)
    def success(self, component, message): self._write(LogLevel.SUCCESS, component, message)
    def warning(self, component, message): self._write(LogLevel.WARNING, component, message)
    def error(self, component, message): self._write(LogLevel.ERROR, component, message)
    def critical(self, component, message): self._write(LogLevel.CRITICAL, component, message)


log = AuditLogger()
atexit.register(log.shutdown)
