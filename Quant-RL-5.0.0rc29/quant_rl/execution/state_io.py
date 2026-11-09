"""Crash-resistant JSON state updates for order and broker-history journals."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid


def store_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('x',encoding='utf-8') as stream:
            json.dump(payload,stream,sort_keys=True,allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)
