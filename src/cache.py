# src/cache.py
from __future__ import annotations
from pathlib import Path
import json
import hashlib
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "gpt_cache.json"

def _load() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save(obj: dict) -> None:
    CACHE_FILE.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _key_from_prompt(prompt: str, params: dict) -> str:
    blob = json.dumps({"p": prompt, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def get(prompt: str, params: dict) -> Any | None:
    data = _load()
    key = _key_from_prompt(prompt, params)
    return data.get(key)

def set(prompt: str, params: dict, value: Any) -> None:
    data = _load()
    key = _key_from_prompt(prompt, params)
    data[key] = value
    _save(data)
