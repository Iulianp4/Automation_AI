import hashlib
import json
from pathlib import Path
from typing import Any, Dict

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

def _hash_key(prompt: str, params: Dict[str, Any]) -> str:
    m = hashlib.sha256()
    m.update(prompt.encode("utf-8"))
    m.update(json.dumps(params, sort_keys=True).encode("utf-8"))
    return m.hexdigest()

def get(prompt: str, params: Dict[str, Any]) -> Any | None:
    key = _hash_key(prompt, params)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def set(prompt: str, params: Dict[str, Any], payload: Any):
    key = _hash_key(prompt, params)
    path = CACHE_DIR / f"{key}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[cache] Failed to write {path}: {e}")