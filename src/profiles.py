# src/profiles.py
import json
from pathlib import Path
from typing import Dict, Any, List

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"
PROFILE_DIR.mkdir(exist_ok=True)

def _file(name: str) -> Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in ("-","_"))
    return PROFILE_DIR / f"{safe}.json"

def save_profile(name: str, settings: Dict[str, Any]):
    path = _file(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

def load_profile(name: str) -> Dict[str, Any] | None:
    path = _file(name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def list_profiles() -> List[str]:
    return [p.stem for p in PROFILE_DIR.glob("*.json")]
