from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

BASE = Path(__file__).resolve().parent.parent  
PROFILES_DIR = BASE / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

def _sanitize_name(name: str) -> str:
    """Elimină caractere problematice din numele profilului pentru a-l folosi ca nume de fișier."""
    name = (name or "").strip()
    if not name:
        name = "profile"
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    return name

def _profile_path(name: str) -> Path:
    """Returnează calea completă la fișierul JSON al profilului."""
    return PROFILES_DIR / f"{_sanitize_name(name)}.json"

def save_profile(name: str, settings: Dict) -> Path:
    """
    Salvează setările în profiles/<name>.json.
    Returnează calea fișierului salvat.
    """
    path = _profile_path(name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"Could not save profile '{name}': {e}")
    return path

def load_profile(name: str) -> Optional[Dict]:
    """
    Încarcă setările din profiles/<name>.json.
    Returnează dict-ul sau None dacă nu există / e corupt.
    """
    path = _profile_path(name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def list_profiles() -> List[str]:
    """
    Listează numele profilurilor disponibile (fără extensie).
    """
    out = []
    for p in PROFILES_DIR.glob("*.json"):
        out.append(p.stem)
    out.sort(key=str.lower)
    return out


DEFAULT_POINTER = PROFILES_DIR / "_default.txt"

def set_default_profile(name: str) -> None:
    """
    Marchează un profil ca 'default' (salvat în profiles/_default.txt).
    """
    name = _sanitize_name(name)
    if not _profile_path(name).exists():
        raise FileNotFoundError(f"Profile '{name}' not found.")
    DEFAULT_POINTER.write_text(name, encoding="utf-8")

def get_default_profile_name() -> Optional[str]:
    """
    Returnează numele profilului default dacă există.
    """
    if DEFAULT_POINTER.exists():
        try:
            val = DEFAULT_POINTER.read_text(encoding="utf-8").strip()
            return val or None
        except Exception:
            return None
    return None

def load_default_if_exists() -> Optional[Dict]:
    """
    Încarcă automat profilul default, dacă a fost setat.
    """
    name = get_default_profile_name()
    if not name:
        return None
    return load_profile(name)