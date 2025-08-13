# src/preprocess.py
import pandas as pd
from pathlib import Path
from .config import DATA_FILES

BASE = Path(__file__).resolve().parent.parent

def _drop_empty_rows(df: pd.DataFrame, cols):
    """Păstrează doar rândurile unde cel puțin una din coloanele 'cols' are conținut non-gol."""
    if df.empty:
        return df
    # normalizare la string + strip
    tmp = df[cols].astype(str).apply(lambda s: s.str.strip())
    mask = tmp.apply(lambda r: any(v for v in r), axis=1)
    return df[mask].copy()

def read_requirements() -> pd.DataFrame:
    try:
        df = pd.read_excel(BASE / DATA_FILES["requirements"], sheet_name="stories").fillna("")
    except Exception:
        return pd.DataFrame()
    rename = {
        "requirements_id": "requirement_id",
        "requirement_id": "requirement_id",
        "requirements_name": "requirement_name",
        "requirements_description": "requirement_text",
    }
    df = df.rename(columns=rename)
    keep = [c for c in ["requirement_id","requirement_name","requirement_text"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep]
    # elimini rândurile cu toate câmpurile goale
    df = _drop_empty_rows(df, keep)
    return df

def read_acceptance() -> pd.DataFrame:
    try:
        df = pd.read_excel(BASE / DATA_FILES["acceptance_criteria"], sheet_name="criteria").fillna("")
    except Exception:
        return pd.DataFrame()
    rename = {
        "acceptance_criteria_story_id": "requirement_id",
        "ac_story_id": "requirement_id",
        "acceptance_criteria": "ac_text",
    }
    df = df.rename(columns=rename)
    keep = [c for c in ["requirement_id","ac_text"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    df = df[keep]
    df = _drop_empty_rows(df, keep)  # <- filtrează rândurile complet goale
    return df

def read_use_cases() -> pd.DataFrame:
    from pathlib import Path
    import pandas as pd
    from .config import DATA_FILES

    BASE = Path(__file__).resolve().parent.parent

    try:
        df = pd.read_excel(BASE / DATA_FILES["use_cases"], sheet_name="manual").fillna("")
    except Exception:
        return pd.DataFrame(columns=["requirement_id", "uc_text"])

    # mapăm antetele
    rename = {
        "use_cases_story_id": "requirement_id",
        "use_cases_title": "uc_title",
        "use_cases_description": "uc_desc",
        "use_cases_preconditions": "uc_pre",
        "use_cases_main_flow": "uc_main",
        "use_cases_alternative_flows": "uc_alt",
        "use_cases_exception_flows": "uc_exc",
        "use_cases_business_rules": "uc_rules",
    }
    df = df.rename(columns=rename)

    # coloane candidate pentru compunerea textului UC
    parts = [c for c in ["uc_title","uc_desc","uc_pre","uc_main","uc_alt","uc_exc","uc_rules"] if c in df.columns]

    # dacă nu avem niciuna din coloane -> întoarcem DF gol, dar cu schema corectă
    if not parts:
        return pd.DataFrame(columns=["requirement_id", "uc_text"])

    # funcție robustă pe FIECARE rând -> returnează un șir
    def build_uc_text(row):
        vals = []
        for c in parts:
            v = str(row.get(c, "")).strip()
            if v:
                vals.append(v)
        return "\n".join(vals)

    # construim UC pe rânduri; rezultatul e garantat Series
    uc_text_series = df.apply(build_uc_text, axis=1)

    # selectăm câmpurile minime
    out = pd.DataFrame({
        "requirement_id": df.get("requirement_id", ""),
        "uc_text": uc_text_series
    })

    # eliminăm rândurile complet goale
    def _nonempty(x): return bool(str(x).strip())
    out = out[out.apply(lambda r: _nonempty(r["requirement_id"]) or _nonempty(r["uc_text"]), axis=1)]

    return out.reset_index(drop=True)

def load_all():
    req = read_requirements()
    ac  = read_acceptance()
    uc  = read_use_cases()

    ac_grp = ac.groupby("requirement_id")["ac_text"].apply(list).rename("ac_list") if not ac.empty else pd.Series(dtype=object, name="ac_list")
    uc_grp = uc.groupby("requirement_id")["uc_text"].apply(list).rename("uc_list") if not uc.empty else pd.Series(dtype=object, name="uc_list")

    merged = req.copy()
    if not ac_grp.empty:
        merged = merged.merge(ac_grp, on="requirement_id", how="left")
    else:
        merged["ac_list"] = [[] for _ in range(len(merged))]
    if not uc_grp.empty:
        merged = merged.merge(uc_grp, on="requirement_id", how="left")
    else:
        merged["uc_list"] = [[] for _ in range(len(merged))]

    merged["ac_list"] = merged["ac_list"].apply(lambda v: v if isinstance(v, list) else [])
    merged["uc_list"] = merged["uc_list"].apply(lambda v: v if isinstance(v, list) else [])
    return merged, ac, uc
