import pandas as pd
from pathlib import Path
from .config import DATA_FILES

BASE = Path(__file__).resolve().parent.parent

# ===== Helpers =====
def _nonempty(x) -> bool:
    return bool(str(x).strip())

def _mk_ids(prefix: str, n: int):
    return [f"{prefix}-{i+1}" for i in range(n)]

def _norm_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.strip()
    )

def _coalesce_duplicate_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Dacă există mai multe coloane cu același nume, păstrează una singură
    cu prima valoare non-goală pe rând, apoi șterge duplicatele."""
    same = [c for c in df.columns if c == col]
    if len(same) <= 1:
        return df
    # prima valoare non-goală pe fiecare rând
    def _first_nonempty(row):
        for v in row:
            if str(v).strip():
                return v
        return ""
    merged = df[same].apply(_first_nonempty, axis=1)
    df = df.drop(columns=same)
    df[col] = merged
    return df

# --- NEW: friendly validator used by main.py ---
def validate_columns(df: pd.DataFrame, required: list[str], file_label: str) -> None:
    """Print a friendly warning if required columns are missing."""
    if df is None or df.empty:
        return
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"⚠️ {file_label}: missing columns -> {', '.join(missing)}")

# ===== Readers =====
def read_requirements() -> pd.DataFrame:
    try:
        df = pd.read_excel(BASE / DATA_FILES["requirements"]).fillna("")
    except Exception:
        return pd.DataFrame()

    rename = {
        "requirements_id": "requirement_id",
        "requirement_id": "requirement_id",
        "requirements_name": "requirement_name",
        "requirements_description": "requirement_text",
        "requirements_rationale": "requirement_rationale",
        "requirements_platform": "requirement_platform",
        "requirement_details": "requirement_details",
    }
    df = df.rename(columns=rename)

    df = _coalesce_duplicate_column(df, "requirement_id")

    for col in ["requirement_text", "requirement_name", "requirement_details"]:
        if col not in df.columns:
            df[col] = ""

    keep = [c for c in [
        "requirement_id","requirement_name","requirement_text",
        "requirement_rationale","requirement_platform","requirement_details"
    ] if c in df.columns]
    df = df[keep].copy()

    if "requirement_id" in df.columns:
        df["requirement_id"] = _norm_id(df["requirement_id"])
        mask_blank = ~df["requirement_id"].apply(_nonempty)
        if mask_blank.any():
            df.loc[mask_blank, "requirement_id"] = _mk_ids("REQ-LONE", int(mask_blank.sum()))
    else:
        df["requirement_id"] = _mk_ids("REQ-LONE", len(df))

    keep_mask = df["requirement_name"].apply(_nonempty) | df["requirement_text"].apply(_nonempty)
    return df[keep_mask].reset_index(drop=True)

def read_acceptance() -> pd.DataFrame:
    try:
        df = pd.read_excel(BASE / DATA_FILES["acceptance_criteria"]).fillna("")
    except Exception:
        return pd.DataFrame()

    rename = {
        "acceptance_criteria_story_id": "requirement_id",
        "ac_story_id": "requirement_id",
        "acceptance_criteria": "ac_text",
        "acceptance_criteria_notes": "ac_notes",
        "acceptance_criteria_comments": "ac_comments",
        "acceptance_criteria_details": "ac_details",
    }
    df = df.rename(columns=rename)

    df = _coalesce_duplicate_column(df, "requirement_id")

    if "ac_text" not in df.columns:
        return pd.DataFrame(columns=["requirement_id","ac_text","ac_details"])

    if "ac_details" not in df.columns:
        df["ac_details"] = ""

    if "requirement_id" in df.columns:
        df["requirement_id"] = _norm_id(df["requirement_id"])
        mask_blank = ~df["requirement_id"].apply(_nonempty)
        if mask_blank.any():
            df.loc[mask_blank, "requirement_id"] = _mk_ids("AC-LONE", int(mask_blank.sum()))
    else:
        df["requirement_id"] = _mk_ids("AC-LONE", len(df))

    df = df[df["ac_text"].apply(_nonempty)][["requirement_id","ac_text","ac_details"]]
    return df.reset_index(drop=True)

def read_use_cases() -> pd.DataFrame:
    try:
        df = pd.read_excel(BASE / DATA_FILES["use_cases"]).fillna("")
    except Exception:
        return pd.DataFrame(columns=["requirement_id","uc_text","uc_details"])

    rename = {
        "use_cases_story_id": "requirement_id",
        "use_cases_title": "uc_title",
        "use_cases_document_information": "uc_doc_info",
        "use_cases_revision_history": "uc_rev",
        "use_cases_description": "uc_desc",
        "use_cases_preconditions": "uc_pre",
        "use_cases_main_flow": "uc_main",
        "use_cases_alternative_flows": "uc_alt",
        "use_cases_exception_flows": "uc_exc",
        "use_cases_business_rules": "uc_rules",
        "use_cases_details": "uc_details",
    }
    df = df.rename(columns=rename)

    df = _coalesce_duplicate_column(df, "requirement_id")

    parts = [c for c in ["uc_title","uc_desc","uc_pre","uc_main","uc_alt","uc_exc","uc_rules"] if c in df.columns]
    if "uc_details" not in df.columns:
        df["uc_details"] = ""

    if not parts:
        if "uc_details" in df.columns and "requirement_id" in df.columns:
            out = pd.DataFrame({
                "requirement_id": df["requirement_id"],
                "uc_text": df["uc_details"].astype(str).fillna(""),
                "uc_details": df["uc_details"].astype(str).fillna(""),
            })
        else:
            return pd.DataFrame(columns=["requirement_id","uc_text","uc_details"])
    else:
        def build_uc_text(row):
            vals = [str(row.get(c, "")).strip() for c in parts]
            vals = [v for v in vals if v]
            return "\n".join(vals)

        out = pd.DataFrame({
            "requirement_id": df.get("requirement_id", ""),
            "uc_text": df.apply(build_uc_text, axis=1),
            "uc_details": df.get("uc_details", "")
        })

    out["requirement_id"] = _norm_id(out["requirement_id"])
    mask_blank = ~out["requirement_id"].apply(_nonempty)
    if mask_blank.any():
        out.loc[mask_blank, "requirement_id"] = _mk_ids("UC-LONE", int(mask_blank.sum()))

    out["uc_text"] = out["uc_text"].astype(str)
    out["uc_details"] = out["uc_details"].astype(str)
    keep_mask = out["uc_text"].apply(_nonempty) | out["uc_details"].apply(_nonempty)
    out = out[keep_mask]
    return out.reset_index(drop=True)

def load_all():
    req_df = read_requirements()
    ac_df  = read_acceptance()
    uc_df  = read_use_cases()

    for d in (req_df, ac_df, uc_df):
        if not d.empty and "requirement_id" in d.columns:
            d["requirement_id"] = _norm_id(d["requirement_id"])

    # group & merge
    if not ac_df.empty:
        ac_grouped = ac_df.groupby("requirement_id")["ac_text"].apply(list).reset_index()
    else:
        ac_grouped = pd.DataFrame(columns=["requirement_id","ac_text"])

    if not uc_df.empty:
        uc_grouped = uc_df.groupby("requirement_id")["uc_text"].apply(list).reset_index()
    else:
        uc_grouped = pd.DataFrame(columns=["requirement_id","uc_text"])

    merged_df = req_df.merge(ac_grouped, on="requirement_id", how="left") \
                      .merge(uc_grouped, on="requirement_id", how="left")

    merged_df["ac_list"] = merged_df["ac_text"].apply(lambda x: x if isinstance(x, list) else [])
    merged_df["uc_list"] = merged_df["uc_text"].apply(lambda x: x if isinstance(x, list) else [])

    return merged_df, ac_df, uc_df

def read_manual_cases() -> pd.DataFrame:
    """
    Reads manually authored test cases from data/manual_cases.xlsx and normalizes columns to our internal schema:
      requirement_id, requirement_name, tc_id, title, preconditions, steps, data, expected, category, gherkin, type
    Accepts both friendly (Excel output) and internal names.
    """
    path = BASE / DATA_FILES["manual"]
    try:
        df = pd.read_excel(path).fillna("")
    except Exception:
        return pd.DataFrame(columns=[
            "requirement_id","requirement_name","tc_id","title","preconditions","steps",
            "data","expected","category","gherkin","type"
        ])

    rename_map = {
        # friendly -> internal
        "Requirement ID": "requirement_id",
        "Requirement Name": "requirement_name",
        "Test Case ID": "tc_id",
        "Title": "title",
        "Preconditions": "preconditions",
        "Steps": "steps",
        "Test Data": "data",
        "Expected Result": "expected",
        "Category": "category",
        "Gherkin": "gherkin",
        "Source": "type",
        # internal names already
        "requirement_id": "requirement_id",
        "requirement_name": "requirement_name",
        "tc_id": "tc_id",
        "title": "title",
        "preconditions": "preconditions",
        "steps": "steps",
        "data": "data",
        "expected": "expected",
        "category": "category",
        "gherkin": "gherkin",
        "type": "type",
    }
    df = df.rename(columns=rename_map)

    # ensure all expected cols exist
    for col in ["requirement_id","requirement_name","tc_id","title","preconditions","steps","data","expected","category","gherkin","type"]:
        if col not in df.columns:
            df[col] = ""

    # default type label if not provided
    df["type"] = df["type"].apply(lambda x: x if str(x).strip() else "Manual (baseline)")

    # normalize whitespace
    def _norm(s: str) -> str:
        return str(s or "").replace("\u00A0", " ").strip()

    for c in ["requirement_id","requirement_name","tc_id","title","preconditions","steps","data","expected","category","gherkin","type"]:
        df[c] = df[c].astype(str).map(_norm)

    # filter out completely empty rows (no title and no expected)
    mask_keep = df["title"].astype(str).str.strip().ne("") | df["expected"].astype(str).str.strip().ne("")
    df = df[mask_keep].reset_index(drop=True)

    return df

   # --- Validation helpers (UI-facing) ---
def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> dict:
    """
    Return a dict with validation info:
      { "ok": bool, "missing": [..], "present": [..], "label": str, "rows": int }
    Doesn't raise; the UI decides how to display.
    """
    present = [c for c in required if c in df.columns]
    missing = [c for c in required if c not in df.columns]
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "present": present,
        "label": label,
        "rows": int(len(df) if df is not None else 0),
    }
 