from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import re
import unicodedata
import difflib
import pandas as pd


# -----------------------------
# Normalization & similarity
# -----------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # strip accents
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _tokens(s: str) -> set:
    return set(normalize_text(s).split())

def jaccard(a: str, b: str) -> float:
    A, B = _tokens(a), _tokens(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def _concat_title_expected(row: pd.Series) -> str:
    return f"{row.get('title','')} || {row.get('expected','')}"

def _concat_title_steps_expected(row: pd.Series, max_len_steps: int = 2000) -> str:
    steps = str(row.get("steps",""))
    if len(steps) > max_len_steps:
        steps = steps[:max_len_steps]
    return f"{row.get('title','')} || {steps} || {row.get('expected','')}"

def similarity(row_a: pd.Series, row_b: pd.Series, strategy: str = "title_expected") -> float:
    if strategy == "title_steps_expected":
        A = _concat_title_steps_expected(row_a)
        B = _concat_title_steps_expected(row_b)
    else:
        A = _concat_title_expected(row_a)
        B = _concat_title_expected(row_b)
    # blend difflib + jaccard
    dl = difflib.SequenceMatcher(None, normalize_text(A), normalize_text(B)).ratio()
    jc = jaccard(A, B)
    return (dl + jc) / 2.0


# -----------------------------
# Greedy best-match comparison
# -----------------------------
def best_match_greedy(
    df_manual: pd.DataFrame,
    df_ai: pd.DataFrame,
    strategy: str,
    threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-pass greedy: for each manual row, pick the best AI row above threshold.
    No AI test is matched twice.
    Returns: (matches_df, manual_only_df, ai_only_df)
    """
    # ensure required cols exist
    need = ["tc_id","title","expected","steps","requirement_id","category","gherkin","type"]
    for c in need:
        if c not in df_manual.columns: df_manual[c] = ""
        if c not in df_ai.columns: df_ai[c] = ""

    used_ai = set()
    matches = []

    for idx_m, m in df_manual.iterrows():
        best_sim = -1.0
        best_idx = None
        for idx_a, a in df_ai.iterrows():
            if a.get("tc_id","") in used_ai:
                continue
            sim = similarity(m, a, strategy=strategy)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx_a
        if best_idx is not None and best_sim >= threshold:
            a = df_ai.loc[best_idx]
            used_ai.add(a.get("tc_id",""))
            matches.append({
                "manual_tc_id": m.get("tc_id",""),
                "manual_title": m.get("title",""),
                "ai_tc_id": a.get("tc_id",""),
                "ai_title": a.get("title",""),
                "requirement_id": m.get("requirement_id","") or a.get("requirement_id",""),
                "similarity": round(float(best_sim), 4),
                "fields_used": strategy,
                "category_ai": a.get("category",""),
                "category_manual": m.get("category",""),
                "has_gherkin_ai": "Yes" if str(a.get("gherkin","")).strip() else "No",
                "has_gherkin_manual": "Yes" if str(m.get("gherkin","")).strip() else "No",
            })
        else:
            # no match found for this manual case
            pass

    matches_df = pd.DataFrame(matches)

    matched_ai_ids = set(matches_df["ai_tc_id"].tolist()) if not matches_df.empty else set()
    matched_manual_ids = set(matches_df["manual_tc_id"].tolist()) if not matches_df.empty else set()

    ai_only_df = df_ai[~df_ai["tc_id"].isin(matched_ai_ids)].copy()
    manual_only_df = df_manual[~df_manual["tc_id"].isin(matched_manual_ids)].copy()

    # make ai_only/manual_only lean for the report
    ai_only_df = ai_only_df[["tc_id","title","requirement_id","category","gherkin","type"]].rename(columns={
        "tc_id":"ai_tc_id","gherkin":"has_gherkin"
    })
    ai_only_df["has_gherkin"] = ai_only_df["has_gherkin"].apply(lambda x: "Yes" if str(x).strip() else "No")

    manual_only_df = manual_only_df[["tc_id","title","requirement_id","category","gherkin","type"]].rename(columns={
        "tc_id":"manual_tc_id","gherkin":"has_gherkin"
    })
    manual_only_df["has_gherkin"] = manual_only_df["has_gherkin"].apply(lambda x: "Yes" if str(x).strip() else "No")

    return matches_df, manual_only_df, ai_only_df


# -----------------------------
# Linting / Quality scoring
# -----------------------------
_IMP_VERBS = re.compile(
    r"^\s*(?:-|\d+\.)?\s*(click|enter|type|select|choose|press|open|navigate|verify|assert|check|submit|login|logout|create|delete|update|save|upload|download)\b",
    re.I
)
_NUMBERED = re.compile(r"^\s*(?:-|\d+\.)\s*")
_VAGUE = re.compile(r"\b(?:etc|maybe)\b|\.{3,}", re.I)
_EXPECTED_VERB = re.compile(
    r"\b(displayed|shown|redirect|navigat|status\s*(?:200|201|204|400|401|403|404|500)|error|success|created|updated|saved|appears|visible|enabled|disabled|returns?)\b",
    re.I
)

def _split_lines(s: str) -> list[str]:
    s = str(s or "")
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return lines

def _rate_steps_imperative(steps: str) -> float:
    lines = _split_lines(steps)
    if not lines: return 0.0
    ok = sum(1 for ln in lines if _IMP_VERBS.search(ln))
    return ok / len(lines)

def _rate_steps_numbered(steps: str) -> float:
    lines = _split_lines(steps)
    if not lines: return 0.0
    ok = sum(1 for ln in lines if _NUMBERED.search(ln))
    return ok / len(lines)

def _rate_expected_verifiable(expected: str) -> float:
    txt = str(expected or "")
    return 1.0 if _EXPECTED_VERB.search(txt) else 0.0

def _rate_gherkin_complete(gherkin: str) -> float:
    g = normalize_text(gherkin)
    has_given = "given" in g
    has_when = "when" in g
    has_then = "then" in g
    return 1.0 if (has_given and has_when and has_then) else 0.0

def _vague_penalty(steps: str, expected: str) -> float:
    s = f"{steps}\n{expected}"
    return 1.0 if _VAGUE.search(s) else 0.0  # 1 if vague present

def lint_quality(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return dict(
            steps_imperative=0.0,
            steps_numbered=0.0,
            expected_verifiable=0.0,
            gherkin_complete=0.0,
            vague_penalty=0.0,
            quality=0.0
        )
    steps_imp = df["steps"].apply(_rate_steps_imperative).mean()
    steps_num = df["steps"].apply(_rate_steps_numbered).mean()
    exp_ver  = df["expected"].apply(_rate_expected_verifiable).mean()
    gherkin  = df.get("gherkin", pd.Series([""]*len(df))).apply(_rate_gherkin_complete).mean()
    vague    = df.apply(lambda r: _vague_penalty(r.get("steps",""), r.get("expected","")), axis=1).mean()

    quality = (
        0.35*steps_imp +
        0.25*exp_ver +
        0.20*gherkin +
        0.10*steps_num +
        0.10*(1 - min(1.0, vague))
    )
    return dict(
        steps_imperative=round(float(steps_imp), 3),
        steps_numbered=round(float(steps_num), 3),
        expected_verifiable=round(float(exp_ver), 3),
        gherkin_complete=round(float(gherkin), 3),
        vague_penalty=round(float(vague), 3),
        quality=round(float(quality), 3)
    )


# -----------------------------
# Aggregations for report
# -----------------------------
def build_category_dists(df_ai: pd.DataFrame, df_manual: pd.DataFrame) -> pd.DataFrame:
    def dist(df, label):
        if df.empty:
            return pd.DataFrame({"Category": [], label: []})
        tmp = df.copy()
        if "category" not in tmp.columns:
            tmp["category"] = ""
        d = tmp.groupby("category").size().reset_index(name=label)
        d = d.rename(columns={"category": "Category"})  # <-- keep friendly name
        return d

    a = dist(df_ai, "AI")
    m = dist(df_manual, "Manual")
    out = pd.merge(a, m, on="Category", how="outer").fillna(0)

    # percentages & delta
    total_ai = out["AI"].sum() if "AI" in out.columns else 0
    total_man = out["Manual"].sum() if "Manual" in out.columns else 0
    out["AI_%"] = (out["AI"] / total_ai * 100.0).round(2) if total_ai else 0.0
    out["Manual_%"] = (out["Manual"] / total_man * 100.0).round(2) if total_man else 0.0
    out["Delta_%"] = (out["AI_%"] - out["Manual_%"]).round(2)

    return out.sort_values("Category", na_position="last")

def build_per_requirement_density(df_ai: pd.DataFrame, df_manual: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    ai_cnt = df_ai.groupby("requirement_id").size().reset_index(name="AI_count") if not df_ai.empty else pd.DataFrame(columns=["requirement_id","AI_count"])
    mn_cnt = df_manual.groupby("requirement_id").size().reset_index(name="Manual_count") if not df_manual.empty else pd.DataFrame(columns=["requirement_id","Manual_count"])
    out = pd.merge(ai_cnt, mn_cnt, on="requirement_id", how="outer").fillna(0)

    if not matches_df.empty:
        m_per_req = matches_df.groupby("requirement_id").size().reset_index(name="Matches")
        out = pd.merge(out, m_per_req, on="requirement_id", how="left").fillna(0)
    else:
        out["Matches"] = 0

    out = out.rename(columns={"requirement_id":"Requirement ID"})
    return out.sort_values("Requirement ID")

def build_trace_matrix(matches_df: pd.DataFrame, df_manual: pd.DataFrame, df_ai: pd.DataFrame) -> pd.DataFrame:
    """
    For each requirement_id: concat manual tc_ids, ai tc_ids, and matched pairs manual->ai
    """
    reqs = set(df_manual.get("requirement_id","").tolist()) | set(df_ai.get("requirement_id","").tolist())
    rows = []
    for rid in sorted(reqs):
        if rid in ("", None): continue
        manual_ids = df_manual[df_manual["requirement_id"]==rid]["tc_id"].tolist() if "tc_id" in df_manual.columns else []
        ai_ids = df_ai[df_ai["requirement_id"]==rid]["tc_id"].tolist() if "tc_id" in df_ai.columns else []
        if not matches_df.empty:
            mpairs = matches_df[matches_df["requirement_id"]==rid].apply(lambda r: f"{r['manual_tc_id']}→{r['ai_tc_id']}", axis=1).tolist()
        else:
            mpairs = []
        rows.append({
            "Requirement ID": rid,
            "Manual TC IDs": ", ".join(manual_ids),
            "AI TC IDs": ", ".join(ai_ids),
            "Matched pairs": ", ".join(mpairs)
        })
    return pd.DataFrame(rows)


# -----------------------------
# Export comparison workbook
# -----------------------------
def export_comparison_excel(
    out_path: Path,
    matches: pd.DataFrame,
    ai_only: pd.DataFrame,
    manual_only: pd.DataFrame,
    scores_summary: pd.DataFrame,
    dist_by_category: pd.DataFrame,
    per_req_density: pd.DataFrame,
    trace_matrix: pd.DataFrame,
    run_info: Dict[str, str] | None = None
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, mode="w") as w:
        # summary sheet (nice front page)
        _grade_df = _grade_from_scores(scores_summary)
        _grade_df.to_excel(w, sheet_name="summary", index=False)

        matches.to_excel(w, sheet_name="matches", index=False)
        ai_only.to_excel(w, sheet_name="ai_only", index=False)
        manual_only.to_excel(w, sheet_name="manual_only", index=False)
        scores_summary.to_excel(w, sheet_name="scores_summary", index=False)
        dist_by_category.to_excel(w, sheet_name="dist_by_category", index=False)
        per_req_density.to_excel(w, sheet_name="per_requirement_density", index=False)
        trace_matrix.to_excel(w, sheet_name="trace_matrix", index=False)

        if run_info:
            meta_df = pd.DataFrame(list(run_info.items()), columns=["Key", "Value"])
            meta_df.to_excel(w, sheet_name="run_info", index=False)


# -----------------------------
# Orchestrator
# -----------------------------
def run_comparison(
    df_ai: pd.DataFrame,
    df_manual: pd.DataFrame,
    strategy: str = "title_expected",
    threshold: float = 0.60
) -> Dict[str, pd.DataFrame | dict]:
    # sanitize columns
    for col in ["tc_id","title","expected","steps","requirement_id","category","gherkin","type"]:
        if col not in df_ai.columns: df_ai[col] = ""
        if col not in df_manual.columns: df_manual[col] = ""

    matches_df, manual_only_df, ai_only_df = best_match_greedy(df_manual, df_ai, strategy, threshold)

    # basic metrics
    manual_total = len(df_manual)
    ai_total = len(df_ai)
    matches_count = len(matches_df)

    coverage = (matches_count / manual_total) if manual_total else 0.0       # how much of manual is covered by AI
    novelty  = (len(ai_only_df) / ai_total) if ai_total else 0.0             # how much AI created beyond manual

    # precision / recall / F1 + grade (0..10)
    precision = (matches_count / ai_total) if ai_total else 0.0
    recall    = (matches_count / manual_total) if manual_total else 0.0
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
    grade = round(f1*10, 1)

    # quality signals
    q_ai = lint_quality(df_ai)
    q_manual = lint_quality(df_manual)

    scores = pd.DataFrame([
        ["manual_total", manual_total],
        ["ai_total", ai_total],
        ["matches_count", matches_count],
        ["manual_only_count", len(manual_only_df)],
        ["ai_only_count", len(ai_only_df)],
        ["coverage", round(float(coverage), 3)],
        ["novelty", round(float(novelty), 3)],
        ["precision", round(float(precision), 3)],
        ["recall", round(float(recall), 3)],
        ["f1_score", round(float(f1), 3)],
        ["grade", grade],
        ["quality_ai", q_ai["quality"]],
        ["quality_manual", q_manual["quality"]],
        ["steps_imperative_ai", q_ai["steps_imperative"]],
        ["steps_imperative_manual", q_manual["steps_imperative"]],
        ["expected_verifiable_ai", q_ai["expected_verifiable"]],
        ["expected_verifiable_manual", q_manual["expected_verifiable"]],
        ["gherkin_complete_ai", q_ai["gherkin_complete"]],
        ["gherkin_complete_manual", q_manual["gherkin_complete"]],
    ], columns=["Key","Value"])

    dist = build_category_dists(df_ai, df_manual)
    density = build_per_requirement_density(df_ai, df_manual, matches_df)
    trace = build_trace_matrix(matches_df, df_manual, df_ai)

    return dict(
        matches=matches_df,
        ai_only=ai_only_df,
        manual_only=manual_only_df,
        scores_summary=scores,
        dist_by_category=dist,
        per_requirement_density=density,
        trace_matrix=trace,
    )


# -----------------------------
# Helpers
# -----------------------------
def _grade_from_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pull f1_score from scores_summary and create a 0..10 grade row for an easy front page.
    """
    try:
        f1_val = float(scores_df.loc[scores_df["Key"]=="f1_score","Value"].iloc[0])
    except Exception:
        f1_val = 0.0
    return pd.DataFrame([
        ["Overall grade (0..10)", round(max(0.0, min(1.0, f1_val))*10, 1)]
    ], columns=["Metric","Value"])
