from pathlib import Path
import os
import pandas as pd

from src import preprocess
from src.config import (
    DATA_FILES,
    EXECUTION_EXPORT_HEADERS,
    AC_MODE,
    UC_MODE,
    COMPARE_SIM_THRESHOLD,
    COMPARE_STRATEGY,
    INTERACTIVE,
    DEFAULTS,
)
from src.generate_gpt import generate_with_gpt
from src.comparison import run_comparison, export_comparison_excel

BASE = Path(__file__).resolve().parent


# ====================== small helpers ======================
def _ensure_list(v):
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if v is None:
        return []
    try:
        import pandas as pd  # noqa
        if pd.isna(v):
            return []
    except Exception:
        pass
    if isinstance(v, str):
        v = v.strip()
        return [v] if v else []
    return []


def _clean_text(x: str) -> str:
    return (str(x or "").replace("\u00A0", " ").strip())


def ask_int(prompt_text, default_val):
    try:
        s = input(f"{prompt_text} (default {default_val}): ").strip()
        return int(s) if s.isdigit() and int(s) >= 0 else default_val
    except Exception:
        return default_val


def ask_float(prompt_text, default_val):
    try:
        s = input(f"{prompt_text} (default {default_val}): ").strip()
        return float(s) if s else default_val
    except Exception:
        return default_val


def ask_choice(prompt_text, choices: list[str], default_val: str):
    ch = "/".join(choices)
    s = input(f"{prompt_text} [{ch}] (default {default_val}): ").strip().lower()
    return s if s in [c.lower() for c in choices] else default_val


def ask_text(prompt_text, default_val=""):
    try:
        s = input(f"{prompt_text} (default '{default_val}'): ").strip()
        return s if s else default_val
    except Exception:
        return default_val


# ====================== export (Excel) ======================
def build_traceability(gen_df_internal: pd.DataFrame) -> pd.DataFrame:
    if gen_df_internal.empty:
        return pd.DataFrame(columns=[
            "Requirement ID", "Requirement Name", "Test Case ID", "Title",
            "Category", "Has Gherkin", "Source"
        ])

    df = gen_df_internal.copy()
    df["Has Gherkin"] = df.get("gherkin", "").apply(lambda x: "Yes" if str(x).strip() else "No")
    cols = {
        "requirement_id": "Requirement ID",
        "requirement_name": "Requirement Name",
        "tc_id": "Test Case ID",
        "title": "Title",
        "category": "Category",
        "type": "Source",
    }
    out = df.rename(columns=cols)[
        ["Requirement ID", "Requirement Name", "Test Case ID", "Title", "Category", "Has Gherkin", "Source"]
    ].sort_values(["Requirement ID", "Test Case ID"], na_position="last")
    return out.reset_index(drop=True)


def build_metrics(gen_df_internal: pd.DataFrame) -> dict:
    if gen_df_internal.empty:
        empty = pd.DataFrame(columns=["Key", "Value"])
        return {
            "by_source": empty, "by_category": empty, "by_req": empty,
            "by_category_and_source": empty, "totals": empty
        }

    df = gen_df_internal.copy()

    by_source = df.groupby("type").size().reset_index(name="Count").rename(columns={"type": "Source"})
    by_category = df.groupby(df.get("category", "Positive")).size().reset_index(name="Count")
    by_category = by_category.rename(columns={by_category.columns[0]: "Category"})

    by_req = df.groupby("requirement_id").size().reset_index(name="Test Cases")
    by_req = by_req.rename(columns={"requirement_id": "Requirement ID"}).sort_values("Test Cases", ascending=False)

    by_cat_src = df.pivot_table(index="category", columns="type", values="tc_id", aggfunc="count", fill_value=0)
    by_cat_src = by_cat_src.rename_axis("Category").reset_index()

    totals = pd.DataFrame([
        ["Total generated test cases", len(df)],
        ["Distinct Requirement IDs", df["requirement_id"].nunique(dropna=True)],
        ["With Gherkin", (df.get("gherkin","").astype(str).str.strip() != "").sum()],
    ], columns=["Key", "Value"])

    return {
        "by_source": by_source,
        "by_category": by_category,
        "by_req": by_req,
        "by_category_and_source": by_cat_src,
        "totals": totals
    }


def export_excel(gen_df_internal: pd.DataFrame, out_path: Path, meta: dict | None = None):
    if gen_df_internal.empty:
        print(f"ℹ Nothing to save for {out_path.name} (0 cases).")
        return

    friendly = {
        "requirement_id": "Requirement ID",
        "requirement_name": "Requirement Name",
        "tc_id": "Test Case ID",
        "title": "Title",
        "preconditions": "Preconditions",
        "steps": "Steps",
        "data": "Test Data",
        "expected": "Expected Result",
        "category": "Category",
        "gherkin": "Gherkin",
        "type": "Source",
    }
    gen_df = gen_df_internal.rename(columns=friendly)

    trc_df = build_traceability(gen_df_internal)
    metrics = build_metrics(gen_df_internal)

    os.makedirs(out_path.parent, exist_ok=True)
    with pd.ExcelWriter(out_path, mode="w") as w:
        gen_df.to_excel(w, sheet_name="generated_raw", index=False)

        exec_df = pd.DataFrame({
            "Nr.Crt": list(range(1, len(gen_df) + 1)),
            "Steps": gen_df["Steps"],
            "Actual Result": ["" for _ in range(len(gen_df))],
            "Expected Result": gen_df["Expected Result"],
            "Document of evidence": ["" for _ in range(len(gen_df))],
        })[EXECUTION_EXPORT_HEADERS]
        exec_df.to_excel(w, sheet_name="execution_export", index=False)

        trc_df.to_excel(w, sheet_name="traceability", index=False)

        metrics["totals"].to_excel(w, sheet_name="metrics_totals", index=False)
        metrics["by_source"].to_excel(w, sheet_name="metrics_by_source", index=False)
        metrics["by_category"].to_excel(w, sheet_name="metrics_by_category", index=False)
        metrics["by_req"].to_excel(w, sheet_name="metrics_by_req", index=False)
        metrics["by_category_and_source"].to_excel(w, sheet_name="metrics_cat_x_source", index=False)

        legend = pd.DataFrame({
            "Field": [
                # Inputs
                "Requirement ID (requirements.xlsx)",
                "Requirement Name (requirements.xlsx)",
                "Requirement Description (requirements.xlsx)",
                "Requirement Rationale (requirements.xlsx)",
                "Requirement Platform (requirements.xlsx)",
                "Requirement Details (requirements.xlsx)",
                "Acceptance Criteria Story ID (acceptance_criteria.xlsx)",
                "Acceptance Criteria (acceptance_criteria.xlsx)",
                "Acceptance Criteria Notes (acceptance_criteria.xlsx)",
                "Acceptance Criteria Comments (acceptance_criteria.xlsx)",
                "Acceptance Criteria Details (acceptance_criteria.xlsx)",
                "Use Case Story ID (use_cases.xlsx)",
                "Use Case Title (use_cases.xlsx)",
                "Use Case Document Information (use_cases.xlsx)",
                "Use Case Revision History (use_cases.xlsx)",
                "Use Case Description (use_cases.xlsx)",
                "Use Case Preconditions (use_cases.xlsx)",
                "Use Case Main Flow (use_cases.xlsx)",
                "Use Case Alternative Flows (use_cases.xlsx)",
                "Use Case Exception Flows (use_cases.xlsx)",
                "Use Case Business Rules (use_cases.xlsx)",
                "Use Case Details (use_cases.xlsx)",
                # Outputs
                "Test Case ID",
                "Title",
                "Preconditions",
                "Steps",
                "Test Data",
                "Expected Result",
                "Category",
                "Gherkin",
                "Source",
                # Synthetic
                "Synthetic ID - Requirements",
                "Synthetic ID - AC",
                "Synthetic ID - UC",
            ],
            "Definition": [
                # Inputs
                "Unique requirement identifier (auto fallback if missing).",
                "Human-friendly requirement name.",
                "Requirement description used for generation.",
                "Rationale (optional).",
                "Target platform (optional).",
                "Tester-provided extra context.",
                "Story ID for AC.",
                "AC text used directly for generation.",
                "AC notes (optional).",
                "AC comments (optional).",
                "Extra AC context.",
                "Story ID for UC.",
                "UC title.",
                "UC document info (optional).",
                "UC revision history (optional).",
                "UC description/flows used in generation.",
                "UC preconditions.",
                "UC main flow.",
                "UC alternative flows.",
                "UC exception flows.",
                "UC business rules.",
                "Extra UC context.",
                # Outputs
                "Generated identifier for the test case row.",
                "Short test case headline.",
                "State/config required before executing steps.",
                "Actionable, imperative list of steps.",
                "Input data or fixtures needed by the steps.",
                "System behavior expected on success.",
                "Type: Positive / Negative / Boundary / Security / AdHoc.",
                "Given/When/Then verification for the same scenario.",
                "Origin of the test (Requirements / Acceptance / Use Case).",
                # Synthetic
                "If missing -> REQ-LONE-<n>.",
                "If missing -> AC-LONE-<n>.",
                "If missing -> UC-LONE-<n>.",
            ]
        })
        legend.to_excel(w, sheet_name="legend", index=False)

        if meta:
            meta_df = pd.DataFrame(list(meta.items()), columns=["Key", "Value"])
            meta_df.to_excel(w, sheet_name="run_info", index=False)

    print(f"✔ Saved: {out_path}")


# ====================== generators ======================
def generate_from_requirements(
    req_df: pd.DataFrame, num_tests: int,
    output_style: str, include_ad_hoc: bool, mix: str,
    *, temperature: float, seed: int | None
) -> pd.DataFrame:
    rows = []
    for _, r in req_df.iterrows():
        rid   = _clean_text(r.get("requirement_id", ""))
        rname = _clean_text(r.get("requirement_name", ""))
        rtext = _clean_text(r.get("requirement_text", "")) or rname
        details = _clean_text(r.get("requirement_details", ""))

        ac_list = _ensure_list(r.get("ac_list", []))
        uc_list = _ensure_list(r.get("uc_list", []))

        if not (rtext or ac_list or uc_list or details):
            continue

        cases = generate_with_gpt(
            rtext, ac_list, uc_list, num_tests=num_tests,
            extra_details=details, output_style=output_style,
            include_ad_hoc=include_ad_hoc, mix=mix,
            temperature=temperature, seed=seed
        )
        for i, c in enumerate(cases, start=1):
            rows.append({
                "requirement_id": rid,
                "requirement_name": rname,
                "tc_id": f"AIGEN-REQ-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "category": c.get("category", "Positive"),
                "gherkin": c.get("gherkin", ""),
                "type": "Generated from Requirement (per-row)"
            })
    return pd.DataFrame(rows)


def generate_from_acceptance_row(
    ac_df: pd.DataFrame, req_name_map: dict, num_tests: int,
    output_style: str, include_ad_hoc: bool, mix: str,
    *, temperature: float, seed: int | None
) -> pd.DataFrame:
    rows = []
    for _, r in ac_df.iterrows():
        rid = _clean_text(r.get("requirement_id",""))
        rname = req_name_map.get(rid, "")
        ac_text = _clean_text(r.get("ac_text",""))
        details = _clean_text(r.get("ac_details",""))
        if not (ac_text or details):
            continue

        ac_list = _ensure_list(ac_text)
        cases = generate_with_gpt(
            "", ac_list, [], num_tests=num_tests,
            extra_details=details, output_style=output_style,
            include_ad_hoc=include_ad_hoc, mix=mix,
            temperature=temperature, seed=seed
        )
        for i, c in enumerate(cases, 1):
            rows.append({
                "requirement_id": rid,
                "requirement_name": rname,
                "tc_id": f"AIGEN-AC-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "category": c.get("category", "Positive"),
                "gherkin": c.get("gherkin", ""),
                "type": "Generated from Acceptance (per-row)"
            })
    return pd.DataFrame(rows)


def generate_from_acceptance_group(
    ac_df: pd.DataFrame, req_name_map: dict, num_tests: int,
    output_style: str, include_ad_hoc: bool, mix: str,
    *, temperature: float, seed: int | None
) -> pd.DataFrame:
    rows = []
    for rid, grp in ac_df.groupby("requirement_id"):
        ac_list = _ensure_list(grp["ac_text"].astype(str).tolist())
        dets = [_clean_text(x) for x in grp.get("ac_details", []).tolist() if _clean_text(x)]
        details = "\n".join(dets) if dets else ""
        if not (ac_list or details):
            continue
        rname = req_name_map.get(str(rid).strip(), "")
        cases = generate_with_gpt(
            "", ac_list, [], num_tests=num_tests,
            extra_details=details, output_style=output_style,
            include_ad_hoc=include_ad_hoc, mix=mix,
            temperature=temperature, seed=seed
        )
        for i, c in enumerate(cases, 1):
            rows.append({
                "requirement_id": str(rid).strip(),
                "requirement_name": rname,
                "tc_id": f"AIGEN-AC-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "category": c.get("category", "Positive"),
                "gherkin": c.get("gherkin", ""),
                "type": "Generated from Acceptance (group)"
            })
    return pd.DataFrame(rows)


def generate_from_use_cases_row(
    uc_df: pd.DataFrame, req_name_map: dict, num_tests: int,
    output_style: str, include_ad_hoc: bool, mix: str,
    *, temperature: float, seed: int | None
) -> pd.DataFrame:
    rows = []
    for _, r in uc_df.iterrows():
        rid = _clean_text(r.get("requirement_id",""))
        rname = req_name_map.get(rid, "")
        uc_text = _clean_text(r.get("uc_text",""))
        details = _clean_text(r.get("uc_details",""))
        if not (uc_text or details):
            continue

        uc_list = _ensure_list(uc_text)
        cases = generate_with_gpt(
            "", [], uc_list, num_tests=num_tests,
            extra_details=details, output_style=output_style,
            include_ad_hoc=include_ad_hoc, mix=mix,
            temperature=temperature, seed=seed
        )
        for i, c in enumerate(cases, 1):
            rows.append({
                "requirement_id": rid,
                "requirement_name": rname,
                "tc_id": f"AIGEN-UC-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "category": c.get("category", "Positive"),
                "gherkin": c.get("gherkin", ""),
                "type": "Generated from Use Case (per-row)"
            })
    return pd.DataFrame(rows)


def generate_from_use_cases_group(
    uc_df: pd.DataFrame, req_name_map: dict, num_tests: int,
    output_style: str, include_ad_hoc: bool, mix: str,
    *, temperature: float, seed: int | None
) -> pd.DataFrame:
    rows = []
    for rid, grp in uc_df.groupby("requirement_id"):
        uc_list = _ensure_list(grp["uc_text"].astype(str).tolist())
        dets = [_clean_text(x) for x in grp.get("uc_details", []).tolist() if _clean_text(x)]
        details = "\n".join(dets) if dets else ""
        if not (uc_list or details):
            continue
        rname = req_name_map.get(str(rid).strip(), "")
        cases = generate_with_gpt(
            "", [], uc_list, num_tests=num_tests,
            extra_details=details, output_style=output_style,
            include_ad_hoc=include_ad_hoc, mix=mix,
            temperature=temperature, seed=seed
        )
        for i, c in enumerate(cases, 1):
            rows.append({
                "requirement_id": str(rid).strip(),
                "requirement_name": rname,
                "tc_id": f"AIGEN-UC-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "category": c.get("category", "Positive"),
                "gherkin": c.get("gherkin", ""),
                "type": "Generated from Use Case (group)"
            })
    return pd.DataFrame(rows)


# ====================== comparison-only loader ======================
def _load_ai_from_results(results_dir: Path) -> pd.DataFrame:
    parts = []
    for name in [
        "report_from_requirements.xlsx",
        "report_from_acceptance.xlsx",
        "report_from_use_cases.xlsx",
        "report.xlsx",
    ]:
        p = results_dir / name
        if p.exists():
            try:
                df = pd.read_excel(p, sheet_name="generated_raw").fillna("")
                rename = {
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
                }
                df = df.rename(columns=rename)
                parts.append(df)
            except Exception:
                pass
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ====================== main ======================
def main():
    # knobs (interactive or fixed defaults)
    if INTERACTIVE:
        num_req = ask_int("How many tests per REQUIREMENT row?", DEFAULTS.get("num_req", 5))
        num_ac  = ask_int(f"How many tests per ACCEPTANCE CRITERIA ({'GROUP' if AC_MODE=='group' else 'ROW'})?", DEFAULTS.get("num_ac", 5))
        num_uc  = ask_int(f"How many tests per USE CASE ({'GROUP' if UC_MODE=='group' else 'ROW'})?", DEFAULTS.get("num_uc", 5))

        output_style   = ask_choice("Output style? (classic/gherkin/both)", ["classic","gherkin","both"], DEFAULTS.get("output_style","both"))
        include_ad_hoc = (ask_choice("Allow AdHoc category? (yes/no)", ["yes","no"], "yes") == "yes")
        mix            = ask_choice("Test mix? (balanced/positive_heavy/negative_heavy)", ["balanced","positive_heavy","negative_heavy"], DEFAULTS.get("mix","balanced"))

        temperature    = ask_float("Temperature 0.0–1.2", DEFAULTS.get("temperature", 0.2))
        s_val          = ask_int("Random seed (0 = auto)", DEFAULTS.get("seed", 0))
        seed           = None if s_val == 0 else s_val

        compare_only   = (ask_choice("Run comparison only (skip generation)?", ["yes","no"], "no") == "yes")

        # FEATURE MODE (single filter REQUIREMENT ID)
        feature_filter = ask_text("Feature mode: filter by Requirement ID (blank = ALL)", "")
        feature_filter = feature_filter.strip()
    else:
        num_req = DEFAULTS.get("num_req", 5)
        num_ac  = DEFAULTS.get("num_ac", 5)
        num_uc  = DEFAULTS.get("num_uc", 5)
        output_style   = DEFAULTS.get("output_style", "both")
        include_ad_hoc = DEFAULTS.get("include_ad_hoc", True)
        mix            = DEFAULTS.get("mix", "balanced")
        compare_only   = DEFAULTS.get("compare_only", False)
        temperature    = float(DEFAULTS.get("temperature", 0.2))
        s_val          = int(DEFAULTS.get("seed", 0))
        seed           = None if s_val == 0 else s_val
        feature_filter = str(DEFAULTS.get("feature_filter", "")).strip()

    # inputs
    req_only = preprocess.read_requirements()
    ac_only  = preprocess.read_acceptance()
    uc_only  = preprocess.read_use_cases()
    manual_df = preprocess.read_manual_cases()

    # validation messages
    preprocess.validate_columns(req_only, ["requirement_id","requirement_name","requirement_text"], "requirements.xlsx")
    preprocess.validate_columns(ac_only,  ["requirement_id","ac_text"], "acceptance_criteria.xlsx")
    preprocess.validate_columns(uc_only,  ["requirement_id","uc_text"], "use_cases.xlsx")
    preprocess.validate_columns(manual_df, ["tc_id","title","expected"], "manual_cases.xlsx")

    # FEATURE MODE filter (if its set)
    if feature_filter:
        req_only = req_only[req_only.get("requirement_id","").astype(str).str.strip() == feature_filter].reset_index(drop=True)
        ac_only  = ac_only[ac_only.get("requirement_id","").astype(str).str.strip() == feature_filter].reset_index(drop=True)
        uc_only  = uc_only[uc_only.get("requirement_id","").astype(str).str.strip() == feature_filter].reset_index(drop=True)

    has_req = not req_only.empty
    has_ac  = not ac_only.empty
    has_uc  = not uc_only.empty
    has_manual = not manual_df.empty

    req_name_map = {
        str(r.get("requirement_id","")).strip(): r.get("requirement_name","")
        for _, r in req_only.iterrows()
    }

    print(f"📊 Found rows: requirements={len(req_only)}, AC={len(ac_only)}, UC={len(uc_only)}, manual={len(manual_df)}"
          + (f" | filter='{feature_filter}'" if feature_filter else ""))

    df_ai_all = pd.DataFrame()
    consolidated_parts = []

    if compare_only:
        df_ai_all = _load_ai_from_results(BASE / "results")
        if feature_filter and not df_ai_all.empty:
            df_ai_all = df_ai_all[df_ai_all.get("requirement_id","").astype(str).str.strip() == feature_filter]
        if df_ai_all.empty:
            print("⚠️ No existing AI results found in results/*.xlsx (generated_raw). Will generate now.")
            compare_only = False

    meta_common = {
        "AC mode": AC_MODE,
        "UC mode": UC_MODE,
        "Req tests per item": num_req,
        "AC tests per item": num_ac,
        "UC tests per item": num_uc,
        "Output style": output_style,
        "AdHoc allowed": include_ad_hoc,
        "Mix": mix,
        "Temperature": temperature,
        "Seed": seed,
        "Feature filter": feature_filter or "ALL",
    }

    # ===== generation flow (skipped if compare_only and we loaded AI) =====
    if not compare_only:
        if has_req and num_req > 0:
            req_df, _, _ = preprocess.load_all()
            if feature_filter:
                req_df = req_df[req_df.get("requirement_id","").astype(str).str.strip() == feature_filter].reset_index(drop=True)
            df_req = generate_from_requirements(
                req_df, num_tests=num_req, output_style=output_style, include_ad_hoc=include_ad_hoc, mix=mix,
                temperature=temperature, seed=seed
            )
            export_excel(
                df_req,
                BASE / "results" / "report_from_requirements.xlsx",
                meta={**meta_common, "Source": "Requirements (per-row)", "Input rows": len(req_only), "Generated cases": len(df_req)}
            )
            consolidated_parts.append(df_req)

        if has_ac and num_ac > 0:
            if AC_MODE == "group":
                df_ac = generate_from_acceptance_group(
                    ac_only, req_name_map, num_tests=num_ac, output_style=output_style, include_ad_hoc=include_ad_hoc, mix=mix,
                    temperature=temperature, seed=seed
                )
                src_label = "Acceptance (group)"
            else:
                df_ac = generate_from_acceptance_row(
                    ac_only, req_name_map, num_tests=num_ac, output_style=output_style, include_ad_hoc=include_ad_hoc, mix=mix,
                    temperature=temperature, seed=seed
                )
                src_label = "Acceptance (per-row)"
            export_excel(
                df_ac,
                BASE / "results" / "report_from_acceptance.xlsx",
                meta={**meta_common, "Source": src_label, "Input rows": len(ac_only), "Generated cases": len(df_ac)}
            )
            consolidated_parts.append(df_ac)

        if has_uc and num_uc > 0:
            if UC_MODE == "group":
                df_uc = generate_from_use_cases_group(
                    uc_only, req_name_map, num_tests=num_uc, output_style=output_style, include_ad_hoc=include_ad_hoc, mix=mix,
                    temperature=temperature, seed=seed
                )
                src_label = "Use Cases (group)"
            else:
                df_uc = generate_from_use_cases_row(
                    uc_only, req_name_map, num_tests=num_uc, output_style=output_style, include_ad_hoc=include_ad_hoc, mix=mix,
                    temperature=temperature, seed=seed
                )
                src_label = "Use Cases (per-row)"
            export_excel(
                df_uc,
                BASE / "results" / "report_from_use_cases.xlsx",
                meta={**meta_common, "Source": src_label, "Input rows": len(uc_only), "Generated cases": len(df_uc)}
            )
            consolidated_parts.append(df_uc)

        if has_manual:
            export_excel(
                manual_df,
                BASE / "results" / "report_manual_baseline.xlsx",
                meta={**meta_common, "Source": "Manual (baseline)", "Input rows": len(manual_df), "Generated cases": len(manual_df)}
            )

        if consolidated_parts:
            consolidated = pd.concat(consolidated_parts, ignore_index=True)
            export_excel(
                consolidated,
                BASE / DATA_FILES["report"],
                meta={**meta_common, "Source": "Consolidated", "Generated cases": len(consolidated)}
            )
            df_ai_all = consolidated.copy()
        else:
            df_ai_all = pd.DataFrame()
            if not (has_req or has_ac or has_uc):
                print("⚠️ No valid input present. Fill at least one of: requirements, acceptance_criteria, use_cases.")
            else:
                print("ℹ Skipped generation for sources with count = 0.")

    # ===== AI vs. Manual comparison =====
    has_ai = not df_ai_all.empty
    if has_ai and has_manual:
        if INTERACTIVE:
            ans = input("Compare AI vs Manual now? (y/N): ").strip().lower()
            do_cmp = (ans == "y")
        else:
            do_cmp = DEFAULTS.get("do_comparison", True)

        if do_cmp:
            if INTERACTIVE:
                strategy_in = input(
                    f"Similarity strategy [title_expected/title_steps_expected] (default {COMPARE_STRATEGY}): "
                ).strip().lower()
                strategy = strategy_in if strategy_in in ("title_expected", "title_steps_expected") else COMPARE_STRATEGY

                th_in = input(f"Similarity threshold 0.0..1.0 (default {COMPARE_SIM_THRESHOLD}): ").strip()
                try:
                    threshold = float(th_in) if th_in else COMPARE_SIM_THRESHOLD
                except Exception:
                    threshold = COMPARE_SIM_THRESHOLD
            else:
                strategy = DEFAULTS.get("similarity_strategy", COMPARE_STRATEGY)
                threshold = float(DEFAULTS.get("similarity_threshold", COMPARE_SIM_THRESHOLD))

            mdf = manual_df.copy()
            if feature_filter:
                mdf = mdf[mdf.get("requirement_id","").astype(str).str.strip() == feature_filter]

            cmp_res = run_comparison(df_ai_all, mdf, strategy=strategy, threshold=threshold)
            export_comparison_excel(
                BASE / "results" / "report_comparison.xlsx",
                matches=cmp_res["matches"],
                ai_only=cmp_res["ai_only"],
                manual_only=cmp_res["manual_only"],
                scores_summary=cmp_res["scores_summary"],
                dist_by_category=cmp_res["dist_by_category"],
                per_req_density=cmp_res["per_requirement_density"],
                trace_matrix=cmp_res["trace_matrix"],
                run_info={
                    "strategy": strategy,
                    "threshold": threshold,
                    "ai_total": len(df_ai_all),
                    "manual_total": len(mdf),
                    "feature_filter": feature_filter or "ALL",
                }
            )
            print("✔ comparison report saved to results/report_comparison.xlsx")
    else:
        if not has_ai:
            print("ℹ No AI-generated tests available for comparison.")
        if not has_manual:
            print("ℹ No manual baseline loaded (data/manual_cases.xlsx is empty or missing).")


if __name__ == "__main__":
    main()
