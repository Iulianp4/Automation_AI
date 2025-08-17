from pathlib import Path
import os
import pandas as pd
from src import preprocess
from src.config import DATA_FILES, EXECUTION_EXPORT_HEADERS, AC_MODE, UC_MODE
from src.generate_gpt import generate_with_gpt

BASE = Path(__file__).resolve().parent

# ---------- utils ----------
def _ensure_list(v):
    """normalizeaza orice la lista de stringuri; NaN/None/vid -> []"""
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if v is None:
        return []
    try:
        import pandas as pd  # local to avoid global import clash
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

# ---------- Export helper (user-friendly) ----------
def export_excel(gen_df_internal: pd.DataFrame, out_path: Path, meta: dict | None = None):
    """
    gen_df_internal: requirement_id, requirement_name, tc_id, title, preconditions, steps, data, expected, type
    meta: info de rulare -> scris in sheet 'run_info'
    """
    if gen_df_internal.empty:
        print(f"ℹ Nimic de salvat pentru {out_path.name} (0 cazuri).")
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
        "type": "Source",
    }
    gen_df = gen_df_internal.rename(columns=friendly)

    os.makedirs(out_path.parent, exist_ok=True)
    with pd.ExcelWriter(out_path, mode="w") as w:
        # 1) generated_raw
        gen_df.to_excel(w, sheet_name="generated_raw", index=False)

        # 2) execution_export
        exec_df = pd.DataFrame({
            "Nr.Crt": list(range(1, len(gen_df) + 1)),
            "Steps": gen_df["Steps"],
            "Actual Result": ["" for _ in range(len(gen_df))],
            "Expected Result": gen_df["Expected Result"],
            "Document of evidence": ["" for _ in range(len(gen_df))],
        })[EXECUTION_EXPORT_HEADERS]
        exec_df.to_excel(w, sheet_name="execution_export", index=False)

        # 3) legend
        legend = pd.DataFrame({
            "Field": [
                # Requirements
                "Requirement ID (requirements.xlsx)",
                "Requirement Name (requirements.xlsx)",
                "Requirement Description (requirements.xlsx)",
                "Requirement Rationale (requirements.xlsx)",
                "Requirement Platform (requirements.xlsx)",
                "Requirement Details (requirements.xlsx)",
                # AC
                "Acceptance Criteria Story ID (acceptance_criteria.xlsx)",
                "Acceptance Criteria (acceptance_criteria.xlsx)",
                "Acceptance Criteria Notes (acceptance_criteria.xlsx)",
                "Acceptance Criteria Comments (acceptance_criteria.xlsx)",
                "Acceptance Criteria Details (acceptance_criteria.xlsx)",
                # UC
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
                # Synthetic
                "Synthetic ID - Requirements",
                "Synthetic ID - AC",
                "Synthetic ID - UC",
            ],
            "Definition": [
                # Requirements
                "Identificator unic (fallback automat daca lipseste).",
                "Numele cerintei.",
                "Descrierea cerintei (sau Name) folosita in generare.",
                "Motivatie (optional).",
                "Platforma tinta (optional).",
                "Context suplimentar de la tester pentru generare mai bogata.",
                # AC
                "ID-ul story-ului asociat criteriului.",
                "Text criteriu AC (folosit direct la generare).",
                "Note (optional).",
                "Comentarii (optional).",
                "Context suplimentar pentru AC.",
                # UC
                "ID-ul story-ului asociat use case-ului.",
                "Titlul use case-ului.",
                "Informatii document (optional).",
                "Istoric revizii (optional).",
                "Descriere/fluxuri folosite in generare.",
                "Preconditii.",
                "Flux principal.",
                "Fluxuri alternative.",
                "Fluxuri de exceptie.",
                "Reguli de business.",
                "Context suplimentar pentru UC.",
                # Synthetic
                "Daca lipseste -> REQ-LONE-<n>.",
                "Daca lipseste -> AC-LONE-<n>.",
                "Daca lipseste -> UC-LONE-<n>.",
            ]
        })
        legend.to_excel(w, sheet_name="legend", index=False)

        # 4) run_info (meta)
        if meta:
            meta_df = pd.DataFrame(list(meta.items()), columns=["Key", "Value"])
            meta_df.to_excel(w, sheet_name="run_info", index=False)

    print(f"✔ Salvat la: {out_path}")

# ---------- REQUIREMENTS (per-rand) ----------
def generate_from_requirements(req_df: pd.DataFrame, num_tests: int) -> pd.DataFrame:
    rows = []
    for _, r in req_df.iterrows():
        rid   = _clean_text(r.get("requirement_id", ""))
        rname = _clean_text(r.get("requirement_name", ""))
        rtext = _clean_text(r.get("requirement_text", "")) or rname
        details = _clean_text(r.get("requirement_details", ""))

        # extra context (daca merge din load_all)
        ac_list = _ensure_list(r.get("ac_list", []))
        uc_list = _ensure_list(r.get("uc_list", []))

        if not (rtext or ac_list or uc_list or details):
            continue

        cases = generate_with_gpt(rtext, ac_list, uc_list, num_tests=num_tests, extra_details=details)
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
                "type": "Generated from Requirement (per-row)"
            })
    return pd.DataFrame(rows)

# ---------- AC: per-rand ----------
def generate_from_acceptance_row(ac_df: pd.DataFrame, req_name_map: dict, num_tests: int) -> pd.DataFrame:
    rows = []
    for _, r in ac_df.iterrows():
        rid = _clean_text(r.get("requirement_id",""))
        rname = req_name_map.get(rid, "")
        ac_text = _clean_text(r.get("ac_text",""))
        details = _clean_text(r.get("ac_details",""))
        if not (ac_text or details):
            continue
        ac_list = _ensure_list(ac_text)  # un singur element per rand
        cases = generate_with_gpt("", ac_list, [], num_tests=num_tests, extra_details=details)
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
                "type": "Generated from Acceptance (per-row)"
            })
    return pd.DataFrame(rows)

# ---------- AC: pe grup (Requirement ID) ----------
def generate_from_acceptance_group(ac_df: pd.DataFrame, req_name_map: dict, num_tests: int) -> pd.DataFrame:
    rows = []
    for rid, grp in ac_df.groupby("requirement_id"):
        ac_list = _ensure_list(grp["ac_text"].astype(str).tolist())
        dets = [ _clean_text(x) for x in grp.get("ac_details", []).tolist() if _clean_text(x) ]
        details = "\n".join(dets) if dets else ""
        if not (ac_list or details):
            continue
        rname = req_name_map.get(str(rid).strip(), "")
        cases = generate_with_gpt("", ac_list, [], num_tests=num_tests, extra_details=details)
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
                "type": "Generated from Acceptance (group)"
            })
    return pd.DataFrame(rows)

# ---------- UC: per-rand ----------
def generate_from_use_cases_row(uc_df: pd.DataFrame, req_name_map: dict, num_tests: int) -> pd.DataFrame:
    rows = []
    for _, r in uc_df.iterrows():
        rid = _clean_text(r.get("requirement_id",""))
        rname = req_name_map.get(rid, "")
        uc_text = _clean_text(r.get("uc_text",""))
        details = _clean_text(r.get("uc_details",""))
        if not (uc_text or details):
            continue
        uc_list = _ensure_list(uc_text)  # un singur element per rand
        cases = generate_with_gpt("", [], uc_list, num_tests=num_tests, extra_details=details)
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
                "type": "Generated from Use Case (per-row)"
            })
    return pd.DataFrame(rows)

# ---------- UC: pe grup (Requirement ID) ----------
def generate_from_use_cases_group(uc_df: pd.DataFrame, req_name_map: dict, num_tests: int) -> pd.DataFrame:
    rows = []
    for rid, grp in uc_df.groupby("requirement_id"):
        uc_list = _ensure_list(grp["uc_text"].astype(str).tolist())
        dets = [ _clean_text(x) for x in grp.get("uc_details", []).tolist() if _clean_text(x) ]
        details = "\n".join(dets) if dets else ""
        if not (uc_list or details):
            continue
        rname = req_name_map.get(str(rid).strip(), "")
        cases = generate_with_gpt("", [], uc_list, num_tests=num_tests, extra_details=details)
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
                "type": "Generated from Use Case (group)"
            })
    return pd.DataFrame(rows)

# ---------- Main ----------
def main():
    # input dinamice (in functie de moduri)
    def ask_int(prompt_text, default_val):
        try:
            s = input(f"{prompt_text} (default {default_val}): ").strip()
            return int(s) if s.isdigit() and int(s) > 0 else default_val
        except Exception:
            return default_val

    req_prompt = "Cate teste per RAND din REQUIREMENTS?"
    ac_prompt  = "Cate teste per ACCEPTANCE CRITERIA {}?".format("GROUP (per Requirement ID)" if AC_MODE=="group" else "RAND")
    uc_prompt  = "Cate teste per USE CASE {}?".format("GROUP (per Requirement ID)" if UC_MODE=="group" else "RAND")

    num_req = ask_int(req_prompt, 5)
    num_ac  = ask_int(ac_prompt, 5)
    num_uc  = ask_int(uc_prompt, 5)

    # citiri independente
    req_only = preprocess.read_requirements()
    ac_only  = preprocess.read_acceptance()
    uc_only  = preprocess.read_use_cases()

    has_req = not req_only.empty
    has_ac  = not ac_only.empty
    has_uc  = not uc_only.empty

    # map nume requirement_id -> requirement_name
    req_name_map = {
        str(r.get("requirement_id","")).strip(): r.get("requirement_name","")
        for _, r in req_only.iterrows()
    }

    print(f"📊 Date găsite: requirements={len(req_only)} rânduri, AC={len(ac_only)} rânduri, UC={len(uc_only)} rânduri.")
    consolidated_parts = []

    # meta comun
    meta_common = {
        "AC mode": AC_MODE,
        "UC mode": UC_MODE,
        "Req tests per item": num_req,
        "AC tests per item": num_ac,
        "UC tests per item": num_uc,
    }

    # REQUIREMENTS
    if has_req:
        req_df, _, _ = preprocess.load_all()  # contine si ac_list/uc_list
        df_req = generate_from_requirements(req_df, num_tests=num_req)
        export_excel(
            df_req,
            BASE / "results" / "report_from_requirements.xlsx",
            meta={**meta_common, "Source": "Requirements (per-row)", "Input rows": len(req_only), "Generated cases": len(df_req)}
        )
        consolidated_parts.append(df_req)

    # ACCEPTANCE
    if has_ac:
        if AC_MODE == "group":
            df_ac = generate_from_acceptance_group(ac_only, req_name_map, num_tests=num_ac)
            src_label = "Acceptance (group)"
        else:
            df_ac = generate_from_acceptance_row(ac_only, req_name_map, num_tests=num_ac)
            src_label = "Acceptance (per-row)"
        export_excel(
            df_ac,
            BASE / "results" / "report_from_acceptance.xlsx",
            meta={**meta_common, "Source": src_label, "Input rows": len(ac_only), "Generated cases": len(df_ac)}
        )
        consolidated_parts.append(df_ac)

    # USE CASES
    if has_uc:
        if UC_MODE == "group":
            df_uc = generate_from_use_cases_group(uc_only, req_name_map, num_tests=num_uc)
            src_label = "Use Cases (group)"
        else:
            df_uc = generate_from_use_cases_row(uc_only, req_name_map, num_tests=num_uc)
            src_label = "Use Cases (per-row)"
        export_excel(
            df_uc,
            BASE / "results" / "report_from_use_cases.xlsx",
            meta={**meta_common, "Source": src_label, "Input rows": len(uc_only), "Generated cases": len(df_uc)}
        )
        consolidated_parts.append(df_uc)

    # CONSOLIDAT
    if consolidated_parts:
        consolidated = pd.concat(consolidated_parts, ignore_index=True)
        export_excel(
            consolidated,
            BASE / DATA_FILES["report"],
            meta={**meta_common, "Source": "Consolidated", "Generated cases": len(consolidated)}
        )
    else:
        print("⚠️ Nu exista date valabile. Completeaza macar unul dintre: requirements, acceptance_criteria, use_cases.")

if __name__ == "__main__":
    main()
