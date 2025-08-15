from pathlib import Path
import os
import pandas as pd
from src import preprocess
from src.config import DATA_FILES, EXECUTION_EXPORT_HEADERS
from src.generate_gpt import generate_with_gpt

BASE = Path(__file__).resolve().parent

# ---------- utils ----------
def _ensure_list(v):
    """normalizează orice la listă de stringuri; NaN/None/vid -> []"""
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if v is None:
        return []
    try:
        import pandas as pd
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
def export_excel(gen_df_internal: pd.DataFrame, out_path: Path):
    """
    gen_df_internal coloane interne:
    requirement_id, requirement_name, tc_id, title, preconditions, steps, data, expected, type
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
                "Identificator unic pentru cerință. Dacă lipsește, se atribuie automat (ex. REQ-LONE-<n>).",
                "Numele cerinței, folosit pentru identificare rapidă.",
                "Descrierea cerinței; dacă este completată (sau Name), rândul este acceptat.",
                "Motivația pentru cerință (opțional).",
                "Platforma/mediul țintă (opțional).",
                "Context suplimentar oferit de tester pentru generare mai bogată.",
                # AC
                "ID-ul story-ului asociat criteriului de acceptare.",
                "Textul criteriului de acceptare; dacă este completat, rândul este acceptat.",
                "Note adiționale pentru criteriul de acceptare.",
                "Comentarii privind criteriul de acceptare.",
                "Context suplimentar oferit de tester pentru criteriu.",
                # UC
                "ID-ul story-ului asociat use case-ului.",
                "Titlul use case-ului.",
                "Informații documentare despre use case.",
                "Istoricul reviziilor pentru use case.",
                "Descrierea use case-ului; folosită în generare.",
                "Condiții necesare înainte de execuție.",
                "Fluxul principal de pași al use case-ului.",
                "Fluxuri alternative.",
                "Fluxuri de excepție (eroare).",
                "Reguli de business asociate.",
                "Context suplimentar oferit de tester pentru use case.",
                # Synthetic
                "Dacă Requirement ID lipsește/este gol -> REQ-LONE-<n>.",
                "Dacă Acceptance Criteria Story ID lipsește/este gol -> AC-LONE-<n>.",
                "Dacă Use Case Story ID lipsește/este gol -> UC-LONE-<n>.",
            ]
        })
        legend.to_excel(w, sheet_name="legend", index=False)

    print(f"✔ Salvat la: {out_path}")

# ---------- Generatoare per sursă ----------
def generate_from_requirements(req_df: pd.DataFrame, num_tests: int) -> pd.DataFrame:
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

        cases = generate_with_gpt(rtext, ac_list, uc_list, num_tests=num_tests, extra_details=details)
        for i, c in enumerate(cases, start=1):
            rows.append({
                "requirement_id": rid,
                "requirement_name": rname,
                "tc_id": f"AIGEN-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "type": "Generated from Requirements"
            })
    return pd.DataFrame(rows)

def generate_from_acceptance(ac_df: pd.DataFrame, req_name_map: dict, num_tests: int) -> pd.DataFrame:
    rows = []
    if ac_df.empty:
        return pd.DataFrame(columns=["requirement_id","requirement_name","tc_id","title","preconditions","steps","data","expected","type"])
    for rid, grp in ac_df.groupby("requirement_id"):
        ac_list = _ensure_list(grp["ac_text"].astype(str).tolist())
        # adaugă details de la AC (toate rândurile grupului)
        ac_details = ""
        if "ac_details" in grp.columns:
            dets = [ _clean_text(x) for x in grp["ac_details"].tolist() if _clean_text(x) ]
            if dets:
                ac_details = "\n".join(dets)

        if not (ac_list or ac_details):
            continue

        rname = req_name_map.get(str(rid).strip(), "")
        prompt_text = "Generate test cases that satisfy the acceptance criteria below:\n" + "\n".join(f"- {t}" for t in ac_list)
        cases = generate_with_gpt(prompt_text, ac_list, [], num_tests=num_tests, extra_details=ac_details)
        for i, c in enumerate(cases, start=1):
            rows.append({
                "requirement_id": str(rid).strip(),
                "requirement_name": rname,
                "tc_id": f"AIGEN-AC-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "type": "Generated from Acceptance Criteria"
            })
    return pd.DataFrame(rows)

def generate_from_use_cases(uc_df: pd.DataFrame, req_name_map: dict, num_tests: int) -> pd.DataFrame:
    rows = []
    if uc_df.empty:
        return pd.DataFrame(columns=["requirement_id","requirement_name","tc_id","title","preconditions","steps","data","expected","type"])
    for rid, grp in uc_df.groupby("requirement_id"):
        uc_list = _ensure_list(grp["uc_text"].astype(str).tolist())
        uc_details = ""
        if "uc_details" in grp.columns:
            dets = [ _clean_text(x) for x in grp["uc_details"].tolist() if _clean_text(x) ]
            if dets:
                uc_details = "\n".join(dets)

        if not (uc_list or uc_details):
            continue

        rname = req_name_map.get(str(rid).strip(), "")
        prompt_text = "Generate UI test cases from the following use case descriptions:\n" + "\n---\n".join(uc_list)
        cases = generate_with_gpt(prompt_text, [], uc_list, num_tests=num_tests, extra_details=uc_details)
        for i, c in enumerate(cases, start=1):
            rows.append({
                "requirement_id": str(rid).strip(),
                "requirement_name": rname,
                "tc_id": f"AIGEN-UC-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "type": "Generated from Use Cases"
            })
    return pd.DataFrame(rows)

# ---------- Main ----------
def main():
    # input de la utilizator (număr de teste per categorie)
    def ask_int(prompt_text, default_val):
        try:
            s = input(f"{prompt_text} (default {default_val}): ").strip()
            return int(s) if s.isdigit() and int(s) > 0 else default_val
        except Exception:
            return default_val

    num_req = ask_int("Cate teste per REQUIREMENT?", 5)
    num_ac  = ask_int("Cate teste per ACCEPTANCE CRITERIA group (per Requirement ID)?", 5)
    num_uc  = ask_int("Cate teste per USE CASE group (per Requirement ID)?", 5)

    # citiri independente
    req_only = preprocess.read_requirements()
    ac_only  = preprocess.read_acceptance()
    uc_only  = preprocess.read_use_cases()

    has_req = not req_only.empty
    has_ac  = not ac_only.empty
    has_uc  = not uc_only.empty

    req_name_map = {
        str(r.get("requirement_id","")).strip(): r.get("requirement_name","")
        for _, r in req_only.iterrows()
    }

    print(f"📊 Date găsite: requirements={len(req_only)} rânduri, AC={len(ac_only)} rânduri, UC={len(uc_only)} rânduri.")
    consolidated_parts = []

    if has_req:
        req_df, _, _ = preprocess.load_all()
        df_req = generate_from_requirements(req_df, num_tests=num_req)
        export_excel(df_req, BASE / "results" / "report_from_requirements.xlsx")
        consolidated_parts.append(df_req)

    if has_ac:
        df_ac = generate_from_acceptance(ac_only, req_name_map, num_tests=num_ac)
        export_excel(df_ac, BASE / "results" / "report_from_acceptance.xlsx")
        consolidated_parts.append(df_ac)

    if has_uc:
        df_uc = generate_from_use_cases(uc_only, req_name_map, num_tests=num_uc)
        export_excel(df_uc, BASE / "results" / "report_from_use_cases.xlsx")
        consolidated_parts.append(df_uc)

    # consolidat independent: concatenează tot ce există
    if consolidated_parts:
        consolidated = pd.concat(consolidated_parts, ignore_index=True)
        export_excel(consolidated, BASE / DATA_FILES["report"])
    else:
        print("⚠️ Nu există date valabile în niciunul dintre fișiere. Completează măcar unul dintre: requirements, acceptance_criteria, use_cases.")

if __name__ == "__main__":
    main()
