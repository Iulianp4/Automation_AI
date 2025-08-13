from pathlib import Path
import os
import pandas as pd
from src import preprocess
from src.config import DATA_FILES, EXECUTION_EXPORT_HEADERS
from src.generate_gpt import generate_with_gpt

BASE = Path(__file__).resolve().parent

def export_excel(gen_df: pd.DataFrame, out_path: Path):
    os.makedirs(out_path.parent, exist_ok=True)
    with pd.ExcelWriter(out_path, mode="w") as w:
        gen_df.to_excel(w, sheet_name="generated_raw", index=False)
        exec_df = pd.DataFrame({
            "Nr.Crt": list(range(1, len(gen_df)+1)),
            "Steps": gen_df["steps"],
            "Actual Result": ["" for _ in range(len(gen_df))],
            "Expected Result": gen_df["expected"],
            "Document of evidence": ["" for _ in range(len(gen_df))]
        })[EXECUTION_EXPORT_HEADERS]
        exec_df.to_excel(w, sheet_name="execution_export", index=False)
    print(f"✔ Salvat la: {out_path}")

def generate_from_requirements(req_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in req_df.iterrows():
        rid   = r.get("requirement_id", "")
        rtext = str(r.get("requirement_text", "") or "")
        ac_list = r.get("ac_list", []) or []
        uc_list = r.get("uc_list", []) or []
        cases = generate_with_gpt(rtext, ac_list, uc_list)
        for i, c in enumerate(cases, start=1):
            rows.append({
                "requirement_id": rid,
                "tc_id": f"AIGEN-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "type": "generated"
            })
    return pd.DataFrame(rows)

def generate_from_acceptance(ac_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, grp in ac_df.groupby("requirement_id"):
        ac_list = grp["ac_text"].astype(str).tolist()
        # Requirement text derivat din AC (fallback simplu)
        req_text = "Generate test cases that satisfy the acceptance criteria below:\n" + "\n".join(f"- {t}" for t in ac_list)
        cases = generate_with_gpt(req_text, ac_list, [])
        for i, c in enumerate(cases, start=1):
            rows.append({
                "requirement_id": rid,
                "tc_id": f"AIGEN-AC-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "type": "generated_from_ac"
            })
    return pd.DataFrame(rows)

def generate_from_use_cases(uc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, grp in uc_df.groupby("requirement_id"):
        uc_list = grp["uc_text"].astype(str).tolist()
        req_text = "Generate UI test cases from the following use case descriptions:\n" + "\n---\n".join(uc_list)
        cases = generate_with_gpt(req_text, [], uc_list)
        for i, c in enumerate(cases, start=1):
            rows.append({
                "requirement_id": rid,
                "tc_id": f"AIGEN-UC-{rid}-{i}",
                "title": c["title"],
                "preconditions": c["preconditions"],
                "steps": c["steps"],
                "data": c["data"],
                "expected": c["expected"],
                "type": "generated_from_uc"
            })
    return pd.DataFrame(rows)

def main():
    # citiri independente
    req_only = preprocess.read_requirements()
    ac_only  = preprocess.read_acceptance()
    uc_only  = preprocess.read_use_cases()

    has_req = not req_only.empty
    has_ac  = not ac_only.empty
    has_uc  = not uc_only.empty

    print(f"📊 Date găsite: requirements={len(req_only)} rânduri, AC={len(ac_only)} rânduri, UC={len(uc_only)} rânduri.")

    # 1) dacă există doar requirements -> generăm din ele
    if has_req and not has_ac and not has_uc:
        # adaugă coloane goale pt. ac_list/uc_list
        req_df, _, _ = preprocess.load_all()  # folosește agregarea (ac_list/uc_list pot fi goale)
        gen_df = generate_from_requirements(req_df)
        export_excel(gen_df, BASE / "results" / "report_from_requirements.xlsx")
        return

    # 2) doar AC
    if has_ac and not has_req and not has_uc:
        gen_df = generate_from_acceptance(ac_only)
        export_excel(gen_df, BASE / "results" / "report_from_acceptance.xlsx")
        return

    # 3) doar UC
    if has_uc and not has_req and not has_ac:
        gen_df = generate_from_use_cases(uc_only)
        export_excel(gen_df, BASE / "results" / "report_from_use_cases.xlsx")
        return

    # 4) combinat (cel mai complet)
    if has_req:
        req_df, _, _ = preprocess.load_all()
        gen_df = generate_from_requirements(req_df)
        export_excel(gen_df, BASE / DATA_FILES["report"])
        return

    print("⚠️ Nu există date în niciunul dintre fișiere. Completează măcar unul dintre: requirements, acceptance_criteria, use_cases.")

if __name__ == "__main__":
    main()
