import os
from pathlib import Path
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

# project
from src import preprocess
from src.generate_gpt import generate_with_gpt
from src.comparison import run_comparison, export_comparison_excel
from src.config import DATA_FILES, AC_MODE, UC_MODE
from src import profiles  # save/load settings

load_dotenv()

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results"
DATA_DIR = BASE / "data"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

st.set_page_config(page_title="Automation_AI – Test Case Generator", layout="wide")

# ---------- Helpers ----------
def save_upload(file, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(file.getbuffer())

def bytes_of_file(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# reuse main.export_excel (same workbook structure)
def export_excel(gen_df_internal: pd.DataFrame, out_path: Path, meta: dict | None = None):
    from main import export_excel as main_export
    return main_export(gen_df_internal, out_path, meta=meta)

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

def ui_log(msg: str):
    try:
        st.toast(msg)
    except Exception:
        st.write(msg)

# ---------- Sidebar ----------
st.sidebar.title("Settings")

# Generation knobs (allow 0)
st.sidebar.subheader("Generation")
num_req = st.sidebar.number_input("Tests per REQUIREMENT row", min_value=0, max_value=50, value=5, step=1)
num_ac  = st.sidebar.number_input(f"Tests per ACCEPTANCE ({'GROUP' if AC_MODE=='group' else 'ROW'})", min_value=0, max_value=50, value=5, step=1)
num_uc  = st.sidebar.number_input(f"Tests per USE CASE ({'GROUP' if UC_MODE=='group' else 'ROW'})", min_value=0, max_value=50, value=5, step=1)

output_style = st.sidebar.selectbox("Output style", ["both", "classic", "gherkin"], index=0)
include_ad_hoc = st.sidebar.checkbox("Allow AdHoc category", value=True)
mix = st.sidebar.selectbox("Test mix", ["balanced","positive_heavy","negative_heavy"], index=0)

# Model controls
st.sidebar.subheader("Model controls")
temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.2, 0.05)
seed = st.sidebar.number_input("Random seed (0 = auto)", min_value=0, max_value=1_000_000, value=0, step=1)

# Comparison
st.sidebar.subheader("Comparison")
similarity_strategy = st.sidebar.selectbox("Similarity strategy", ["title_expected","title_steps_expected"], index=0)
similarity_threshold = st.sidebar.slider("Similarity threshold", 0.50, 0.95, 0.75, 0.01)

# Data source
st.sidebar.subheader("Data source")
use_uploaded = st.sidebar.radio("Use data from", ["Existing /data", "Upload now"], index=0)

# Profiles
st.sidebar.subheader("Profiles")
with st.sidebar.expander("Save / Load profiles", expanded=False):
    current_settings = {
        "num_req": int(num_req),
        "num_ac": int(num_ac),
        "num_uc": int(num_uc),
        "output_style": output_style,
        "include_ad_hoc": bool(include_ad_hoc),
        "mix": mix,
        "temperature": float(temperature),
        "seed": int(seed),
        "similarity_strategy": similarity_strategy,
        "similarity_threshold": float(similarity_threshold),
    }
    prof_name = st.text_input("Profile name", value="my_profile")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("Save profile"):
            profiles.save_profile(prof_name, current_settings)
            st.success(f"Saved profile: {prof_name}")
    with col_p2:
        options = profiles.list_profiles()
        pick = st.selectbox("Load profile", options, index=0 if options else None)
    with col_p3:
        if st.button("Load"):
            if pick:
                payload = profiles.load_profile(pick)
                if payload:
                    st.session_state["loaded_profile"] = payload
                    st.info("Profile loaded into session_state['loaded_profile']. Click Rerun to apply to controls.")

# ---------- Main UI ----------
st.title("Automation_AI — UI Test Case Generation & Comparison")

with st.expander("Upload data (optional)", expanded=(use_uploaded == "Upload now")):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        req_file = st.file_uploader("requirements.xlsx", type=["xlsx"])
        if req_file is not None:
            save_upload(req_file, BASE / DATA_FILES["requirements"])
            st.success("requirements.xlsx uploaded.")
    with col2:
        ac_file = st.file_uploader("acceptance_criteria.xlsx", type=["xlsx"])
        if ac_file is not None:
            save_upload(ac_file, BASE / DATA_FILES["acceptance_criteria"])
            st.success("acceptance_criteria.xlsx uploaded.")
    with col3:
        uc_file = st.file_uploader("use_cases.xlsx", type=["xlsx"])
        if uc_file is not None:
            save_upload(uc_file, BASE / DATA_FILES["use_cases"])
            st.success("use_cases.xlsx uploaded.")
    with col4:
        manual_file = st.file_uploader("manual_cases.xlsx (optional)", type=["xlsx"])
        if manual_file is not None:
            save_upload(manual_file, BASE / DATA_FILES["manual"])
            st.success("manual_cases.xlsx uploaded.")

st.markdown("---")

tab_generate, tab_compare, tab_results = st.tabs(["🧪 Generate", "📊 Compare", "📁 Results"])

# ---------- TAB: Generate ----------
with tab_generate:
    st.subheader("Generate AI Test Cases")

    run_gen = st.button("Run Generation", type="primary")
    if run_gen:
        with st.spinner("Generating..."):
            req_only = preprocess.read_requirements()
            ac_only  = preprocess.read_acceptance()
            uc_only  = preprocess.read_use_cases()

            req_name_map = {
                str(r.get("requirement_id","")).strip(): r.get("requirement_name","")
                for _, r in req_only.iterrows()
            }

            consolidated_parts = []
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
            }

            # REQUIREMENTS
            if not req_only.empty and num_req > 0:
                req_df, _, _ = preprocess.load_all()
                rows = []
                for _, r in req_df.iterrows():
                    rid   = _clean_text(r.get("requirement_id",""))
                    rname = _clean_text(r.get("requirement_name",""))
                    rtext = _clean_text(r.get("requirement_text","")) or rname
                    details = _clean_text(r.get("requirement_details",""))
                    ac_list = _ensure_list(r.get("ac_list", []))
                    uc_list = _ensure_list(r.get("uc_list", []))
                    if not (rtext or ac_list or uc_list or details):
                        continue
                    cases = generate_with_gpt(
                        rtext, ac_list, uc_list,
                        num_tests=int(num_req),
                        extra_details=details,
                        output_style=output_style,
                        include_ad_hoc=include_ad_hoc,
                        mix=mix,
                        temperature=float(temperature),
                        seed=(None if int(seed) == 0 else int(seed)),
                        debug_logger=ui_log
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
                            "category": c.get("category","Positive"),
                            "gherkin": c.get("gherkin",""),
                            "type": "Generated from Requirement (per-row)"
                        })
                df_req = pd.DataFrame(rows)
                if not df_req.empty:
                    export_excel(df_req, RESULTS_DIR / "report_from_requirements.xlsx",
                                 meta={**meta_common, "Source": "Requirements"})
                    consolidated_parts.append(df_req)
            elif not req_only.empty and num_req == 0:
                st.warning("Skipping REQUIREMENTS (user chose 0).")

            # ACCEPTANCE
            if not ac_only.empty and num_ac > 0:
                rows = []
                if AC_MODE == "group":
                    for rid, grp in ac_only.groupby("requirement_id"):
                        ac_list = [str(x).strip() for x in grp["ac_text"].astype(str).tolist() if str(x).strip()]
                        dets = [str(x).strip() for x in grp.get("ac_details", []).astype(str).tolist() if str(x).strip()]
                        details = "\n".join(dets) if dets else ""
                        if not (ac_list or details):
                            continue
                        rname = req_name_map.get(str(rid).strip(), "")
                        cases = generate_with_gpt(
                            "", ac_list, [],
                            num_tests=int(num_ac),
                            extra_details=details,
                            output_style=output_style,
                            include_ad_hoc=include_ad_hoc,
                            mix=mix,
                            temperature=float(temperature),
                            seed=(None if int(seed) == 0 else int(seed)),
                            debug_logger=ui_log
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
                                "category": c.get("category","Positive"),
                                "gherkin": c.get("gherkin",""),
                                "type": "Generated from Acceptance (group)"
                            })
                else:
                    for _, r in ac_only.iterrows():
                        rid = str(r.get("requirement_id","")).strip()
                        rname = req_name_map.get(rid, "")
                        ac_text = str(r.get("ac_text","")).strip()
                        details = str(r.get("ac_details","")).strip()
                        if not (ac_text or details):
                            continue
                        cases = generate_with_gpt(
                            "", [ac_text], [],
                            num_tests=int(num_ac),
                            extra_details=details,
                            output_style=output_style,
                            include_ad_hoc=include_ad_hoc,
                            mix=mix,
                            temperature=float(temperature),
                            seed=(None if int(seed) == 0 else int(seed)),
                            debug_logger=ui_log
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
                                "category": c.get("category","Positive"),
                                "gherkin": c.get("gherkin",""),
                                "type": "Generated from Acceptance (per-row)"
                            })
                df_ac = pd.DataFrame(rows)
                if not df_ac.empty:
                    export_excel(df_ac, RESULTS_DIR / "report_from_acceptance.xlsx",
                                 meta={**meta_common, "Source": "Acceptance"})
                    consolidated_parts.append(df_ac)
            elif not ac_only.empty and num_ac == 0:
                st.warning("Skipping ACCEPTANCE (user chose 0).")

            # USE CASES
            if not uc_only.empty and num_uc > 0:
                rows = []
                if UC_MODE == "group":
                    for rid, grp in uc_only.groupby("requirement_id"):
                        uc_list = [str(x).strip() for x in grp["uc_text"].astype(str).tolist() if str(x).strip()]
                        dets = [str(x).strip() for x in grp.get("uc_details", []).astype(str).tolist() if str(x).strip()]
                        details = "\n".join(dets) if dets else ""
                        if not (uc_list or details):
                            continue
                        rname = req_name_map.get(str(rid).strip(), "")
                        cases = generate_with_gpt(
                            "", [], uc_list,
                            num_tests=int(num_uc),
                            extra_details=details,
                            output_style=output_style,
                            include_ad_hoc=include_ad_hoc,
                            mix=mix,
                            temperature=float(temperature),
                            seed=(None if int(seed) == 0 else int(seed)),
                            debug_logger=ui_log
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
                                "category": c.get("category","Positive"),
                                "gherkin": c.get("gherkin",""),
                                "type": "Generated from Use Case (group)"
                            })
                else:
                    for _, r in uc_only.iterrows():
                        rid = str(r.get("requirement_id","")).strip()
                        rname = req_name_map.get(rid, "")
                        uc_text = str(r.get("uc_text","")).strip()
                        details = str(r.get("uc_details","")).strip()
                        if not (uc_text or details):
                            continue
                        cases = generate_with_gpt(
                            "", [], [uc_text],
                            num_tests=int(num_uc),
                            extra_details=details,
                            output_style=output_style,
                            include_ad_hoc=include_ad_hoc,
                            mix=mix,
                            temperature=float(temperature),
                            seed=(None if int(seed) == 0 else int(seed)),
                            debug_logger=ui_log
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
                                "category": c.get("category","Positive"),
                                "gherkin": c.get("gherkin",""),
                                "type": "Generated from Use Case (per-row)"
                            })
                df_uc = pd.DataFrame(rows)
                if not df_uc.empty:
                    export_excel(df_uc, RESULTS_DIR / "report_from_use_cases.xlsx",
                                 meta={**meta_common, "Source": "Use Cases"})
                    consolidated_parts.append(df_uc)
            elif not uc_only.empty and num_uc == 0:
                st.warning("Skipping USE CASES (user chose 0).")

            # CONSOLIDATED (Path fix)
            if consolidated_parts:
                consolidated = pd.concat(consolidated_parts, ignore_index=True)
                final_path = Path(DATA_FILES["report"])  # e.g. "results/report.xlsx"
                export_excel(consolidated, final_path,
                             meta={**meta_common, "Source": "Consolidated",
                                   "Generated cases": len(consolidated)})
                st.success("Generation done. See the Results tab to download files.")
            else:
                st.warning("No valid input found, or all counts were 0.")

# ---------- TAB: Compare ----------
with tab_compare:
    st.subheader("Compare AI vs Manual baseline")
    do_compare = st.button("Run Comparison")
    if do_compare:
        with st.spinner("Comparing..."):
            ai_path = RESULTS_DIR / "report.xlsx"
            manual_path = BASE / DATA_FILES["manual"]
            if not ai_path.exists():
                st.error("Missing results/report.xlsx. Please run Generation first.")
            elif not manual_path.exists():
                st.error("Missing data/manual_cases.xlsx. Please upload/provide a manual baseline.")
            else:
                try:
                    df_ai = pd.read_excel(ai_path, sheet_name="generated_raw").fillna("")
                except Exception:
                    st.error("Could not read 'generated_raw' from results/report.xlsx.")
                    df_ai = pd.DataFrame()

                try:
                    df_manual = pd.read_excel(manual_path).fillna("")
                except Exception:
                    st.error("Could not read data/manual_cases.xlsx.")
                    df_manual = pd.DataFrame()

                if df_ai.empty or df_manual.empty:
                    st.warning("AI or Manual data is empty. Check files.")
                else:
                    res = run_comparison(
                        df_ai=df_ai,
                        df_manual=df_manual,
                        strategy=similarity_strategy,
                        threshold=similarity_threshold
                    )
                    out_cmp = RESULTS_DIR / "report_comparison.xlsx"
                    export_comparison_excel(
                        out_cmp,
                        matches=res["matches"],
                        ai_only=res["ai_only"],
                        manual_only=res["manual_only"],
                        scores_summary=res["scores_summary"],
                        dist_by_category=res["dist_by_category"],
                        per_req_density=res["per_requirement_density"],
                        trace_matrix=res["trace_matrix"],
                        run_info={
                            "strategy": similarity_strategy,
                            "threshold": similarity_threshold
                        }
                    )
                    st.success("Comparison finished.")
                    st.caption("Scores summary")
                    st.dataframe(res["scores_summary"])
                    st.caption("Category distribution")
                    st.dataframe(res["dist_by_category"])

# ---------- TAB: Results ----------
with tab_results:
    st.subheader("Download outputs")
    files = [
        RESULTS_DIR / "report_from_requirements.xlsx",
        RESULTS_DIR / "report_from_acceptance.xlsx",
        RESULTS_DIR / "report_from_use_cases.xlsx",
        RESULTS_DIR / "report.xlsx",
        RESULTS_DIR / "report_comparison.xlsx",
    ]
    for f in files:
        if f.exists():
            st.write(f"**{f.name}**")
            st.download_button(
                label="Download",
                data=bytes_of_file(f),
                file_name=f.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.write(f"_{f.name} (not generated yet)_")
