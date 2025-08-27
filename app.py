import os
import platform
import subprocess
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

# ---------- Session defaults & profile autoload ----------
if "settings" not in st.session_state:
    st.session_state["settings"] = {}

# try autoload default profile once
_default_payload = profiles.load_default_if_exists()
if _default_payload:
    st.session_state["settings"].update(_default_payload)

# ---------- Sidebar ----------
st.sidebar.title("Settings")

# Generation knobs (allow 0)
st.sidebar.subheader("Generation")
num_req = st.sidebar.number_input(
    "Tests per REQUIREMENT row",
    min_value=0, max_value=50,
    value=int(st.session_state["settings"].get("num_req", 5)),
    step=1, key="num_req",
    help="Poți pune 0 ca să sari peste Requirements."
)
num_ac  = st.sidebar.number_input(
    f"Tests per ACCEPTANCE ({'GROUP' if AC_MODE=='group' else 'ROW'})",
    min_value=0, max_value=50,
    value=int(st.session_state["settings"].get("num_ac", 5)),
    step=1, key="num_ac",
    help="0 = nu generezi din acceptance_criteria.xlsx."
)
num_uc  = st.sidebar.number_input(
    f"Tests per USE CASE ({'GROUP' if UC_MODE=='group' else 'ROW'})",
    min_value=0, max_value=50,
    value=int(st.session_state["settings"].get("num_uc", 5)),
    step=1, key="num_uc",
    help="0 = nu generezi din use_cases.xlsx."
)

output_style = st.sidebar.selectbox(
    "Output style",
    ["both", "classic", "gherkin"],
    index=["both","classic","gherkin"].index(st.session_state["settings"].get("output_style","both")),
    key="output_style",
    help="classic = doar pași; gherkin = doar Given/When/Then; both = ambele."
)
include_ad_hoc = st.sidebar.checkbox(
    "Allow AdHoc category",
    value=bool(st.session_state["settings"].get("include_ad_hoc", True)),
    key="include_ad_hoc",
    help="Dacă e bifat, generatorul poate include și categoria AdHoc."
)
mix = st.sidebar.selectbox(
    "Test mix",
    ["balanced","positive_heavy","negative_heavy"],
    index=["balanced","positive_heavy","negative_heavy"].index(st.session_state["settings"].get("mix","balanced")),
    key="mix",
    help="balanced = distribuție uniformă; *_heavy = accent pe pozitiv/negativ."
)

# Model controls
st.sidebar.subheader("Model controls")
temperature = st.sidebar.slider(
    "Temperature", 0.0, 1.2,
    float(st.session_state["settings"].get("temperature", 0.2)),
    0.05, key="temperature",
    help="Mai mic = mai determinist; mai mare = mai creativ."
)
seed = st.sidebar.number_input(
    "Random seed (0 = auto)",
    min_value=0, max_value=1_000_000,
    value=int(st.session_state["settings"].get("seed", 0)),
    step=1, key="seed",
    help="Setează un seed >0 pentru rulări reproductibile."
)

# Model selector + rough cost estimate
MODEL_PRICES = {
    # USD per 1M tokens (aprox.)
    "gpt-4o-mini": {"in": 0.150, "out": 0.600},
    "gpt-4.1":     {"in": 5.000, "out": 15.000},
}
model_choices = list(MODEL_PRICES.keys())
model_default = st.session_state["settings"].get("model_name", "gpt-4o-mini")
if model_default not in model_choices:
    model_default = model_choices[0]
model_name = st.sidebar.selectbox(
    "Model",
    model_choices,
    index=model_choices.index(model_default),
    key="model_name",
    help="Alege modelul OpenAI folosit pentru generare."
)

# Comparison (renamed + friendlier labels)
st.sidebar.subheader("Comparison")

# map UI labels -> internal strategy
FRIENDLY_STRATEGY = {
    "Title & Expected (simple)": "title_expected",
    "Title + Steps + Expected (detailed)": "title_steps_expected",
}
# find default index by internal value
_internal_default = st.session_state["settings"].get("similarity_strategy", "title_expected")
_default_label = [k for k, v in FRIENDLY_STRATEGY.items() if v == _internal_default]
if not _default_label:
    _default_label = ["Title & Expected (simple)"]
selected_label = st.sidebar.selectbox(
    "Matching method",
    list(FRIENDLY_STRATEGY.keys()),
    index=list(FRIENDLY_STRATEGY.keys()).index(_default_label[0]),
    help=(
        "Cum comparăm testele AI cu baseline-ul manual:\n"
        "• simple = doar Titlu + Expected\n"
        "• detailed = Titlu + Steps + Expected (mai strict, dar mai scump)"
    )
)
# store internal value in session
st.session_state["similarity_strategy"] = FRIENDLY_STRATEGY[selected_label]

similarity_threshold = st.sidebar.slider(
    "Match threshold",
    0.50, 0.95,
    float(st.session_state["settings"].get("similarity_threshold", 0.75)),
    0.01, key="similarity_threshold",
    help="Prag de potrivire (0.50 tolerant … 0.95 foarte strict)."
)

# Data source
st.sidebar.subheader("Data source")
use_uploaded = st.sidebar.radio(
    "Use data from",
    ["Existing /data", "Upload now"],
    index=0,
    help="Poți încărca fișierele acum sau folosi ce e deja în /data."
)

# Profiles
st.sidebar.subheader("Profiles")
with st.sidebar.expander("Save / Load profiles", expanded=False):
    current_settings = {
        "num_req": int(st.session_state["num_req"]),
        "num_ac": int(st.session_state["num_ac"]),
        "num_uc": int(st.session_state["num_uc"]),
        "output_style": st.session_state["output_style"],
        "include_ad_hoc": bool(st.session_state["include_ad_hoc"]),
        "mix": st.session_state["mix"],
        "temperature": float(st.session_state["temperature"]),
        "seed": int(st.session_state["seed"]),
        "model_name": st.session_state["model_name"],
        "similarity_strategy": st.session_state["similarity_strategy"],
        "similarity_threshold": float(st.session_state["similarity_threshold"]),
    }
    prof_name = st.text_input("Profile name", value="my_profile", key="prof_name", help="Un nume scurt pentru setul de configurări.")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        if st.button("Save profile", help="Salvează toate controalele din stânga sub acest nume."):
            profiles.save_profile(st.session_state["prof_name"], current_settings)
            st.success(f"Saved profile: {st.session_state['prof_name']}")
    with col_p2:
        options = profiles.list_profiles()
        pick = st.selectbox("Load profile", options, index=0 if options else None, help="Alege un profil existent.")
    with col_p3:
        if st.button("Load", help="Aplică profilul selectat mai sus."):
            if pick:
                payload = profiles.load_profile(pick)
                if payload:
                    st.session_state["settings"].update(payload)
                    st.success("Profile loaded → applying controls…")
                    st.rerun()
    with col_p4:
        if st.button("Set as default", help="Acest profil va fi încărcat automat la pornire."):
            try:
                profiles.set_default_profile(st.session_state["prof_name"])
                st.success(f"'{st.session_state['prof_name']}' set as default")
            except Exception as e:
                st.error(str(e))

with st.sidebar.expander("Cost estimate (rough)", expanded=False):
    # build a quick context proxy from currently loaded data
    req_preview = preprocess.read_requirements().head(5)
    ac_preview  = preprocess.read_acceptance().head(5)
    uc_preview  = preprocess.read_use_cases().head(5)

    def rough_token_estimate(texts: list[str]) -> int:
        # very rough: ~4 chars per token
        chars = sum(len(str(t)) for t in texts)
        return max(1, chars // 4)

    ctx_texts = []
    ctx_texts += req_preview.get("requirement_text", pd.Series([], dtype=str)).astype(str).tolist()
    ctx_texts += ac_preview.get("ac_text", pd.Series([], dtype=str)).astype(str).tolist()
    ctx_texts += uc_preview.get("uc_text", pd.Series([], dtype=str)).astype(str).tolist()

    in_tokens = rough_token_estimate(ctx_texts)
    t_req = min(10, len(req_preview)) * int(st.session_state["num_req"])
    t_ac  = min(10, len(ac_preview))  * int(st.session_state["num_ac"])
    t_uc  = min(10, len(uc_preview))  * int(st.session_state["num_uc"])
    total_tests = t_req + t_ac + t_uc

    # heuristically: ~200 output tokens per test
    out_tokens = max(0, total_tests * 200)

    prices = MODEL_PRICES.get(st.session_state["model_name"], {"in": 0.0, "out": 0.0})
    usd = (in_tokens / 1_000_000) * prices["in"] + (out_tokens / 1_000_000) * prices["out"]

    st.write(f"Input tokens (rough): **{in_tokens:,}**")
    st.write(f"Output tokens (rough): **{out_tokens:,}**")
    st.write(f"Estimated cost: **${usd:.4f}**")
    st.caption("Estimate is approximate and may differ from actual billing.")

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
    st.caption("Tip: orice control numeric poate fi 0 ca să sari peste sursa respectivă.")

    # --- Pre-flight validation boxes (vizibile)
    req_only = preprocess.read_requirements()
    ac_only  = preprocess.read_acceptance()
    uc_only  = preprocess.read_use_cases()

    v_req = preprocess.validate_columns(req_only, ["requirement_id","requirement_name","requirement_text"], "requirements.xlsx")
    v_ac  = preprocess.validate_columns(ac_only,  ["requirement_id","ac_text"], "acceptance_criteria.xlsx")
    v_uc  = preprocess.validate_columns(uc_only,  ["requirement_id","uc_text"], "use_cases.xlsx")

    colv1, colv2, colv3 = st.columns(3)
    for v, col in [(v_req,colv1),(v_ac,colv2),(v_uc,colv3)]:
        with col:
            title = f"**{v['label']}** ({v['rows']} rows)"
            if v["ok"]:
                st.success(title)
            else:
                st.error(title)
                if v["missing"]:
                    st.caption("Missing columns:")
                    st.code(", ".join(v["missing"]))
                if v["present"]:
                    st.caption("Present:")
                    st.code(", ".join(v["present"]))

    run_gen = st.button("Run Generation", type="primary")
    if run_gen:
        with st.spinner("Generating..."):
            req_name_map = {
                str(r.get("requirement_id","")).strip(): r.get("requirement_name","")
                for _, r in req_only.iterrows()
            }

            consolidated_parts = []
            meta_common = {
                "AC mode": AC_MODE,
                "UC mode": UC_MODE,
                "Req tests per item": int(st.session_state["num_req"]),
                "AC tests per item": int(st.session_state["num_ac"]),
                "UC tests per item": int(st.session_state["num_uc"]),
                "Output style": st.session_state["output_style"],
                "AdHoc allowed": bool(st.session_state["include_ad_hoc"]),
                "Mix": st.session_state["mix"],
                "Temperature": float(st.session_state["temperature"]),
                "Seed": int(st.session_state["seed"]),
                "Model": st.session_state["model_name"],
            }

            # REQUIREMENTS
            if not req_only.empty and int(st.session_state["num_req"]) > 0:
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
                        num_tests=int(st.session_state["num_req"]),
                        extra_details=details,
                        output_style=st.session_state["output_style"],
                        include_ad_hoc=bool(st.session_state["include_ad_hoc"]),
                        mix=st.session_state["mix"],
                        temperature=float(st.session_state["temperature"]),
                        seed=(None if int(st.session_state["seed"]) == 0 else int(st.session_state["seed"])),
                        model=st.session_state["model_name"],
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
            elif not req_only.empty and int(st.session_state["num_req"]) == 0:
                st.warning("Skipping REQUIREMENTS (user chose 0).")

            # ACCEPTANCE
            if not ac_only.empty and int(st.session_state["num_ac"]) > 0:
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
                            num_tests=int(st.session_state["num_ac"]),
                            extra_details=details,
                            output_style=st.session_state["output_style"],
                            include_ad_hoc=bool(st.session_state["include_ad_hoc"]),
                            mix=st.session_state["mix"],
                            temperature=float(st.session_state["temperature"]),
                            seed=(None if int(st.session_state["seed"]) == 0 else int(st.session_state["seed"])),
                            model=st.session_state["model_name"],
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
                            num_tests=int(st.session_state["num_ac"]),
                            extra_details=details,
                            output_style=st.session_state["output_style"],
                            include_ad_hoc=bool(st.session_state["include_ad_hoc"]),
                            mix=st.session_state["mix"],
                            temperature=float(st.session_state["temperature"]),
                            seed=(None if int(st.session_state["seed"]) == 0 else int(st.session_state["seed"])),
                            model=st.session_state["model_name"],
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
            elif not ac_only.empty and int(st.session_state["num_ac"]) == 0:
                st.warning("Skipping ACCEPTANCE (user chose 0).")

            # USE CASES
            if not uc_only.empty and int(st.session_state["num_uc"]) > 0:
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
                            num_tests=int(st.session_state["num_uc"]),
                            extra_details=details,
                            output_style=st.session_state["output_style"],
                            include_ad_hoc=bool(st.session_state["include_ad_hoc"]),
                            mix=st.session_state["mix"],
                            temperature=float(st.session_state["temperature"]),
                            seed=(None if int(st.session_state["seed"]) == 0 else int(st.session_state["seed"])),
                            model=st.session_state["model_name"],
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
                            num_tests=int(st.session_state["num_uc"]),
                            extra_details=details,
                            output_style=st.session_state["output_style"],
                            include_ad_hoc=bool(st.session_state["include_ad_hoc"]),
                            mix=st.session_state["mix"],
                            temperature=float(st.session_state["temperature"]),
                            seed=(None if int(st.session_state["seed"]) == 0 else int(st.session_state["seed"])),
                            model=st.session_state["model_name"],
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
            elif not uc_only.empty and int(st.session_state["num_uc"]) == 0:
                st.warning("Skipping USE CASES (user chose 0).")

            # CONSOLIDATED
            if consolidated_parts:
                consolidated = pd.concat(consolidated_parts, ignore_index=True)
                final_path = Path(DATA_FILES["report"])  # e.g. "results/report.xlsx"
                export_excel(
                    consolidated, final_path,
                    meta={**meta_common, "Source": "Consolidated",
                          "Generated cases": len(consolidated)}
                )
                st.success("Generation done. See the Results tab to download files.")
            else:
                st.warning("No valid input found, or all counts were 0.")

# ---------- TAB: Compare ----------
with tab_compare:
    st.subheader("Compare AI vs Manual baseline")
    st.caption(
        f"Matching method: **{selected_label}** · Threshold: **{st.session_state['similarity_threshold']:.2f}**"
    )
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
                        strategy=st.session_state["similarity_strategy"],
                        threshold=float(st.session_state["similarity_threshold"])
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
                            "strategy": st.session_state["similarity_strategy"],
                            "threshold": float(st.session_state["similarity_threshold"])
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

    # open folder button + path
    st.write(f"Results path: `{RESULTS_DIR}`")
    if st.button("Open results folder"):
        try:
            sysname = platform.system()
            if sysname == "Windows":
                os.startfile(str(RESULTS_DIR))  # type: ignore[attr-defined]
            elif sysname == "Darwin":
                subprocess.Popen(["open", str(RESULTS_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(RESULTS_DIR)])
        except Exception as e:
            st.warning(f"Could not open folder: {e}")

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
