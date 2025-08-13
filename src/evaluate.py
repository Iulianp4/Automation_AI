from typing import List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

# model mic, rapid; se descarcă la prima rulare
_model = None
def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def embed(texts: List[str]) -> np.ndarray:
    model = _get_model()
    return np.array(model.encode(texts, normalize_embeddings=True))

def cosine_sim(a: str, b: str) -> float:
    E = embed([a, b])
    v1, v2 = E[0], E[1]
    return float(np.clip(np.dot(v1, v2), -1.0, 1.0))

def coverage_score(ac_list: List[str], steps_or_expected: str) -> float:
    """
    scor simplu: fractioneaza criteriile; considera acoperit daca
    cel putin 6+ caractere din fiecare AC apar in text (insensibil la caps).
    """
    if not ac_list:
        return 0.0
    text = (steps_or_expected or "").lower()
    covered = 0
    for ac in ac_list:
        s = (ac or "").lower().strip()
        if len(s) < 6:
            continue
        # ia o bucata semnificativa (primele 10-14 caractere fara spatii multiple)
        key = " ".join(s.split())[:14]
        if key and key in text:
            covered += 1
    return covered / max(1, len(ac_list))

def map_generated_to_manual(gen_df: pd.DataFrame, man_df: pd.DataFrame, criteria_df: pd.DataFrame) -> pd.DataFrame:
    """
    face mapare pe story_id; pentru fiecare TC generat, cauta cel mai similar manual.
    """
    rows = []
    for sid, part in gen_df.groupby("story_id"):
        man_part = man_df[man_df["story_id"] == sid]
        ac_list = criteria_df[criteria_df["story_id"] == sid]["ac_text"].dropna().astype(str).tolist()
        for _, g in part.iterrows():
            best = None
            best_score = -1.0
            for _, m in man_part.iterrows():
                # similaritate compozita
                s_title = cosine_sim(str(g["title"]), str(m["title"]))
                s_steps = cosine_sim(str(g["steps"]), str(m["steps"]))
                s_exp   = cosine_sim(str(g["expected"]), str(m["expected"]))
                sim_avg = (s_title + s_steps + s_exp) / 3.0
                if sim_avg > best_score:
                    best_score = sim_avg
                    best = (m.get("tc_id", ""), s_title, s_steps, s_exp, sim_avg)
            cov = coverage_score(ac_list, f"{g.get('steps','')}\n{g.get('expected','')}")
            rows.append({
                "story_id": sid,
                "generated_tc_id": g.get("tc_id",""),
                "manual_tc_id": best[0] if best else "",
                "similarity_title": round(best[1], 4) if best else 0.0,
                "similarity_steps": round(best[2], 4) if best else 0.0,
                "similarity_expected": round(best[3], 4) if best else 0.0,
                "match_score": round(best[4], 4) if best else 0.0,
                "coverage": round(cov, 4),
            })
    return pd.DataFrame(rows)

def aggregate_metrics(mapping_df: pd.DataFrame) -> pd.DataFrame:
    if mapping_df.empty:
        return pd.DataFrame([{"metric":"samples","value":0,"notes":"no data"}])
    return pd.DataFrame([
        {"metric":"samples", "value": len(mapping_df), "notes": ""},
        {"metric":"avg_match_score", "value": round(mapping_df["match_score"].mean(),4), "notes":"mean(title,steps,expected)"},
        {"metric":"avg_coverage", "value": round(mapping_df["coverage"].mean(),4), "notes":"AC text overlap heuristic"},
    ])
