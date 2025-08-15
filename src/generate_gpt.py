from typing import Dict, List
import os, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SECTION_PATTERN = re.compile(
    r"Titlu:\s*(?P<title>.+?)\s*"
    r"Preconditii:\s*(?P<pre>.+?)\s*"
    r"Pasi:\s*(?P<steps>.+?)\s*"
    r"Date:\s*(?P<data>.+?)\s*"
    r"Rezultat asteptat:\s*(?P<expected>.+?)(?:\n{2,}|\Z)",
    re.S | re.I
)

def _lines(items) -> str:
    """transformă None/NaN/str/list în listă de bullet-uri sau '- (none)'"""
    try:
        import pandas as pd
        if items is None or (isinstance(items, float) and pd.isna(items)):
            return "- (none)"
    except Exception:
        pass
    if isinstance(items, (str, bytes)):
        items = [items.decode() if isinstance(items, bytes) else items]
    if not isinstance(items, list):
        return "- (none)"
    items = [str(x).strip() for x in items if str(x).strip()]
    return "\n".join(f"- {x}" for x in items) if items else "- (none)"

def build_prompt(requirement_text: str, ac_list, uc_list, num_tests: int, extra_details: str = "") -> str:
    ac_lines = _lines(ac_list)
    uc_lines = _lines(uc_list)
    req_text = (requirement_text or "").strip()
    details = (extra_details or "").strip()

    details_block = f"\n\nDetalii suplimentare (pentru context):\n{details}" if details else ""

    return (
        f"Genereaza EXACT {num_tests} cazuri de test UI (include cel putin 1 negativ) pentru cerinta/contextul de mai jos.\n"
        "Respecta STRICT formatul pentru FIECARE caz:\n"
        "Titlu:\nPreconditii:\nPasi:\nDate:\nRezultat asteptat:\n\n"
        "Reguli:\n"
        "- Pasi numerotati pe linii separate (imperativ: 'Click', 'Introdu', 'Verifica').\n"
        "- Fara text in afara sectiunilor cerute.\n"
        "- Evita formulari vagi (\"etc.\", \"maybe\").\n\n"
        f"Requirement (daca exista):\n{req_text}\n\n"
        f"Acceptance criteria (daca exista):\n{ac_lines}\n\n"
        f"Use cases (daca exista):\n{uc_lines}"
        f"{details_block}\n"
    )

def parse_generated_text(text: str) -> List[Dict]:
    cases = []
    if not text:
        return cases
    for m in SECTION_PATTERN.finditer(text.strip()):
        import re as _re
        steps = _re.sub(r"^\s*\d+\.\s*", "- ", m.group("steps").strip(), flags=_re.M)
        cases.append({
            "title": m.group("title").strip(),
            "preconditions": m.group("pre").strip(),
            "steps": steps.strip(),
            "data": m.group("data").strip(),
            "expected": m.group("expected").strip(),
        })
    return cases

def generate_with_gpt(requirement_text: str, ac_list, uc_list, num_tests: int, extra_details: str = "") -> List[Dict]:
    prompt = build_prompt(requirement_text, ac_list, uc_list, num_tests, extra_details)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    return parse_generated_text(text)
