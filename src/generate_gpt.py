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

def build_prompt(requirement_text: str, ac_list: List[str], uc_list: List[str]) -> str:
    ac_lines = "\n".join(f"- {a}" for a in ac_list) if ac_list else "- (none)"
    uc_lines = "\n".join(f"- {u}" for u in uc_list) if uc_list else "- (none)"
    return f"""
Genereaza MAXIM 4 cazuri de test UI (cel putin 1 negativ) pentru cerinta de mai jos.
Respecta STRICT formatul pentru FIECARE caz:
Titlu:
Preconditii:
Pasi:
Date:
Rezultat asteptat:

Reguli:
- Pasi numerotati pe linii separate (imperativ).
- Fara text in afara sectiunilor cerute.
- Evita formulări vagi ("etc.", "maybe").

Requirement:
{requirement_text}

Acceptance criteria:
{ac_lines}

Use cases:
{uc_lines}
"""

def parse_generated_text(text: str) -> List[Dict]:
    cases = []
    for m in SECTION_PATTERN.finditer(text.strip()):
        steps = re.sub(r"^\s*\d+\.\s*", "- ", m.group("steps").strip(), flags=re.M)
        cases.append({
            "title": m.group("title").strip(),
            "preconditions": m.group("pre").strip(),
            "steps": steps.strip(),
            "data": m.group("data").strip(),
            "expected": m.group("expected").strip(),
        })
    return cases

def generate_with_gpt(requirement_text: str, ac_list: List[str], uc_list: List[str]) -> List[Dict]:
    prompt = build_prompt(requirement_text, ac_list, uc_list)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    return parse_generated_text(text)
