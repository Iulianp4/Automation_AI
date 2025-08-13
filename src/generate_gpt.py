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

def build_prompt(story: str, ac_list: List[str]) -> str:
    ac_lines = "\n".join(f"- {a}" for a in ac_list)
    return f"""
Genereaza cazuri de test UI (pozitive si negative) pentru cerinta de mai jos.
Respecta STRICT urmatorul format pentru FIECARE caz de test, fara tex­te suplimentare:
Titlu:
Preconditii:
Pasi:
Date:
Rezultat asteptat:

User story:
{story}

Acceptance criteria:
{ac_lines}

Note:
- Scrie pasi numerotati si clari (imperativ).
- Include cel putin 1 test pozitiv si 1 test negativ.
- Fara explicatii in afara sectiunilor cerute.
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

def generate_with_gpt(story: str, ac_list: List[str]) -> List[Dict]:
    prompt = build_prompt(story, ac_list)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    return parse_generated_text(text)
