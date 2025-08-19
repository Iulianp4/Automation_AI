from typing import Dict, List
import os, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# We parse blocks with required fields + optional Category + optional Gherkin.
# Order is enforced in the prompt; parser stays robust if blank lines appear.
SECTION_PATTERN = re.compile(
    r"Title:\s*(?P<title>.+?)\s*"
    r"Preconditions:\s*(?P<pre>.+?)\s*"
    r"Steps:\s*(?P<steps>.+?)\s*"
    r"Test Data:\s*(?P<data>.+?)\s*"
    r"Expected Result:\s*(?P<expected>.+?)\s*"
    r"(?:Category:\s*(?P<category>.+?)\s*)?"
    r"(?:Gherkin:\s*(?P<gherkin>.+?))?(?:\n{2,}|\Z)",
    re.S | re.I
)

def _lines(items) -> str:
    """Render list-like content as bullets or '- (none)'."""
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

def build_prompt(
    requirement_text: str,
    ac_list,
    uc_list,
    num_tests: int,
    extra_details: str = "",
    output_style: str = "both",         # "classic" | "gherkin" | "both"
    include_ad_hoc: bool = True,        # allow AdHoc category based on tester context
    mix: str = "balanced"               # "balanced" | "positive_heavy" | "negative_heavy"
) -> str:
    """
    Builds a strict instruction prompt:
    - Always return EXACTLY num_tests cases
    - Each case includes Category (Positive/Negative/Boundary/Security/AdHoc)
    - Optionally include a Gherkin scenario
    """
    ac_lines = _lines(ac_list)
    uc_lines = _lines(uc_list)
    req_text = (requirement_text or "").strip()
    details  = (extra_details or "").strip()

    details_block = f"\n\nAdditional tester context:\n{details}" if details else ""

    style_rules = []
    if output_style in ("classic", "both"):
        style_rules.append("Include the detailed classic fields.")
    if output_style in ("gherkin", "both"):
        style_rules.append("Also include a Gherkin scenario (Given/When/Then) that verifies the same behavior.")
    style_note = "\n- ".join(style_rules) if style_rules else "Include the detailed classic fields."

    adhoc_note = "AdHoc category is allowed if tester context suggests extra edge/business scenarios." if include_ad_hoc else "Do NOT use AdHoc category."

    mix_note = {
        "balanced": (
            "Across the set: include at least 1 Negative; where inputs/limits exist include at least 1 Boundary; "
            "include 1 Security if relevant (auth, roles, PII, rate-limit). Fill the rest with Positive."
        ),
        "positive_heavy": "Prefer Positive tests; include at least 1 Negative; Boundary/Security only if clearly relevant.",
        "negative_heavy": "Prefer Negative tests; include at least 1 Positive; add Boundary/Security where relevant."
    }.get(mix, "Prefer a balanced mix.")

    return (
        f"Generate EXACTLY {num_tests} UI test cases for the feature/context below.\n"
        f"- {style_note}\n"
        f"- Assign a Category to each case: Positive | Negative | Boundary | Security"
        f"{' | AdHoc' if include_ad_hoc else ''}.\n"
        f"- {mix_note}\n"
        f"- Steps MUST be imperative, numbered lines. No vague language ('maybe', 'etc').\n"
        f"- Keep outputs strictly in the specified fields. No extra commentary.\n\n"
        "Source material follows:\n"
        f"Requirement (if any):\n{req_text}\n\n"
        f"Acceptance criteria (if any):\n{ac_lines}\n\n"
        f"Use cases (if any):\n{uc_lines}"
        f"{details_block}\n\n"
        "=== OUTPUT FORMAT (repeat per test case) ===\n"
        "Title:\n"
        "Preconditions:\n"
        "Steps:\n"
        "Test Data:\n"
        "Expected Result:\n"
        "Category:\n"
        + ("Gherkin:\n" if output_style in ("gherkin", "both") else "")
    )

def _normalize_steps(text: str) -> str:
    """Turn '1. ...' into markdown bullets '- ...' per line."""
    return re.sub(r"^\s*\d+\.\s*", "- ", text.strip(), flags=re.M)

def parse_generated_text(text: str) -> List[Dict]:
    cases = []
    if not text:
        return cases
    for m in SECTION_PATTERN.finditer(text.strip()):
        steps = _normalize_steps(m.group("steps"))
        case = {
            "title": m.group("title").strip(),
            "preconditions": m.group("pre").strip(),
            "steps": steps.strip(),
            "data": m.group("data").strip(),
            "expected": m.group("expected").strip(),
            "category": (m.group("category") or "Positive").strip(),
            "gherkin": (m.group("gherkin") or "").strip(),
        }
        cases.append(case)
    return cases

def generate_with_gpt(
    requirement_text: str,
    ac_list,
    uc_list,
    num_tests: int,
    extra_details: str = "",
    output_style: str = "both",
    include_ad_hoc: bool = True,
    mix: str = "balanced"
) -> List[Dict]:
    prompt = build_prompt(
        requirement_text, ac_list, uc_list, num_tests,
        extra_details=extra_details,
        output_style=output_style,
        include_ad_hoc=include_ad_hoc,
        mix=mix
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content
    return parse_generated_text(text)
