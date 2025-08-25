# src/generate_gpt.py
from __future__ import annotations
import os
import time
import json
import re
from typing import Any, List, Dict

from openai import OpenAI
from dotenv import load_dotenv

# local cache (JSON on disk): src/cache.py
from src import cache

load_dotenv()

# -----------------------------
# Mini logger helper
# -----------------------------
def _log(debug_logger, msg: str):
    """Log în Streamlit dacă e dat, altfel în stdout dacă AIAI_VERBOSE=1."""
    try:
        if debug_logger is not None:
            debug_logger(msg)
        elif os.getenv("AIAI_VERBOSE", "0") == "1":
            print(msg)
    except Exception:
        pass

# -----------------------------
# Model / client
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Category mixing
# -----------------------------
BASE_CATEGORIES = ["Positive", "Negative", "Boundary", "Security"]
ALL_CATEGORIES  = BASE_CATEGORIES + ["AdHoc"]

def _distribute_categories(n: int, mix: str = "balanced", allow_adhoc: bool = True) -> List[str]:
    """
    Return a list of length n with target categories per test.
    mix: balanced | positive_heavy | negative_heavy
    """
    cats = BASE_CATEGORIES.copy()
    if allow_adhoc:
        cats = ALL_CATEGORIES.copy()

    if n <= 0:
        return []

    weights = {c: 1.0 for c in cats}
    if mix == "positive_heavy" and "Positive" in weights:
        weights["Positive"] = 2.0
    elif mix == "negative_heavy" and "Negative" in weights:
        weights["Negative"] = 2.0

    # expand by weights then slice in a pseudo-shuffled order
    bag: List[str] = []
    for c, w in weights.items():
        bag += [c] * int(max(1, round(w * 10)))

    out: List[str] = []
    idx = 0
    for _ in range(n):
        out.append(bag[idx % len(bag)])
        idx += 7  # pseudo-shuffle step
    return out[:n]

# -----------------------------
# Prompt builder
# -----------------------------
def _mk_context(requirement_text: str, ac_list: List[str], uc_list: List[str], extra_details: str) -> str:
    parts: List[str] = []
    if requirement_text.strip():
        parts.append(f"REQUIREMENT:\n{requirement_text}")
    if ac_list:
        joined = "\n- ".join([a for a in ac_list if str(a).strip()])
        if joined.strip():
            parts.append(f"ACCEPTANCE CRITERIA:\n- {joined}")
    if uc_list:
        joined = "\n- ".join([u for u in uc_list if str(u).strip()])
        if joined.strip():
            parts.append(f"USE CASES:\n- {joined}")
    if str(extra_details or "").strip():
        parts.append(f"TESTER DETAILS (extra context):\n{extra_details}")
    return "\n\n".join(parts).strip()

def _mk_schema_instruction(output_style: str, categories: List[str]) -> str:
    """
    Constrângeri clare de ieșire.
    """
    style_hint = {
        "classic": "Use classic step-by-step style. Gherkin must be empty string.",
        "gherkin": "Use Given/When/Then. Steps can be empty, but Gherkin must be present.",
        "both": "Provide both: classic steps AND a Gherkin variant."
    }.get(output_style, "Provide both: classic steps AND a Gherkin variant.")

    allowed = ", ".join(ALL_CATEGORIES)
    return f"""
You MUST output ONLY a JSON array. No prose, no code fences.
Each element must be an object with EXACTLY the fields:
- "title": string (short, imperative)
- "preconditions": string (can be empty)
- "steps": string (numbered or bullet steps; can be empty if Gherkin-only)
- "data": string (inputs/fixtures; can be empty)
- "expected": string (clear, verifiable outcome)
- "category": one of [{allowed}]
- "gherkin": string (Given/When/Then block; can be empty per style)

Style requirement: {style_hint}
Match the requested categories in order: {categories}.
If content does not allow a category, adapt reasonably but still output the requested number of items.
"""

def build_prompt(
    requirement_text: str,
    ac_list: List[str],
    uc_list: List[str],
    num_tests: int,
    extra_details: str,
    output_style: str,
    categories: List[str]
) -> List[Dict[str, str]]:
    ctx = _mk_context(requirement_text, ac_list, uc_list, extra_details)
    schema = _mk_schema_instruction(output_style, categories)

    user = f"""
Generate {num_tests} UI test cases that cover the context below.
Focus on realistic web UI actions (click, type, select, navigate) and verifiable expected results.
Avoid duplicates; diversify flows (happy path, negative, boundary, security, ad-hoc if allowed).
Ensure the array length matches exactly {num_tests}.

CONTEXT:
{ctx}
""".strip()

    system = f"""
You are a senior QA automation engineer. Produce only a pure JSON array with {num_tests} objects.
Do not include explanations. No markdown. No extra keys. No trailing commas.
{schema}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

# -----------------------------
# Robust JSON extractor
# -----------------------------
_JSON_ARRAY_RE = re.compile(r"\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\s*\]", re.DOTALL)

def _clean_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _json_load_attempts(txt: str) -> Any:
    try:
        return json.loads(txt)
    except Exception:
        pass
    try:
        fixed = re.sub(r"(?<!\\)'", '"', txt)
        return json.loads(fixed)
    except Exception:
        pass
    try:
        no_trailing = re.sub(r",\s*([}\]])", r"\1", txt)
        return json.loads(no_trailing)
    except Exception:
        raise

def _extract_json_array(raw_text: str) -> List[dict]:
    if not isinstance(raw_text, str):
        raise ValueError("Model returned non-text content.")
    s = raw_text.strip()

    # 1) direct
    try:
        obj = _json_load_attempts(s)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            return obj["items"]
    except Exception:
        pass

    # 2) code fences
    try:
        cleaned = _clean_fences(s)
        obj = _json_load_attempts(cleaned)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            return obj["items"]
    except Exception:
        pass

    # 3) first array via regex
    m = _JSON_ARRAY_RE.search(s)
    if m:
        block = _clean_fences(m.group(0))
        obj = _json_load_attempts(block)
        if isinstance(obj, list):
            return obj

    raise ValueError("Model did not return valid JSON array.")

# -----------------------------
# Normalize output rows
# -----------------------------
def _norm_str(x: Any) -> str:
    return str(x or "").replace("\u00A0", " ").strip()

def _ensure_row_fields(o: dict) -> dict:
    return {
        "title": _norm_str(o.get("title", "")),
        "preconditions": _norm_str(o.get("preconditions", "")),
        "steps": _norm_str(o.get("steps", "")),
        "data": _norm_str(o.get("data", "")),
        "expected": _norm_str(o.get("expected", "")),
        "category": _norm_str(o.get("category", "Positive")) or "Positive",
        "gherkin": _norm_str(o.get("gherkin", "")),
    }

# -----------------------------
# Call wrapper with retry
# -----------------------------
def _openai_chat(
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    timeout_s: int,
    seed: int | None,
    retries: int = 2,
    debug_logger=None
):
    """
    Small wrapper with simple backoff retry.
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            _log(debug_logger, f"OpenAI call attempt {attempt+1}/{retries+1}")
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                seed=seed,           # reproducibility if provided (OpenAI v1 supports it)
                timeout=timeout_s,   # per-request timeout (works with httpx transport)
            )
        except Exception as e:
            _log(debug_logger, f"OpenAI error: {e!s}")
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                _log(debug_logger, "Retrying...")
            else:
                raise last_err

# -----------------------------
# Public API
# -----------------------------
def generate_with_gpt(
    requirement_text: str,
    ac_list: List[str] | None,
    uc_list: List[str] | None,
    num_tests: int = 5,
    extra_details: str = "",
    output_style: str = "both",              # classic | gherkin | both
    include_ad_hoc: bool = True,
    mix: str = "balanced",                    # balanced | positive_heavy | negative_heavy
    temperature: float = 0.2,
    timeout_s: int = 60,
    seed: int | None = None,                  # reproducibility knob (optional)
    debug_logger=None,                        # UI logger (e.g., Streamlit toast)
) -> List[dict]:
    """
    Returns a list of normalized test-case dicts.
    """
    ac_list = [str(x) for x in (ac_list or []) if str(x).strip()]
    uc_list = [str(x) for x in (uc_list or []) if str(x).strip()]
    requirement_text = _norm_str(requirement_text)
    extra_details = _norm_str(extra_details)

    if num_tests <= 0:
        return []

    categories = _distribute_categories(num_tests, mix=mix, allow_adhoc=include_ad_hoc)
    messages = build_prompt(
        requirement_text=requirement_text,
        ac_list=ac_list,
        uc_list=uc_list,
        num_tests=num_tests,
        extra_details=extra_details,
        output_style=output_style,
        categories=categories
    )

    # ----------------- cache check -----------------
    params_for_cache = {
        "model": MODEL,
        "num_tests": num_tests,
        "output_style": output_style,
        "include_ad_hoc": include_ad_hoc,
        "mix": mix,
        "temperature": float(temperature),
        "timeout_s": int(timeout_s),
        "seed": int(seed) if isinstance(seed, int) else None,
        "categories": categories,
    }
    key_prompt = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    hit = cache.get(key_prompt, params_for_cache)
    if hit is not None:
        _log(debug_logger, "cache: HIT ✅")
        return [_ensure_row_fields(o) for o in (hit if isinstance(hit, list) else [])]

    _log(debug_logger, "cache: MISS → calling OpenAI ⏳")
    # ------------- call OpenAI -------------
    resp = _openai_chat(
        messages,
        temperature=temperature,
        timeout_s=timeout_s,
        seed=seed,
        retries=2,
        debug_logger=debug_logger
    )
    raw_text = resp.choices[0].message.content

    try:
        payload = _extract_json_array(raw_text)
    except ValueError:
        _log(debug_logger, "Strict retry with harder JSON constraint…")
        strict_suffix = (
            "\nIMPORTANT: Return ONLY a JSON array of objects as specified. "
            "No natural language, no markdown, no extra keys. "
            "If unsure, output just the array."
        )
        resp2 = _openai_chat(
            messages + [{"role": "system", "content": strict_suffix}],
            temperature=temperature,
            timeout_s=timeout_s,
            seed=seed,
            retries=1,
            debug_logger=debug_logger
        )
        raw_text2 = resp2.choices[0].message.content
        payload = _extract_json_array(raw_text2)

    # Normalize rows & ensure count (if model sent more/less, we clamp/pad)
    rows = [_ensure_row_fields(o) for o in payload if isinstance(o, dict)]
    if len(rows) > num_tests:
        rows = rows[:num_tests]
    elif len(rows) < num_tests:
        for _ in range(num_tests - len(rows)):
            rows.append(_ensure_row_fields({}))

    # ----------------- cache store -----------------
    cache.set(key_prompt, params_for_cache, rows)
    _log(debug_logger, f"cache: STORE ({len(rows)} cases) 💾")
    return rows
