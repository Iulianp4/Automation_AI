import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# One-time client init (API key via env)
# Make sure you have OPENAI_API_KEY in your .env or system env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model can be overridden via env OPENAI_MODEL; sensible default below
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------- utils: retry + safe call ----------------
def safe_completion(**kwargs):
    """
    Thin retry wrapper around client.chat.completions.create
    """
    last_err = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(timeout=40, **kwargs)
        except Exception as e:
            last_err = e
            print(f"⚠️ OpenAI call failed (attempt {attempt+1}/3): {e}")
            time.sleep(2)
    raise RuntimeError(f"OpenAI API failed after 3 attempts: {last_err}")


def _distribute_tests(num_tests: int, mix: str, include_ad_hoc: bool) -> Dict[str, int]:
    """
    Returns a dict mapping categories to counts that sum to num_tests.
    Categories used: Positive, Negative, Boundary, Security (+ AdHoc if allow)
    """
    base_cats = ["Positive", "Negative", "Boundary", "Security"]
    if include_ad_hoc:
        base_cats.append("AdHoc")

    if num_tests <= 0:
        return {}

    # default: balanced among chosen cats
    if mix == "balanced":
        cats = base_cats
        base = num_tests // len(cats)
        rem = num_tests % len(cats)
        dist = {c: base for c in cats}
        for i in range(rem):
            dist[cats[i]] += 1
        return dist

    # tilts
    if mix == "positive_heavy":
        # 70/20/10 split across Positive/Negative/Boundary, spill to others if present
        p = max(1, int(round(num_tests * 0.7)))
        n = max(0, int(round(num_tests * 0.2)))
        b = max(0, num_tests - (p + n))
        dist = {"Positive": p, "Negative": n, "Boundary": b}
    elif mix == "negative_heavy":
        n = max(1, int(round(num_tests * 0.5)))
        p = max(0, int(round(num_tests * 0.3)))
        b = max(0, num_tests - (p + n))
        dist = {"Positive": p, "Negative": n, "Boundary": b}
    else:
        dist = {"Positive": num_tests}

    # if Security/AdHoc exist and we haven't assigned, spread from Boundary/Positive
    remaining_cats = set(base_cats) - set(dist.keys())
    for c in remaining_cats:
        # steal 1 by 1 from the largest bucket until we can allocate at least 1
        if num_tests <= 2:
            continue
        # pick donor
        donor = max(dist, key=lambda k: dist[k])
        if dist[donor] > 1:
            dist[donor] -= 1
            dist[c] = 1

    # ensure sum == num_tests
    delta = num_tests - sum(dist.values())
    while delta != 0:
        if delta > 0:
            # add to the currently smallest category to even out
            target = min(dist, key=lambda k: dist[k])
            dist[target] += 1
            delta -= 1
        else:
            # remove from largest
            donor = max(dist, key=lambda k: dist[k])
            if dist[donor] > 0:
                dist[donor] -= 1
                delta += 1
            else:
                break
    return dist


def _build_system_prompt(output_style: str) -> str:
    """
    Gives strict guidance to keep responses structured & compact.
    """
    gherkin_hint = ""
    if output_style in ("gherkin", "both"):
        gherkin_hint = (
            "For each test also include a concise 'gherkin' field with Given/When/Then verification."
        )
    return (
        "You are a senior QA engineer. Generate high-quality UI test cases. "
        "Be concrete, imperative, and avoid vagueness. "
        "Output MUST be STRICT JSON (no commentary, no markdown), a list of objects with: "
        "title, preconditions, steps, data, expected, category"
        + (", gherkin" if gherkin_hint else "")
        + ". "
        "Keep steps as a numbered list string (e.g., '1) ..., 2) ...'). "
        "Categories allowed: Positive, Negative, Boundary, Security, AdHoc. "
        + gherkin_hint
    )


def _build_user_prompt(
    requirement_text: str,
    ac_list: List[str],
    uc_list: List[str],
    num_tests: int,
    dist: Dict[str, int],
    extra_details: str,
    output_style: str,
    mix: str,
    include_ad_hoc: bool,
) -> str:
    # Plain text context assembly
    blocks = []
    if requirement_text.strip():
        blocks.append(f"REQUIREMENT:\n{requirement_text.strip()}")
    if ac_list:
        blocks.append("ACCEPTANCE CRITERIA:\n- " + "\n- ".join([s.strip() for s in ac_list if s.strip()]))
    if uc_list:
        blocks.append("USE CASE NOTES:\n- " + "\n- ".join([s.strip() for s in uc_list if s.strip()]))
    if extra_details.strip():
        blocks.append(f"EXTRA TESTER DETAILS:\n{extra_details.strip()}")

    ctx = "\n\n".join(blocks) if blocks else "No upstream text was provided."

    dist_txt = ", ".join([f"{k}={v}" for k, v in dist.items()])
    style_txt = (
        "classic only" if output_style == "classic" else
        "gherkin only" if output_style == "gherkin" else
        "both classic and gherkin"
    )

    return (
        f"{ctx}\n\n"
        f"Please generate exactly {num_tests} UI test cases distributed as: {dist_txt}. "
        f"Write them in {style_txt}. "
        "Each test case must contain fields: title, preconditions, steps, data, expected, category"
        + (", gherkin" if output_style in ("gherkin", "both") else "")
        + ". "
        "Be specific about inputs and verifications. Do not invent backend APIs; keep it UI-level. "
        "Prefer crisp, atomic steps; avoid multi-action steps. "
        "Return ONLY valid JSON (a JSON array)."
    )


def _extract_json_array(text: str) -> Any:
    """
    Robust parser that tries strict json first, then extracts codefences if needed.
    """
    try:
        return json.loads(text)
    except Exception:
        # try to find a JSON array within the text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end+1]
            try:
                return json.loads(snippet)
            except Exception:
                pass
    raise ValueError("Model did not return valid JSON array.")


def _normalize_case_obj(obj: Dict[str, Any], want_gherkin: bool) -> Dict[str, str]:
    def _g(key):  # get & strip
        return str(obj.get(key, "") or "").replace("\u00A0", " ").strip()

    out = {
        "title": _g("title"),
        "preconditions": _g("preconditions"),
        "steps": _g("steps"),
        "data": _g("data"),
        "expected": _g("expected"),
        "category": _g("category") or "Positive",
    }
    if want_gherkin:
        out["gherkin"] = _g("gherkin")
    else:
        out["gherkin"] = ""
    return out


# ---------------- main generation entry ----------------
def generate_with_gpt(
    requirement_text: str,
    ac_list: List[str],
    uc_list: List[str],
    num_tests: int = 5,
    extra_details: str = "",
    output_style: str = "both",           # classic / gherkin / both
    include_ad_hoc: bool = True,
    mix: str = "balanced",                # balanced / positive_heavy / negative_heavy
) -> List[Dict[str, str]]:
    """
    Returns a list of {title, preconditions, steps, data, expected, category, gherkin}
    """
    num_tests = max(1, int(num_tests or 1))
    dist = _distribute_tests(num_tests, mix=mix, include_ad_hoc=include_ad_hoc)

    system_msg = _build_system_prompt(output_style)
    user_msg = _build_user_prompt(
        requirement_text=requirement_text or "",
        ac_list=ac_list or [],
        uc_list=uc_list or [],
        num_tests=num_tests,
        dist=dist,
        extra_details=extra_details or "",
        output_style=output_style,
        mix=mix,
        include_ad_hoc=include_ad_hoc,
    )

    resp = safe_completion(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        response_format={"type": "json_object"} if OPENAI_MODEL.startswith("gpt-4.1") else None,  # optional
    )

    # print usage for cost awareness
    usage = getattr(resp, "usage", None)
    if usage:
        prompt_toks = getattr(usage, "prompt_tokens", 0)
        comp_toks = getattr(usage, "completion_tokens", 0)
        total_toks = getattr(usage, "total_tokens", 0)
        print(f"ℹ️ Tokens used: prompt={prompt_toks}, completion={comp_toks}, total={total_toks}")

    raw_text = resp.choices[0].message.content or ""
    payload = _extract_json_array(raw_text)

    if not isinstance(payload, list):
        raise ValueError("Model returned JSON but not a list.")

    want_gherkin = output_style in ("gherkin", "both")
    cases: List[Dict[str, str]] = []
    for item in payload:
        try:
            norm = _normalize_case_obj(item, want_gherkin=want_gherkin)
            cases.append(norm)
        except Exception:
            continue

    # Keep at most num_tests in case the model overshot
    if len(cases) > num_tests:
        cases = cases[:num_tests]

    # If model undershot, pad with minimal positive stubs (rare)
    while len(cases) < num_tests:
        idx = len(cases) + 1
        cases.append({
            "title": f"Auto-generated test {idx}",
            "preconditions": "",
            "steps": "1) Do the main user action\n2) Observe system output",
            "data": "",
            "expected": "System responds as per requirement.",
            "category": "Positive",
            "gherkin": "" if not want_gherkin else "Given context\nWhen user performs main action\nThen system responds per requirement",
        })

    return cases
