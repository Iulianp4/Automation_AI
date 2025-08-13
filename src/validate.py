from typing import List

def coverage_score(criteria: List[str], steps_text: str) -> float:
    # Heuristic: fraction of criteria substrings present in steps
    if not criteria:
        return 0.0
    found = sum(1 for c in criteria if c.lower()[:20] in steps_text.lower())
    return found / len(criteria)
