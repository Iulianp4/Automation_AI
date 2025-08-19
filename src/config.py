from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_FILES = {
    "requirements": "data/requirements.xlsx",
    "acceptance_criteria": "data/acceptance_criteria.xlsx",
    "use_cases": "data/use_cases.xlsx",
    "report": "results/report.xlsx",

    # NEW: manual test cases written by a human tester
    "manual": "data/manual_cases.xlsx",
}

EXECUTION_EXPORT_HEADERS = [
    "Nr.Crt",
    "Steps",
    "Actual Result",
    "Expected Result",
    "Document of evidence",
]

# Modes for AC/UC
AC_MODE = "row"
UC_MODE = "row"

# --- Comparison settings (for next step – scoring) ---
# Title+Expected-based soft similarity (0..1). We'll start with a safe threshold.
COMPARE_SIM_THRESHOLD = 0.60
# Strategy can be: "title_expected" or "title_steps_expected"
COMPARE_STRATEGY = "title_expected"
