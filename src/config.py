from pathlib import Path

# Locații fișiere
BASE = Path(__file__).resolve().parent.parent
DATA_FILES = {
    "requirements": "data/requirements.xlsx",
    "acceptance_criteria": "data/acceptance_criteria.xlsx",
    "use_cases": "data/use_cases.xlsx",
    "report": "results/report.xlsx",
}

# Export sheet-ul pentru execuție
EXECUTION_EXPORT_HEADERS = [
    "Nr.Crt",
    "Steps",
    "Actual Result",
    "Expected Result",
    "Document of evidence",
]

# === MODURI DE GENERARE ===
# Valori permise: "row" (per-rând) sau "group" (agregat pe Requirement ID)
AC_MODE = "row"     # "row" = fiecare rând AC produce teste independent
UC_MODE = "row"     # "group" = se grupează toate UC cu același Requirement ID

# (poți schimba aici rapid fără să modifici main.py)
