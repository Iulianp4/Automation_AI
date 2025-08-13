from pathlib import Path

# Directorul de bază al proiectului
BASE_DIR = Path(__file__).resolve().parent.parent

# Locațiile fișierelor de intrare și ieșire
DATA_FILES = {
    "requirements": BASE_DIR / "data" / "requirements.xlsx",
    "acceptance_criteria": BASE_DIR / "data" / "acceptance_criteria.xlsx",
    "use_cases": BASE_DIR / "data" / "use_cases.xlsx",
    "report": BASE_DIR / "results" / "report.xlsx"
}

# Coloanele din exportul pentru execuția testelor
EXECUTION_EXPORT_HEADERS = [
    "Nr.Crt",
    "Steps",
    "Actual Result",
    "Expected Result",
    "Document of evidence"
]
