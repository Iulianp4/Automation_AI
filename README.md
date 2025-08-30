# Automation_AI — Test Case Generator

Generate high-quality UI test cases from **Requirements / Acceptance Criteria / Use Cases**, add tester **Details**, export execution-ready Excel, and **compare** AI output vs. a **manual baseline** (precision/recall/F1, coverage, distributions, traceability).

---

## ✨ Features

- **Input sources**: `requirements.xlsx`, `acceptance_criteria.xlsx`, `use_cases.xlsx` (+ optional `manual_cases.xlsx`)
- **Extra tester context** via `*_details` to boost coverage
- **Balanced test mix** (Positive / Negative / Boundary / Security / AdHoc)
- **Output styles**: classic steps or Given/When/Then (Gherkin) — or both
- **Exports**: `generated_raw`, `execution_export`, `traceability`, `metrics`, `legend`, `run_info`
- **Comparison** vs manual: greedy matching + **precision / recall / F1**, per-category & per-requirement stats, trace matrix
- **Comparison-only mode** (skip generation; reuse `results/*.xlsx`)
- **Robust OpenAI calls**: retry, timeout, token usage print

---

## 🗂 Repo structure

```
Automation_AI/
├─ data/
│  ├─ requirements.xlsx
│  ├─ acceptance_criteria.xlsx
│  ├─ use_cases.xlsx
│  └─ manual_cases.xlsx        # optional baseline
├─ results/
│  ├─ report_from_requirements.xlsx
│  ├─ report_from_acceptance.xlsx
│  ├─ report_from_use_cases.xlsx
│  ├─ report.xlsx              # consolidated AI
│  └─ report_comparison.xlsx   # AI vs Manual
├─ src/
│  ├─ config.py
│  ├─ preprocess.py
│  ├─ generate_gpt.py
│  ├─ comparison.py
│  └─ __init__.py
├─ main.py
├─ .env                        # contains OPENAI_API_KEY (NOT committed)
├─ README.md
├─ CHANGELOG.md
└─ requirements.txt
```

---

## 🔧 Setup

1. **Python & venv**
   ```bash
   py -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment**
   Create `.env` (do NOT commit):
   ```
   OPENAI_API_KEY=sk-...
   # optional:
   OPENAI_MODEL=gpt-4o-mini
   ```

3. **Config flags** (`src/config.py`)
   ```python
   INTERACTIVE = True  # set False to run with defaults (no prompts)
   DEFAULTS = {
     "num_req": 5, "num_ac": 5, "num_uc": 5,
     "output_style": "both",              # classic / gherkin / both
     "include_ad_hoc": True,
     "mix": "balanced",                   # balanced / positive_heavy / negative_heavy
     "compare_only": False,
     "do_comparison": True,
     "similarity_strategy": "title_expected",  # or title_steps_expected
     "similarity_threshold": 0.75,
   }
   ```

---

## 📥 Data templates (columns)

- **requirements.xlsx**
  - `requirement_id`, `requirement_name`, `requirement_text`,
    `requirement_rationale` (opt), `requirement_platform` (opt),
    `requirement_details` (extra tester context)

- **acceptance_criteria.xlsx**
  - `acceptance_criteria_story_id` → `requirement_id`
  - `acceptance_criteria` → `ac_text`
  - `acceptance_criteria_notes` (opt), `acceptance_criteria_comments` (opt)
  - `acceptance_criteria_details` → `ac_details`

- **use_cases.xlsx**
  - `use_cases_story_id` → `requirement_id`
  - Any of: `use_cases_title`, `use_cases_description`,
    `use_cases_preconditions`, `use_cases_main_flow`,
    `use_cases_alternative_flows`, `use_cases_exception_flows`,
    `use_cases_business_rules`
  - `use_cases_details` → `uc_details` (extra tester context)

- **manual_cases.xlsx** (optional baseline; friendly headers accepted)
  - `Requirement ID, Requirement Name, Test Case ID, Title, Preconditions, Steps,
     Test Data, Expected Result, Category, Gherkin, Source`

> Missing IDs fall back to synthetic: `REQ-LONE-*`, `AC-LONE-*`, `UC-LONE-*`.

---

## ▶️ Run

### Generate + (optional) compare
```bash
python main.py
```
- Choose **how many tests per source** (Req/AC/UC), **style** (classic/gherkin/both),
  **mix** (balanced/positive_heavy/negative_heavy), allow **AdHoc** or not.
- Results saved to `results/report_from_*.xlsx` and `results/report.xlsx`.

If `data/manual_cases.xlsx` exists, you’ll be prompted to **compare AI vs manual** and produce `results/report_comparison.xlsx`.

### Comparison-only mode
Use existing AI results from `results/*.xlsx` (sheet `generated_raw`) without regenerating:
- When prompted: *Run comparison only (skip generation)?* → **yes**, or
- Set in `config.py`: `INTERACTIVE=False`, `DEFAULTS["compare_only"]=True`.

---

## 📤 Outputs (Excel sheets)

- **generated_raw** – normalized test cases
- **execution_export** – `Nr.Crt / Steps / Actual Result / Expected Result / Document of evidence`
- **traceability** – one row / test (with req id/name, category, has gherkin)
- **metrics** – totals, by source, by category, by requirement, cat×source pivot
- **legend** – input/output field dictionary
- **run_info** – parameters used

**Comparison (`report_comparison.xlsx`):**
- `matches` – aligned pairs with similarity
- `ai_only`, `manual_only` – non-matched items
- `scores_summary` – overall **precision, recall, F1, grade**
- `dist_by_category` – AI vs manual (counts & %)
- `per_requirement_density` – counts by requirement id
- `trace_matrix` – cross-tab of matched reqs (AI×Manual)

---

## 🧠 How it works (short)

- `preprocess.py` reads & normalizes Excel, groups AC/UC by `requirement_id`,
  builds per-row context + optional `*_details`.
- `generate_gpt.py` builds a strict JSON prompt, **distributes categories** based on your `mix`, calls OpenAI with **retry/timeout** and prints token usage.
- `main.py` orchestrates generation/export for each source and the consolidated report; supports **comparison-only**.  
- `comparison.py` computes greedy matches and metrics (**precision/recall/F1**, coverage, novelty, quality), plus analysis tables.

---

## 🩹 Troubleshooting

- **PowerShell cannot activate venv**  
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (then retry).
- **Missing `openpyxl`**  
  `pip install openpyxl`
- **API key / quota**  
  Ensure `.env` contains `OPENAI_API_KEY` and your account has credit.
- **Invalid columns**  
  The app prints `⚠️ <file>: missing columns -> ...`. Fix Excel headers accordingly.
- **No matches in comparison**  
  Lower `DEFAULTS["similarity_threshold"]` (e.g., `0.65`) or use `title_steps_expected` strategy.

---

## 🔒 Security & privacy

- **Never commit secrets** (`.env` in `.gitignore`).
- Use sanitized, non-confidential data for public demos.
- For company data, confirm policy & retention before usage.

---

## 📄 License

MIT (or your preferred license). Update `LICENSE` accordingly.
