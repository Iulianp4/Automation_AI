# Changelog

Toate modificările notabile aduse proiectului **Automation_AI**.

## [Unreleased]
- Tuning și documentare suplimentară pentru raportul de disertație.

## [2025-08-23]
### Added
- Creat fișiere `README.md` și `CHANGELOG.md`.
- Documentat structura proiectului, setup, și mod de rulare.

## [2025-08-20–22]
### Added
- Integrare completă a motorului de comparație (AI vs Manual).
- Măsurători: coverage, novelty, precision, recall, F1, scor calitate, grading.
- Export Excel: matches, ai_only, manual_only, scores_summary, trace_matrix, distribuții.
- Suport pentru **mode comparison-only** (fără generare).
- Prompturi îmbogățite cu detalii (`requirement_details`, `ac_details`, `uc_details`).

### Changed
- Generarea testelor face acum independent pe Requirements, AC, și UC.
- Export consolidat `results/report.xlsx`.

## [2025-08-15–19]
### Added
- Input interactiv pentru:
  - câte teste per Requirement / AC / UC,
  - stil de output (classic/gherkin/both),
  - mix (balanced/positive_heavy/negative_heavy),
  - includere AdHoc.
- Traceability și metrics pe multiple sheet-uri în Excel.
- Legend user-friendly cu mapping complet pentru toate câmpurile.

### Fixed
- Normalizarea ID-urilor lipsă (`REQ-LONE-*`, `AC-LONE-*`, `UC-LONE-*`).
- Bug-uri la citirea fișierelor Excel cu coloane lipsă.

## [2025-08-10–14]
### Added
- Data loaders (`preprocess.py`) pentru Requirements, Acceptance Criteria și Use Cases.
- Export de bază: `generated_raw`, `execution_export`.
- Prompt builder inițial și integrare cu OpenAI API.
- `.env` pentru cheie API și `.gitignore` actualizat.
