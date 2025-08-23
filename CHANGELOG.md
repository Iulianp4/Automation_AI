# Changelog

## [2025-08-12]
- Add comparison engine (manual ↔ AI) with greedy best-match and blended similarity (difflib + jaccard).
- Metrics: coverage, novelty, precision, recall, F1, quality (imperative steps, verifiable expected, Gherkin completeness, numbering, vagueness penalty), grade (0..10).
- Reports: matches, ai_only, manual_only, scores_summary, category_distribution, per_requirement_density, trace_matrix, summary sheet.
- Generation knobs: per-source test counts, output style (classic/gherkin/both), mix (balanced/positive_heavy/negative_heavy), AdHoc toggle.
- Independent generation per source (Requirements / AC / UC) + consolidated report.
- Details columns for richer prompts: `requirement_details`, `acceptance_criteria_details`, `use_cases_details`.
- Robust OpenAI calls (retry/timeout), token usage print.

## [2025-08-10–11]
- Data loaders & normalization (`preprocess.py`), synthetic IDs (`REQ-LONE`, `AC-LONE`, `UC-LONE`).
- Excel exports: generated_raw, execution_export, traceability, metrics, legend, run_info.
- Initial prompt builder and model integration.
