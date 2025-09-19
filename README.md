<div align="center">

# Clinical Copilot (Single File Edition)

**A single Python file (`source_code.py`) Streamlit app that ingests patient data (FHIR or Excel), generates a concise structured clinical summary (Problem / Key Relevant Factors / 3‑Step Plan) using an LLM via OpenRouter, lets you select plan steps, add notes, export a PDF, and log visits.**

</div>

> This README is intentionally trimmed to describe only the self‑contained Streamlit script. Remove any unused sections before publishing if you further simplify the file.

## What You Get
One file: `source_code.py` (rename from `try.py` if needed) containing:
- FHIR retrieval (public HAPI server).
- Optional Excel upload with flexible column mapping.
- Relevance filtering (heuristic) to shrink prompt context.
- Deterministic LLM prompt + parser (Problem / Factors / Plan).
- Interactive plan step multi‑select (stable form handling).
- Notes + optional PDF export (ReportLab) or browser print fallback.
- Patient & visit persistence (in‑memory workbook if uploaded, otherwise local Excel file).

## Quick Start (Windows PowerShell Example)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENROUTER_API_KEY="sk-or-xxxx"
streamlit run source_code.py
```

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-xxxx
streamlit run source_code.py
```

Open the printed local URL, enter a patient name + complaint, optionally upload an Excel file, and click Analyze.

## Minimal Dependencies
`requirements.txt` already lists them:
- streamlit, requests, pandas, openpyxl, openai, reportlab (optional PDF), plus standard ML libs if you keep them.
- Remove unused lines (e.g., `fastapi`, `transformers`, `torch`) if you are truly shipping only the Streamlit app.

### If You Want a Minimal Requirements File
```text
streamlit
requests
pandas
openpyxl
openai
reportlab  # optional; delete if you skip PDF
```

## Environment Variables
| Name | Required | Purpose |
|------|----------|---------|
| `OPENROUTER_API_KEY` | Yes | Enables LLM generation. Without it you get a warning and placeholder message. |
| `OPENROUTER_MODEL` | No | Override default model (defaults to a Gemma free tier then fallback). |
| `FHIR_BASE_URL` | No | Alternate FHIR endpoint (defaults to public HAPI). |
| `STORAGE_XLSX` | No | Local Excel filename for persistence when no upload present. |

Set them in your shell or a `.env` (not loaded automatically unless you add python-dotenv).

## Excel Input Schema (Flexible)
Canonical columns + synonyms (case‑insensitive):
| Canonical | Synonyms |
|-----------|----------|
| name | patient, patient_name, full_name |
| conditions | dx, problems, medical_history |
| medications | meds, drugs |
| procedures | history_procedures, surgery, implants |
| diagnostic_reports | reports, labs, imaging |
| gender | sex |
| birthDate | dob, date_of_birth |
| address | addr |

Value separators: semicolon, comma, or newline.

## Persistence Behavior
- With upload: In‑memory mutated workbook; download updated copy after saves.
- Without upload: Auto‑creates/updates `patients_storage.xlsx` (or `STORAGE_XLSX`).
- Patients are upserted by case‑insensitive name.

## PDF Export
If `reportlab` is installed a “Download Prescription PDF” button appears. Otherwise the UI suggests using your browser's print dialog.

## How the LLM Prompt Works
The code constructs a strict template (Problem / Key Relevant Factors / Plan 3 steps, <=120 words). Output parsing relies only on section headers and simple numbering, making it robust to minor model phrasing changes.

## Trimming for Distribution
If you only ship `source_code.py`:
1. Rename `try.py` → `source_code.py`.
2. Remove unused dependencies from `requirements.txt`.
3. Keep this README or shorten further (e.g., only Quick Start + Disclaimer).

## Clinical & Ethical Disclaimer
Not a medical device. Educational/demo only. Does not diagnose, treat, or replace professional judgment. Do not use with real PHI unless you have implemented appropriate compliance, security, and logging measures.

## Security Notes
- Never commit real API keys or real patient data.
- Public HAPI FHIR is a shared sandbox—queries and data may be purged or rate‑limited.

## Optional Future Enhancements
- Add streaming output.
- Add auth (basic / OAuth) if deployed.
- Replace heuristic relevance with embeddings.
- Containerize (Docker) for reproducible deployment.

## Short Description (<= 350 chars)
Single‑file Streamlit clinical copilot (`source_code.py`) that ingests FHIR or Excel patient data, filters relevance, prompts an LLM for structured Problem/Factors/3‑Step Plan, supports plan selection, notes, PDF export, and visit logging—no embedded secrets, env‑based configuration.

## License
Add a `LICENSE` file (e.g., MIT) before distribution.

---
Questions or suggestions? Open an issue or start a discussion.
