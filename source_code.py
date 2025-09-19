import os
import re
import io
import textwrap
from datetime import datetime
import streamlit as st
import requests
import pandas as pd
from typing import Dict, List, Tuple

# Optional PDF support
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False
"""Single-file Streamlit clinical copilot.

Change log:
- Replaced OpenAI SDK usage with direct OpenRouter REST calls (SDK caused TypeError under Python 3.13 on Streamlit Cloud).
"""

# OpenRouter API setup (must be provided via environment variable)
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    st.warning("OPENROUTER_API_KEY not set. Set it in your environment before using analysis features.")

FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"

def _safe_get_text(obj, *keys):
    cur = obj
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k, {})
        else:
            return ""
    if isinstance(cur, str):
        return cur
    return cur.get("text", "") if isinstance(cur, dict) else ""


def fetch_patient_data_by_keyword(name_keyword):
    try:
        url = f"{FHIR_BASE_URL}/Patient"
        params = {"name": name_keyword, "_count": 5}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for entry in data.get("entry", []):
            patient_info = {}
            resource = entry.get("resource", {})
            names = resource.get("name", [])
            if names:
                given = names[0].get("given", [])
                family = names[0].get("family", "")
                full_name = " ".join(given) + " " + family
                patient_info["name"] = full_name.strip()
            else:
                patient_info["name"] = "Unknown"
            patient_info["gender"] = resource.get("gender", "")
            patient_info["birthDate"] = resource.get("birthDate", "")
            patient_info["address"] = ", ".join([addr.get("text", "") for addr in resource.get("address", [])]) if resource.get("address") else ""

            patient_id = resource.get("id", "")

            # Fetch conditions
            cond_url = f"{FHIR_BASE_URL}/Condition"
            cond_params = {"patient": patient_id, "_count": 10}
            cond_resp = requests.get(cond_url, params=cond_params)
            cond_resp.raise_for_status()
            conditions_data = []
            for ce in cond_resp.json().get("entry", []):
                r = ce.get("resource", {})
                condition_text = r.get("code", {}).get("text", "")
                if condition_text:
                    conditions_data.append(condition_text)
            patient_info["conditions"] = conditions_data

            # Fetch medications (MedicationStatement)
            meds_url = f"{FHIR_BASE_URL}/MedicationStatement"
            meds_params = {"patient": patient_id, "_count": 20}
            meds_resp = requests.get(meds_url, params=meds_params)
            meds_resp.raise_for_status()
            meds_data = []
            for me in meds_resp.json().get("entry", []):
                r = me.get("resource", {})
                med_text = _safe_get_text(r, "medicationCodeableConcept")
                if not med_text:
                    # Try contained medication
                    med_text = _safe_get_text(r, "medicationReference", "display")
                if med_text:
                    meds_data.append(med_text)
            patient_info["medications"] = meds_data

            # Fetch procedures (e.g., implants/rods)
            proc_url = f"{FHIR_BASE_URL}/Procedure"
            proc_params = {"patient": patient_id, "_count": 20}
            proc_resp = requests.get(proc_url, params=proc_params)
            proc_resp.raise_for_status()
            procs = []
            for pe in proc_resp.json().get("entry", []):
                r = pe.get("resource", {})
                ptxt = _safe_get_text(r, "code")
                if ptxt:
                    procs.append(ptxt)
            patient_info["procedures"] = procs

            # Fetch diagnostic reports
            diag_url = f"{FHIR_BASE_URL}/DiagnosticReport"
            diag_params = {"patient": patient_id, "_count": 10}
            diag_resp = requests.get(diag_url, params=diag_params)
            diag_resp.raise_for_status()
            diag_data = []
            for de in diag_resp.json().get("entry", []):
                r = de.get("resource", {})
                code = r.get("code", {}).get("text", "")
                if code:
                    diag_data.append(code)
            patient_info["diagnostic_reports"] = diag_data

            results.append(patient_info)
        return results
    except Exception as e:
        st.error(f"Error fetching patient data: {e}")
        return []

def _filter_relevance(current_complaint: str, patient_data: Dict) -> Tuple[List[str], List[str]]:
    cc = current_complaint.lower()
    text_items = []
    # Gather phrases from multiple sources
    for k in ("conditions", "medications", "procedures", "diagnostic_reports"):
        for t in patient_data.get(k, []) or []:
            if isinstance(t, str) and t.strip():
                text_items.append(t.strip())

    relevant, ignored = [], []
    # Heuristic keywords for musculoskeletal injuries/fracture context
    rel_kw = [
        "rod", "implant", "prosthesis", "hardware", "fixation", "orthopedic",
        "diabetes", "metformin", "warfarin", "apixaban", "rivaroxaban", "heparin",
        "osteoporosis", "steroid", "prednisone", "bisphosphonate", "smoker", "smoking",
        "peripheral vascular", "neuropathy", "osteomyelitis", "fracture", "bone",
    ]
    # Commonly non-relevant to isolated leg injury (unless anesthesia planning etc.)
    ign_kw = ["asthma", "copd", "common cold", "seasonal allergy", "resolved cancer", "eczema"]

    # If complaint includes injury/fracture/leg
    if re.search(r"leg|ankle|knee|tibia|fibula|femur|fractur|sprain|fall|injur", cc):
        for item in text_items:
            lit = item.lower()
            if any(k in lit for k in rel_kw):
                relevant.append(item)
            elif any(k in lit for k in ign_kw):
                ignored.append(item)
            else:
                # neutral: ignore by default to stay concise
                ignored.append(item)
    else:
        # If not a musculoskeletal context, mark nothing special
        ignored = []
        relevant = text_items[:5]

    # De-duplicate while preserving order
    def dedup(seq):
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    return dedup(relevant), dedup(ignored)


def generate_prompt(patient_data, current_complaint):
    relevant, ignored = _filter_relevance(current_complaint, patient_data)
    rel_str = "\n".join(f"- {r}" for r in relevant[:6]) or "- None"
    ign_str = "\n".join(f"- {i}" for i in ignored[:6]) or "- None"

    prompt = f"""
You are a clinical copilot. Be concise and actionable.

Context:
Current Complaint: {current_complaint}
Relevant History:
{rel_str}
Ignored History (not relevant now):
{ign_str}

Produce exactly this template, max 120 words total, no preamble, no disclaimers:
Problem: <one sentence>
Key Relevant Factors:
- <2-4 bullets>
Plan (3 steps):
1. <short>
2. <short>
3. <short>
""".strip()
    return prompt

def call_openrouter_llm(prompt: str) -> str:
    if not API_KEY:
        return "API key missing. Set OPENROUTER_API_KEY and rerun."
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-3n-e2b-it:free")
    headers = {
        "Authorization": f"Bearer {API_KEY}
"        , "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for attribution/analytics
        "HTTP-Referer": os.getenv("APP_URL", "https://github.com/KrithikVishal/Clinical-Copilot-AI"),
        "X-Title": os.getenv("APP_TITLE", "Clinical Copilot"),
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 220,
    }
    url = "https://openrouter.ai/api/v1/chat/completions"
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "(no content)")
    except Exception as e:
        # Attempt a single fallback model if explicitly different
        if model != "mistralai/mistral-7b-instruct:free":
            payload["model"] = "mistralai/mistral-7b-instruct:free"
            try:
                resp2 = requests.post(url, headers=headers, json=payload, timeout=60)
                resp2.raise_for_status()
                data2 = resp2.json()
                return data2.get("choices", [{}])[0].get("message", {}).get("content", f"Fallback error: {e}")
            except Exception as e2:
                return f"LLM error (fallback failed): {e2}"
        return f"LLM error: {e}"

# Streamlit app UI
st.title("Clinical Copilot AI Assistant")

# Optional XLSX upload overrides FHIR dataset
uploaded = st.file_uploader("Upload patient dataset (.xlsx)", type=["xlsx"]) 

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "idle"
if "selected" not in st.session_state:
    st.session_state.selected = {}
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = {}
if "analyze_active" not in st.session_state:
    st.session_state.analyze_active = False
if "patients_current" not in st.session_state:
    st.session_state.patients_current = []
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
if "uploaded_bytes_current" not in st.session_state:
    st.session_state.uploaded_bytes_current = None
if "uploaded_bytes_original" not in st.session_state:
    st.session_state.uploaded_bytes_original = None

# Capture uploaded file bytes/name for read/write
if uploaded is not None:
    try:
        b = uploaded.getvalue()
    except Exception:
        b = uploaded.read()
    st.session_state.uploaded_name = uploaded.name
    st.session_state.uploaded_bytes_original = b
    # Initialize current bytes only once per new upload
    if st.session_state.uploaded_bytes_current is None or st.session_state.uploaded_name != uploaded.name:
        st.session_state.uploaded_bytes_current = b

# Local storage for persistence (patients + visits)
STORAGE_XLSX_PATH = os.getenv(
    "STORAGE_XLSX", os.path.join(os.getcwd(), "patients_storage.xlsx")
)

def _ensure_storage_file(path: str) -> None:
    if os.path.exists(path):
        return
    empty_patients = pd.DataFrame(
        columns=[
            "name",
            "gender",
            "birthDate",
            "address",
            "conditions",
            "medications",
            "procedures",
            "diagnostic_reports",
        ]
    )
    empty_visits = pd.DataFrame(
        columns=[
            "timestamp",
            "name",
            "complaint",
            "problem",
            "selected_factors",
            "selected_plan",
            "notes",
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        empty_patients.to_excel(writer, sheet_name="patients", index=False)
        empty_visits.to_excel(writer, sheet_name="visits", index=False)

def _read_storage(path: str):
    _ensure_storage_file(path)
    xls = pd.read_excel(path, sheet_name=None)
    if "patients" not in xls:
        xls["patients"] = pd.DataFrame(
            columns=[
                "name",
                "gender",
                "birthDate",
                "address",
                "conditions",
                "medications",
                "procedures",
                "diagnostic_reports",
            ]
        )
    if "visits" not in xls:
        xls["visits"] = pd.DataFrame(
            columns=[
                "timestamp",
                "name",
                "complaint",
                "problem",
                "selected_factors",
                "selected_plan",
                "notes",
            ]
        )
    return xls

def _write_storage(path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

# Byte-based workbook helpers for uploaded files
def _read_workbook_bytes(data: bytes) -> Dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=None)
    except Exception:
        return {}

def _write_workbook_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    buf.seek(0)
    return buf.getvalue()

def _upsert_patient_record(p: Dict) -> None:
    sheets = _read_storage(STORAGE_XLSX_PATH)
    patients_df = sheets["patients"].copy()

    def join_list(v):
        return "; ".join(v) if isinstance(v, list) else (v if isinstance(v, str) else "")

    new_row = {
        "name": p.get("name", "Unknown"),
        "gender": p.get("gender", ""),
        "birthDate": p.get("birthDate", ""),
        "address": p.get("address", ""),
        "conditions": join_list(p.get("conditions", [])),
        "medications": join_list(p.get("medications", [])),
        "procedures": join_list(p.get("procedures", [])),
        "diagnostic_reports": join_list(p.get("diagnostic_reports", [])),
    }
    if not patients_df.empty and (
        patients_df["name"].str.lower() == new_row["name"].lower()
    ).any():
        idx = patients_df[
            patients_df["name"].str.lower() == new_row["name"].lower()
        ].index[0]
        for k, v in new_row.items():
            patients_df.loc[idx, k] = v
    else:
        patients_df = pd.concat([patients_df, pd.DataFrame([new_row])], ignore_index=True)
    sheets["patients"] = patients_df
    _write_storage(STORAGE_XLSX_PATH, sheets)

def _upsert_patient_record_uploaded(p: Dict) -> Tuple[Dict[str, pd.DataFrame], bytes]:
    sheets = _read_workbook_bytes(st.session_state.uploaded_bytes_current or st.session_state.uploaded_bytes_original or b"")
    if "patients" not in sheets:
        sheets["patients"] = pd.DataFrame(
            columns=[
                "name",
                "gender",
                "birthDate",
                "address",
                "conditions",
                "medications",
                "procedures",
                "diagnostic_reports",
            ]
        )
    patients_df = sheets["patients"].copy()

    def join_list(v):
        return "; ".join(v) if isinstance(v, list) else (v if isinstance(v, str) else "")

    new_row = {
        "name": p.get("name", "Unknown"),
        "gender": p.get("gender", ""),
        "birthDate": p.get("birthDate", ""),
        "address": p.get("address", ""),
        "conditions": join_list(p.get("conditions", [])),
        "medications": join_list(p.get("medications", [])),
        "procedures": join_list(p.get("procedures", [])),
        "diagnostic_reports": join_list(p.get("diagnostic_reports", [])),
    }
    if not patients_df.empty and (
        patients_df.get("name", pd.Series([], dtype=str)).astype(str).str.lower() == new_row["name"].lower()
    ).any():
        idx = patients_df[
            patients_df.get("name", pd.Series([], dtype=str)).astype(str).str.lower() == new_row["name"].lower()
        ].index[0]
        for k, v in new_row.items():
            patients_df.loc[idx, k] = v
    else:
        patients_df = pd.concat([patients_df, pd.DataFrame([new_row])], ignore_index=True)
    sheets["patients"] = patients_df
    bytes_out = _write_workbook_bytes(sheets)
    return sheets, bytes_out

def _append_visit(
    name: str,
    complaint: str,
    problem: str,
    selected_factors: List[str],
    selected_plan: List[str],
    notes: str,
) -> None:
    sheets = _read_storage(STORAGE_XLSX_PATH)
    visits_df = sheets["visits"].copy()
    visits_df = pd.concat(
        [
            visits_df,
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds")
                        + "Z",
                        "name": name,
                        "complaint": complaint,
                        "problem": problem,
                        "selected_factors": "; ".join(selected_factors or []),
                        "selected_plan": "; ".join(selected_plan or []),
                        "notes": notes or "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    sheets["visits"] = visits_df
    _write_storage(STORAGE_XLSX_PATH, sheets)

def _append_visit_uploaded(
    name: str,
    complaint: str,
    problem: str,
    selected_factors: List[str],
    selected_plan: List[str],
    notes: str,
) -> Tuple[Dict[str, pd.DataFrame], bytes]:
    sheets = _read_workbook_bytes(
        st.session_state.uploaded_bytes_current or st.session_state.uploaded_bytes_original or b""
    )
    if "visits" not in sheets:
        sheets["visits"] = pd.DataFrame(
            columns=[
                "timestamp",
                "name",
                "complaint",
                "problem",
                "selected_factors",
                "selected_plan",
                "notes",
            ]
        )
    visits_df = sheets["visits"].copy()
    visits_df = pd.concat(
        [
            visits_df,
            pd.DataFrame(
                [
                    {
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "name": name,
                        "complaint": complaint,
                        "problem": problem,
                        "selected_factors": "; ".join(selected_factors or []),
                        "selected_plan": "; ".join(selected_plan or []),
                        "notes": notes or "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    sheets["visits"] = visits_df
    bytes_out = _write_workbook_bytes(sheets)
    return sheets, bytes_out

def parse_ai_output(text: str) -> Dict:
    problem = ""
    factors: List[str] = []
    steps: List[str] = []
    lines = [l.strip() for l in (text or "").splitlines()]
    section = None
    for ln in lines:
        low = ln.lower()
        if low.startswith("problem:"):
            section = "problem"
            problem = ln.split(":", 1)[1].strip() if ":" in ln else ln
            continue
        if low.startswith("key relevant factors"):
            section = "factors"
            continue
        if low.startswith("plan"):
            section = "plan"
            continue
        if section == "problem":
            if ln:
                problem = (problem + " " + ln).strip()
        elif section == "factors":
            if ln.startswith("-"):
                factors.append(ln.lstrip("- ").strip())
        elif section == "plan":
            m = re.match(r"^\d+\.\s*(.*)$", ln)
            if m:
                steps.append(m.group(1).strip())
    return {"problem": problem, "factors": factors, "steps": steps}

def bold_keywords(text: str) -> str:
    kws = [
        r"fracture",
        r"broken",
        r"leg",
        r"knee",
        r"ankle",
        r"tibia",
        r"fibula",
        r"femur",
        r"diabetes",
        r"asthma",
        r"cancer",
        r"rod",
        r"implant",
        r"hardware",
        r"mri",
        r"x-?ray",
        r"ct",
        r"fall",
        r"sprain",
        r"pain",
    ]
    def repl(m):
        return f"**{m.group(0)}**"
    out = text or ""
    for kw in kws:
        out = re.sub(kw, repl, out, flags=re.IGNORECASE)
    return out

def make_prescription_pdf(
    patient_name: str,
    complaint: str,
    problem: str,
    factors: List[str],
    plan: List[str],
    notes: str,
) -> bytes:
    if not REPORTLAB_AVAILABLE:
        return b""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    x, y = 40, height - 50
    def write_line(s: str, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", 11)
        for w in textwrap.wrap(s, width=95):
            c.drawString(x, y, w)
            y -= 14
    write_line("Prescription / Plan", bold=True)
    y -= 6
    write_line(f"Patient: {patient_name}")
    write_line(f"Complaint: {complaint}")
    write_line(
        f"Date (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z"
    )
    y -= 8
    write_line("Problem:", bold=True)
    write_line(problem or "(not specified)")
    y -= 6
    write_line("Key Relevant Factors:", bold=True)
    for f in (factors or [])[:6]:
        write_line(f"- {f}")
    y -= 6
    write_line("Plan:", bold=True)
    for i, s in enumerate((plan or [])[:6], 1):
        write_line(f"{i}. {s}")
    if notes:
        y -= 6
        write_line("Notes:", bold=True)
        write_line(notes)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # Map common variants
    aliases = {
        "name": ["name", "patient", "patient_name", "full_name"],
        "conditions": ["conditions", "dx", "problems", "medical_history"],
        "medications": ["medications", "meds", "drugs"],
        "procedures": ["procedures", "history_procedures", "surgery", "implants"],
        "diagnostic_reports": ["diagnostic_reports", "reports", "labs", "imaging"],
        "gender": ["gender", "sex"],
        "birthdate": ["birthdate", "dob", "date_of_birth"],
        "address": ["address", "addr"],
    }
    colmap = {}
    for key, opts in aliases.items():
        for o in opts:
            if o in df.columns:
                colmap[key] = o
                break
    # Ensure required
    if "name" not in colmap:
        # pick the first column as name if present
        colmap["name"] = df.columns[0]
    return df, colmap

def _split_multi(val: str) -> List[str]:
    if not isinstance(val, str):
        return []
    parts = re.split(r"[;,\n]\s*", val.strip())
    return [p for p in (x.strip() for x in parts) if p]

def patients_from_xlsx(file) -> List[Dict]:
    try:
        df = pd.read_excel(file)
        df, colmap = _normalize_columns(df)
        results: List[Dict] = []
        for _, row in df.iterrows():
            p: Dict = {}
            p["name"] = str(row.get(colmap.get("name"), "Unknown")).strip() or "Unknown"
            p["gender"] = str(row.get(colmap.get("gender", ""), "")).strip()
            p["birthDate"] = str(row.get(colmap.get("birthdate", ""), "")).strip()
            p["address"] = str(row.get(colmap.get("address", ""), "")).strip()
            p["conditions"] = _split_multi(str(row.get(colmap.get("conditions", ""), "")))
            p["medications"] = _split_multi(str(row.get(colmap.get("medications", ""), "")))
            p["procedures"] = _split_multi(str(row.get(colmap.get("procedures", ""), "")))
            p["diagnostic_reports"] = _split_multi(str(row.get(colmap.get("diagnostic_reports", ""), "")))
            results.append(p)
        return results
    except Exception as e:
        st.error(f"Failed to parse XLSX: {e}")
        return []

# User inputs
patient_name = st.text_input("Enter patient name:")
current_complaint = st.text_area("Enter current complaint or clinical question:")

if st.button("Analyze"):
    if not patient_name or not current_complaint:
        st.warning("Please enter both patient name and current complaint.")
    else:
        with st.spinner("Fetching patient data and analyzing..."):
            if uploaded is not None:
                patients = patients_from_xlsx(uploaded)
                kw = (patient_name or "").lower().strip()
                if kw:
                    patients = [p for p in patients if kw in p.get("name", "").lower()]
            else:
                patients = fetch_patient_data_by_keyword(patient_name)

            if not patients:
                st.info("No patient records found.")
                with st.expander("Create new patient record"):
                    np_name = st.text_input("Full name", key="np_name")
                    np_gender = st.selectbox("Gender", ["", "male", "female", "other"], key="np_gender")
                    np_dob = st.text_input("Birth date (YYYY-MM-DD)", key="np_dob")
                    np_addr = st.text_area("Address", key="np_addr")
                    np_conditions = st.text_area("Conditions (separate with ;)", key="np_cond")
                    np_meds = st.text_area("Medications (separate with ;)", key="np_meds")
                    np_procs = st.text_area("Procedures/Implants (separate with ;)", key="np_procs")
                    np_reports = st.text_area("Diagnostic Reports (separate with ;)", key="np_reports")
                    if st.button("Save new patient"):
                        new_p = {
                            "name": (np_name or "Unknown").strip(),
                            "gender": np_gender,
                            "birthDate": (np_dob or "").strip(),
                            "address": (np_addr or "").strip(),
                            "conditions": [x.strip() for x in (np_conditions or "").split(";") if x.strip()],
                            "medications": [x.strip() for x in (np_meds or "").split(";") if x.strip()],
                            "procedures": [x.strip() for x in (np_procs or "").split(";") if x.strip()],
                            "diagnostic_reports": [x.strip() for x in (np_reports or "").split(";") if x.strip()],
                        }
                        if st.session_state.get("uploaded_bytes_current"):
                            sheets, bts = _upsert_patient_record_uploaded(new_p)
                            st.session_state.uploaded_bytes_current = bts
                            st.success("Patient added to uploaded workbook.")
                            st.download_button(
                                label="Download updated workbook",
                                data=bts,
                                file_name=st.session_state.get("uploaded_name", "patients_updated.xlsx"),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                        else:
                            _upsert_patient_record(new_p)
                            st.success(
                                f"Patient record saved to {os.path.basename(STORAGE_XLSX_PATH)}. Re-run analysis to include this patient."
                            )
            else:
                st.session_state.patients_current = patients
                st.session_state.last_analysis = {}
                for idx, patient in enumerate(patients):
                    pid = f"p{idx}_{patient.get('name','Unknown')}"
                    prompt = generate_prompt(patient, current_complaint)
                    ai_output = call_openrouter_llm(prompt)
                    parsed = parse_ai_output(ai_output)
                    st.session_state.last_analysis[pid] = {
                        "patient": patient,
                        "complaint": current_complaint,
                        "problem": parsed.get("problem", ""),
                        "factors": parsed.get("factors", []),
                        "steps": parsed.get("steps", []),
                        "raw": ai_output,
                    }
                st.session_state.page = "analyze"
                st.session_state.analyze_active = True

# Render analysis/prescription if active
if st.session_state.get("analyze_active"):
    patients = st.session_state.get("patients_current", [])
    for idx, patient in enumerate(patients):
        pid = f"p{idx}_{patient.get('name','Unknown')}"
        data = st.session_state.last_analysis.get(pid)
        if not data:
            # Fallback: compute on-demand if missing
            prompt = generate_prompt(patient, current_complaint)
            ai_output = call_openrouter_llm(prompt)
            parsed = parse_ai_output(ai_output)
            data = {
                "patient": patient,
                "complaint": current_complaint,
                "problem": parsed.get("problem", ""),
                "factors": parsed.get("factors", []),
                "steps": parsed.get("steps", []),
                "raw": ai_output,
            }
            st.session_state.last_analysis[pid] = data

        st.subheader(f"Patient: {patient.get('name', 'Unknown')}")
        st.markdown("**Problem:** " + bold_keywords(data.get("problem", "")))
        if data.get("factors"):
            st.markdown("**Key Relevant Factors:**")
            for f in data["factors"]:
                st.markdown("- " + bold_keywords(f))
        if data.get("steps"):
            st.markdown("**Plan (select steps):**")
            sel = st.session_state.selected.get(pid, [False] * len(data["steps"]))
            with st.form(f"plan_form_{pid}", clear_on_submit=False):
                new_sel = []
                for i, s in enumerate(data["steps"]):
                    checked = st.checkbox(
                        s,
                        value=sel[i] if i < len(sel) else False,
                        key=f"{pid}_step_{i}"
                    )
                    new_sel.append(checked)
                submit = st.form_submit_button("Next → Prescription")
            if submit:
                st.session_state.selected[pid] = new_sel
                if any(new_sel):
                    st.session_state.page = "prescription"
                    st.session_state.active_pid = pid
                else:
                    st.warning("Please select at least one step.")
        st.caption("Why filtering matters: Focus on factors that change decisions and save context window.")
        st.markdown("---")

    if st.session_state.get("page") == "prescription" and st.session_state.get("active_pid"):
        pid = st.session_state.active_pid
        data = st.session_state.last_analysis.get(pid, {})
        if data:
            patient = data["patient"]
            name = patient.get("name", "Unknown")
            problem = data.get("problem", "")
            factors = data.get("factors", [])
            steps = data.get("steps", [])
            sel_mask = st.session_state.selected.get(pid, [False] * len(steps))
            selected_steps = [s for s, m in zip(steps, sel_mask) if m]

            st.header("Prescription / Plan")
            st.markdown(f"**Patient:** {name}")
            st.markdown(f"**Complaint:** {bold_keywords(data.get('complaint',''))}")
            st.markdown(f"**Problem:** {bold_keywords(problem)}")
            if factors:
                st.markdown("**Key Relevant Factors:**")
                for f in factors:
                    st.markdown("- " + bold_keywords(f))
            st.markdown("**Selected Plan:**")
            if selected_steps:
                for i, s in enumerate(selected_steps, 1):
                    st.markdown(f"{i}. {s}")
            else:
                st.info("No steps selected yet.")

            st.subheader("Add Notes / Instructions")
            notes = st.text_area("Notes", key=f"notes_{pid}")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Generate PDF", key=f"pdf_{pid}"):
                    pdf_bytes = make_prescription_pdf(
                        name,
                        data.get("complaint", ""),
                        problem,
                        factors,
                        selected_steps,
                        notes,
                    )
                    if REPORTLAB_AVAILABLE and pdf_bytes:
                        st.download_button(
                            "Download Prescription PDF",
                            data=pdf_bytes,
                            file_name=f"prescription_{name.replace(' ','_')}.pdf",
                            mime="application/pdf",
                        )
                    else:
                        st.warning("PDF module not available. Use browser print (Ctrl+P) to save as PDF.")
            with col2:
                if st.button("Save Visit", key=f"save_{pid}"):
                    if st.session_state.get("uploaded_bytes_current"):
                        _, bts1 = _upsert_patient_record_uploaded(patient)
                        _, bts2 = _append_visit_uploaded(
                            name,
                            data.get("complaint", ""),
                            problem,
                            factors,
                            selected_steps,
                            notes,
                        )
                        st.session_state.uploaded_bytes_current = bts2 or bts1
                        st.success("Visit saved to uploaded workbook.")
                        st.download_button(
                            label="Download updated workbook",
                            data=st.session_state.uploaded_bytes_current,
                            file_name=st.session_state.get("uploaded_name", "patients_updated.xlsx"),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    else:
                        _upsert_patient_record(patient)
                        _append_visit(
                            name,
                            data.get("complaint", ""),
                            problem,
                            factors,
                            selected_steps,
                            notes,
                        )
                        st.success(
                            f"Visit saved to {os.path.basename(STORAGE_XLSX_PATH)}"
                        )
            with col3:
                if st.button("← Back", key=f"back_{pid}"):
                    st.session_state.page = "analyze"
                    # stay in analyze_active to keep results visible