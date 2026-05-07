import os
import re
import io
import hmac
import base64
import logging
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px

try:
    from sympy import sympify, simplify, solve
    from sympy.core.sympify import SympifyError
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

from openai import OpenAI

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ============================================================
# STEMGrade AI — Tutor SaaS MVP v5.0
# Security fixes: secrets-only API key, hmac auth, parallel
# batch grading, typed exception handling, session isolation.
# ============================================================

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("stemgrade")

# ─────────────────────────────────────────
# App config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="STEMGrade AI | Tutor Grading Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_VERSION = "5.0"
LOGO_PATH = Path("assets/logo.png")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@stemgrade.ai")
STRIPE_PAYMENT_LINK = os.getenv("STRIPE_PAYMENT_LINK", "")

# ─────────────────────────────────────────
# Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp { background: #F8FAFC; color: #0F172A; }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    .brand-fallback {
        font-size: 1.65rem;
        font-weight: 900;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }

    .hero {
        background: #1E3A8A;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
    }

    .hero h1 { color: #FFFFFF !important; font-size: 2.4rem; font-weight: 900; margin-bottom: 0.4rem; }
    .hero p  { color: #BFDBFE !important; font-size: 1.1rem; max-width: 820px; margin: 0; }

    .app-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        min-height: 110px;
    }

    .app-card h3 { color: #0F172A !important; font-size: 1.1rem; font-weight: 800; margin: 0 0 0.4rem; }
    .app-card p, .app-card li { color: #475569 !important; font-size: 0.95rem; line-height: 1.5; }

    .pill {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        background: #DBEAFE;
        color: #1E40AF;
        font-weight: 800;
        font-size: 0.8rem;
        margin-bottom: 0.7rem;
    }

    .stat-card {
        background: #F1F5F9;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    .stat-number { font-size: 2rem; font-weight: 900; color: #1E40AF; line-height: 1; }
    .stat-label  { font-size: 0.8rem; color: #64748B; margin-top: 0.2rem; }

    .step-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #2563EB;
        border-radius: 0;
        padding: 0.55rem 0.75rem;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 0.85rem;
        color: #0F172A;
    }

    .step-ok  { border-left-color: #16A34A; background: #F0FDF4; }
    .step-bad { border-left-color: #DC2626; background: #FEF2F2; }

    .pilot-note {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E3A8A;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 9px !important;
        border: 1px solid #1D4ED8 !important;
        background: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        min-height: 2.5rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #1D4ED8 !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="select"] div {
        border-radius: 9px !important;
    }

    h1, h2, h3 { color: #0F172A !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Session state (per-session isolation)
# ─────────────────────────────────────────
DEFAULTS = {
    "authenticated": False,
    "tutor_name": "",
    "current_result": None,
    "batch_results": None,
    "batch_history": [],
    "extracted_solution": "",
    "usage_count": 0,
    "total_graded": 0,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────
# Security helpers
# ─────────────────────────────────────────
def get_access_code() -> str:
    """Read access code from secrets or env — never hardcode."""
    try:
        return st.secrets.get("TUTOR_ACCESS_CODE", "demo-tutor")
    except Exception:
        return os.getenv("TUTOR_ACCESS_CODE", "demo-tutor")


def verify_access_code(provided: str) -> bool:
    """Constant-time comparison to prevent timing attacks."""
    expected = get_access_code()
    return hmac.compare_digest(
        provided.strip().encode("utf-8"),
        expected.encode("utf-8")
    )


def get_openai_key() -> str:
    """
    Read API key from Streamlit secrets or environment variable ONLY.
    Never store in session_state — keeps it off the client.
    """
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")


def get_client() -> Optional[OpenAI]:
    key = get_openai_key()
    if not key:
        return None
    return OpenAI(api_key=key)


# ─────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────
def normalize_math(text: str) -> str:
    text = text.strip().replace("^", "**")
    # Only insert * between digit→letter if not a subscript pattern (x2 stays x2)
    text = re.sub(r"(\d)([a-df-wyzA-Z])", r"\1*\2", text)
    text = re.sub(r"([a-zA-Z])\(", r"\1*(", text)
    return text


def split_steps(solution: str) -> List[str]:
    if not solution:
        return []
    raw = re.split(r";", solution)
    return [normalize_math(s) for s in raw if s.strip()]


def final_answer(solution: str) -> Optional[str]:
    steps = split_steps(solution)
    if not steps:
        return None
    last = steps[-1]
    if "=" in last:
        return last.split("=")[-1].strip().rstrip(".")
    return last.strip().rstrip(".")


def equivalent_answer(predicted: Optional[str], correct: str) -> bool:
    if predicted is None:
        return False
    if not SYMPY_AVAILABLE:
        return str(predicted).strip().lower() == str(correct).strip().lower()
    try:
        return simplify(sympify(str(predicted)) - sympify(str(correct))) == 0
    except (SympifyError, TypeError, AttributeError, ValueError):
        return str(predicted).strip().lower() == str(correct).strip().lower()


def equation_expression(step: str):
    if not SYMPY_AVAILABLE:
        return None
    try:
        step = normalize_math(step)
        if "=" not in step:
            return None
        left, right = step.split("=", 1)
        return simplify(sympify(left) - sympify(right))
    except (SympifyError, TypeError, AttributeError, ValueError, RecursionError):
        return None


def solve_equation_step(step: str):
    if not SYMPY_AVAILABLE:
        return None, []
    try:
        expr = equation_expression(step)
        if expr is None:
            return None, []
        symbols = sorted(list(expr.free_symbols), key=str)
        if len(symbols) != 1:
            return None, []
        variable = symbols[0]
        return variable, [simplify(x) for x in solve(expr, variable)]
    except (SympifyError, TypeError, AttributeError, ValueError, RecursionError):
        return None, []


def valid_transition(prev: str, curr: str) -> bool:
    try:
        prev_var, prev_sols = solve_equation_step(prev)
        curr_var, curr_sols = solve_equation_step(curr)
        if prev_var is None or curr_var is None:
            return True
        if not prev_sols or not curr_sols:
            return True
        if prev_var != curr_var:
            return False
        return {str(simplify(x)) for x in prev_sols} == {str(simplify(x)) for x in curr_sols}
    except (SympifyError, TypeError, AttributeError, ValueError, RecursionError):
        return True  # safe default: don't flag steps we can't verify


def first_wrong_step(steps: List[str]) -> Optional[int]:
    if len(steps) <= 1:
        return None
    for i in range(1, len(steps)):
        if not valid_transition(steps[i - 1], steps[i]):
            return i + 1
    return None


def problem_matches_student_solution(student_steps: List[str], expected_steps: List[str]) -> bool:
    if not student_steps or not expected_steps:
        return False
    try:
        se = equation_expression(student_steps[0])
        ee = equation_expression(expected_steps[0])
        if se is None or ee is None:
            return False
        return simplify(se - ee) == 0
    except (SympifyError, TypeError, AttributeError, ValueError, RecursionError):
        return False


def inconsistent_steps(steps: List[str], answer: Optional[str]) -> List[int]:
    issues = []
    if not steps or answer is None or not SYMPY_AVAILABLE:
        return issues
    try:
        variable, _ = solve_equation_step(steps[0])
        if variable is None:
            return issues
        answer_value = sympify(str(answer))
        for idx, step in enumerate(steps, start=1):
            expr = equation_expression(step)
            if expr is None:
                continue
            if simplify(expr.subs(variable, answer_value)) != 0:
                issues.append(idx)
    except (SympifyError, TypeError, AttributeError, ValueError, RecursionError):
        pass
    return issues


# ─────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────
def score_grading_result(
    final_correct: bool,
    problem_match: bool,
    wrong_step: Optional[int],
    inconsistencies: List[int],
) -> Tuple[int, str, str]:
    if final_correct and problem_match and wrong_step is None and not inconsistencies:
        return 10, "none", "All steps appear correct and the final answer matches."
    if final_correct and problem_match and wrong_step is not None:
        return 7, "reasoning_path_error", "Final answer is correct, but a reasoning step appears invalid."
    if final_correct and not problem_match:
        return 5, "problem_mismatch", "Final answer matches, but the solution may not match the assigned problem."
    if not final_correct and problem_match and wrong_step is None:
        return 6, "final_answer_error", "Reasoning appears mostly valid, but the final answer is incorrect."
    if not final_correct and wrong_step == 2:
        return 4, "early_reasoning_error", "A mistake appears early in the solution."
    if not final_correct and wrong_step is not None:
        return 5, "reasoning_error", "The solution contains a reasoning error."
    return 3, "needs_human_review", "The solution could not be graded reliably — tutor review recommended."


def confidence_score(
    final_correct: bool,
    problem_match: bool,
    wrong_step: Optional[int],
    inconsistencies: List[int],
) -> float:
    score = 0.35 * int(final_correct)
    score += 0.30 * int(problem_match)
    score += 0.20 * int(wrong_step is None)
    score += 0.15 * int(not inconsistencies)
    return round(score, 2)


# ─────────────────────────────────────────
# OpenAI calls (cached where possible)
# ─────────────────────────────────────────
SUBJECT_PROMPTS = {
    "Algebra": "Solve step by step. Return ONLY semicolon-separated steps. Use explicit multiplication like 3*x.",
    "Quadratics": "Solve the quadratic step by step. Return ONLY semicolon-separated steps.",
    "Systems of Equations": "Solve the system step by step. Return ONLY semicolon-separated steps.",
    "Physics": "Solve with formula, substitution, and answer. Return ONLY semicolon-separated steps.",
    "Chemistry": "Balance or solve the chemistry problem. Return ONLY semicolon-separated steps.",
    "Calculus": "Solve the calculus problem step by step. Return ONLY semicolon-separated steps.",
}


@st.cache_data(ttl=3600, show_spinner=False)
def generate_expected_steps(problem: str, subject: str) -> List[str]:
    """Cached — same problem+subject only calls OpenAI once per hour."""
    client = get_client()
    if client is None:
        return []
    instruction = SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS["Algebra"])
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise STEM solution engine. Return only clean semicolon-separated steps.",
                },
                {"role": "user", "content": f"{instruction}\nProblem: {problem}"},
            ],
            temperature=0,
            timeout=30,
        )
        return split_steps(response.choices[0].message.content.strip())
    except Exception as exc:
        logger.error("generate_expected_steps failed: %s", exc)
        return []


def extract_handwritten_solution(
    image_bytes: bytes,
    problem: str,
    subject: str,
    mime_type: str = "image/jpeg",
) -> str:
    client = get_client()
    if client is None:
        st.error("OpenAI API key missing.")
        return ""
    # Validate mime_type to prevent injection
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        mime_type = "image/jpeg"
    try:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Extract the handwritten {subject} solution for: {problem}. "
                                "Return only the student's mathematical steps separated by semicolons. "
                                "Use explicit multiplication like 3*x. No explanations."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
            max_tokens=600,
            timeout=30,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("extract_handwritten_solution failed: %s", exc)
        st.error(f"Handwriting extraction failed: {exc}")
        return ""


# ─────────────────────────────────────────
# Core grading function
# ─────────────────────────────────────────
def grade_solution(problem: str, solution: str, correct_answer: str, subject: str) -> Dict:
    expected = generate_expected_steps(problem, subject)
    steps = split_steps(solution)
    answer = final_answer(solution)
    final_correct = equivalent_answer(answer, correct_answer)
    problem_match = problem_matches_student_solution(steps, expected) if steps and expected else False
    wrong_step = first_wrong_step(steps)
    inconsistencies = inconsistent_steps(steps, answer)
    score, error_type, feedback = score_grading_result(
        final_correct, problem_match, wrong_step, inconsistencies
    )
    confidence = confidence_score(final_correct, problem_match, wrong_step, inconsistencies)

    needs_review = (
        score < 10
        or confidence < 0.90
        or wrong_step is not None
        or bool(inconsistencies)
        or not final_correct
        or not problem_match
    )

    return {
        "score": score,
        "confidence": confidence,
        "needs_review": needs_review,
        "error_type": error_type,
        "feedback": feedback,
        "final_answer": answer,
        "final_correct": final_correct,
        "problem_match": problem_match,
        "first_wrong_step": wrong_step,
        "inconsistent_steps": inconsistencies,
        "expected_steps": expected,
        "student_steps": steps,
    }


def _grade_one_worker(args: Tuple) -> Tuple[str, Dict]:
    """Worker for parallel batch grading."""
    student, solution, problem, correct_answer, subject = args
    result = grade_solution(problem, solution, correct_answer, subject)
    return student, result


# ─────────────────────────────────────────
# Report helpers
# ─────────────────────────────────────────
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def create_pdf_report(title: str, subtitle: str, df: pd.DataFrame, notes: str = "") -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Paragraph(subtitle, styles["Normal"]),
        Spacer(1, 12),
    ]
    if notes:
        story += [Paragraph(notes, styles["BodyText"]), Spacer(1, 12)]

    display_cols = [c for c in ["Student", "Score", "Confidence", "Error Type", "Human Review", "Feedback"] if c in df.columns]
    table_data = [display_cols] + df[display_cols].astype(str).values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ─────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────
def show_logo(width: int = 280) -> None:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=width)
    else:
        st.markdown('<div class="brand-fallback">📘 STEMGrade AI</div>', unsafe_allow_html=True)


def html_card(title: str, body: str) -> None:
    st.markdown(f'<div class="app-card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)


def html_list_card(title: str, items: List[str]) -> None:
    lis = "".join(f"<li>{i}</li>" for i in items)
    st.markdown(f'<div class="app-card"><h3>{title}</h3><ul>{lis}</ul></div>', unsafe_allow_html=True)


def render_result(result: Dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{result['score']}/10")
    c2.metric("Confidence", f"{result['confidence']:.2f}")
    c3.metric("Tutor Review", "Yes ⚠️" if result["needs_review"] else "No ✓")
    c4.metric("Error Type", result["error_type"].replace("_", " ").title())
    st.progress(result["score"] / 10)

    if result["score"] >= 8 and not result["needs_review"]:
        st.success(result["feedback"])
    elif result["score"] >= 6:
        st.warning(result["feedback"])
    else:
        st.error(result["feedback"])

    st.markdown("### Step-by-step review")
    left, right = st.columns(2)

    with left:
        st.markdown("**Expected steps**")
        if result["expected_steps"]:
            for idx, step in enumerate(result["expected_steps"], 1):
                st.markdown(f'<div class="step-box">{idx}. {step}</div>', unsafe_allow_html=True)
        else:
            st.info("Expected steps unavailable.")

    with right:
        st.markdown("**Student steps**")
        if result["student_steps"]:
            fw = result.get("first_wrong_step")
            for idx, step in enumerate(result["student_steps"], 1):
                bad = fw is not None and idx >= fw
                css = "step-box step-bad" if bad else "step-box step-ok"
                label = "Review ✗" if bad else "OK ✓"
                st.markdown(f'<div class="{css}"><b>{label}</b> {idx}. {step}</div>', unsafe_allow_html=True)
        else:
            st.info("No student steps detected.")


# ─────────────────────────────────────────
# Sample data
# ─────────────────────────────────────────
SUBJECTS = ["Algebra", "Quadratics", "Systems of Equations", "Physics", "Chemistry", "Calculus"]

SAMPLE_BY_SUBJECT = {
    "Algebra":               {"problem": "Solve for x: 3*x - 6 = 9",              "answer": "5",       "solution": "3*x - 6 = 9; 3*x = 15; x = 5"},
    "Quadratics":            {"problem": "Solve: x**2 - 5*x + 6 = 0",             "answer": "2",       "solution": "x**2 - 5*x + 6 = 0; (x-2)*(x-3) = 0; x = 2"},
    "Systems of Equations":  {"problem": "Solve: x + y = 5 and x - y = 1",        "answer": "3",       "solution": "x + y = 5; x - y = 1; 2*x = 6; x = 3; y = 2"},
    "Physics":               {"problem": "Find force: mass=5kg, acceleration=10",  "answer": "50",      "solution": "F = m*a; F = 5*10; F = 50"},
    "Chemistry":             {"problem": "Balance: H2 + O2 = H2O",                "answer": "2",       "solution": "H2 + O2 = H2O; 2*H2 + O2 = 2*H2O"},
    "Calculus":              {"problem": "Find derivative of x**2 + 3*x",          "answer": "2*x + 3", "solution": "f(x) = x**2 + 3*x; f_prime(x) = 2*x + 3"},
}


# ─────────────────────────────────────────
# Top nav
# ─────────────────────────────────────────
show_logo(width=260)
st.caption(f"Tutor SaaS MVP · v{APP_VERSION}")

if st.session_state["authenticated"]:
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.success(f"Signed in as **{st.session_state['tutor_name'] or 'Tutor'}**")
    with top_right:
        if st.button("Sign out", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    page = st.radio("Workspace", ["Dashboard", "Grade One", "Batch Grade", "Reports"], horizontal=True)
    subject = st.selectbox("Subject", SUBJECTS, index=0)
    st.markdown("---")
else:
    page = "Landing"
    subject = "Algebra"


# ══════════════════════════════════════════
# LANDING / LOGIN
# ══════════════════════════════════════════
if page == "Landing":
    st.markdown("""
<div class="hero">
  <span class="pill">Tutor-focused MVP</span>
  <h1>Grade handwritten STEM work faster.</h1>
  <p>Upload student solutions, detect the first wrong step, and export review-ready reports in seconds.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="pilot-note">Pilot note: Use anonymized submissions only. Do not upload student names, IDs, or personal information.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        html_card("Step-by-step checking", "Evaluates the reasoning path, not just the final answer.")
    with c2:
        html_card("Human review flag", "Low-confidence work is routed back to the tutor automatically.")
    with c3:
        html_card("Class reports", "Batch grade and export CSV/PDF reports for your records.")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("## Tutor login")
        tutor_name = st.text_input("Your name or center", placeholder="e.g. Vamshi Tutoring")
        access_code = st.text_input("Access code", type="password", placeholder="Enter your access code")
        st.caption("Demo code: `demo-tutor`")
        if st.button("Enter Workspace", use_container_width=True):
            if not access_code.strip():
                st.warning("Please enter an access code.")
            elif verify_access_code(access_code):
                st.session_state["authenticated"] = True
                st.session_state["tutor_name"] = tutor_name.strip() or "Tutor"
                st.rerun()
            else:
                st.error("Invalid access code. Please try again.")

    with right:
        st.markdown("## What you get")
        st.markdown("- Upload or paste student solutions")
        st.markdown("- Review first wrong step and confidence score")
        st.markdown("- Batch grade an entire class at once")
        st.markdown("- Export CSV and PDF grading reports")
        st.markdown("- Human-in-the-loop review flags for low confidence")
        if STRIPE_PAYMENT_LINK:
            st.link_button("Start Paid Pilot →", STRIPE_PAYMENT_LINK)


# ══════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════
elif page == "Dashboard":
    st.markdown("""
<div class="hero">
  <h1>Tutor grading workspace</h1>
  <p>Grade one solution, batch grade a class, and export review-ready reports.</p>
</div>
""", unsafe_allow_html=True)

    latest_df = st.session_state.get("batch_results")
    total_submissions = 0 if latest_df is None else len(latest_df)
    avg_score = 0.0 if latest_df is None or latest_df.empty else latest_df["Score"].mean()
    review_count = 0 if latest_df is None or latest_df.empty else int(latest_df["Human Review"].sum())
    max_score = 0 if latest_df is None or latest_df.empty else latest_df["Score"].max()
    total_ever = st.session_state["total_graded"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total graded (session)", total_ever)
    m2.metric("Latest batch", total_submissions)
    m3.metric("Latest avg score", f"{avg_score:.1f}/10" if total_submissions else "—")
    m4.metric("Need review", review_count if total_submissions else "—")

    if latest_df is not None and not latest_df.empty:
        top_students = latest_df[latest_df["Score"] == max_score]["Student"].tolist()
        st.info(f"Top score ({max_score}/10): {', '.join(top_students)}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        html_list_card("Recommended pilot flow", [
            "Grade 20–30 real anonymized submissions.",
            "Manually review low-confidence cases.",
            "Export report and collect tutor feedback.",
            "Track accuracy and time saved vs manual grading.",
        ])
    with c2:
        html_list_card("Current MVP limits", [
            "Not a replacement for teacher judgment.",
            "Best for algebra-style symbolic problems first.",
            "Needs production auth/database before public launch.",
            "All student data should be anonymized during pilots.",
        ])

    if not get_openai_key():
        st.error("⚠️ No OpenAI API key found. Add `OPENAI_API_KEY` to your Streamlit secrets before grading.")


# ══════════════════════════════════════════
# GRADE ONE
# ══════════════════════════════════════════
elif page == "Grade One":
    st.title("Grade one submission")
    st.caption("Use this for live demos and one-on-one grading.")

    if not get_openai_key():
        st.error("⚠️ OpenAI API key missing. Add it to Streamlit secrets.")
        st.stop()

    sample = SAMPLE_BY_SUBJECT[subject]
    col1, col2 = st.columns(2)
    with col1:
        problem = st.text_input("Problem", value=sample["problem"])
        correct_answer = st.text_input("Correct answer", value=sample["answer"])
    with col2:
        input_method = st.radio("Input method", ["Paste solution", "Upload handwritten image"], horizontal=True)

    solution = ""
    if input_method == "Paste solution":
        solution = st.text_area("Student solution (steps separated by semicolons)", value=sample["solution"], height=140)
    else:
        uploaded = st.file_uploader("Upload handwritten solution", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            st.image(uploaded, width=400)
            mime_type = uploaded.type if uploaded.type in ("image/jpeg", "image/png", "image/webp") else "image/jpeg"
            if st.button("Extract handwriting"):
                with st.spinner("Reading handwritten steps..."):
                    st.session_state["extracted_solution"] = extract_handwritten_solution(
                        uploaded.read(), problem, subject, mime_type=mime_type
                    )
            if st.session_state["extracted_solution"]:
                solution = st.text_area(
                    "Extracted solution — review and edit before grading",
                    st.session_state["extracted_solution"],
                    height=140,
                )

    if st.button("Grade submission", use_container_width=True):
        if not solution.strip():
            st.warning("Please provide a student solution.")
            st.stop()
        with st.spinner("Grading reasoning steps..."):
            result = grade_solution(problem, solution, correct_answer, subject)
            st.session_state["current_result"] = result
            st.session_state["usage_count"] += 1
            st.session_state["total_graded"] += 1

    if st.session_state.get("current_result"):
        st.markdown("---")
        render_result(st.session_state["current_result"])


# ══════════════════════════════════════════
# BATCH GRADE
# ══════════════════════════════════════════
elif page == "Batch Grade":
    st.title("Batch grade a class")
    st.caption("Paste 5–30 anonymized submissions. Graded in parallel — much faster than one-by-one.")

    if not get_openai_key():
        st.error("⚠️ OpenAI API key missing. Add it to Streamlit secrets.")
        st.stop()

    problem = st.text_input("Assignment problem", value="Solve for x: 2*x + 3 = 7")
    correct_answer = st.text_input("Correct answer", value="2")

    st.markdown("### Student submissions")
    st.caption("Format: `StudentName: step1; step2; final answer` — one student per line. Use anonymous IDs.")

    raw = st.text_area(
        "Paste submissions",
        value=(
            "Student A: 2*x + 3 = 7; 2*x = 4; x = 2\n"
            "Student B: 2*x + 3 = 7; 2*x = 10; x = 5\n"
            "Student C: 2*x + 3 = 7; 2*x = 4; x = 2\n"
            "Student D: 2*x + 3 = 7; 2*x = 6; x = 3\n"
            "Student E: 2*x + 3 = 7; 2*x = 4; x = 2"
        ),
        height=200,
    )

    def parse_batch(text: str) -> List[Tuple[str, str]]:
        parsed = []
        for line in text.splitlines():
            line = line.strip()
            if ":" in line:
                name, sol = line.split(":", 1)
                if name.strip() and sol.strip():
                    parsed.append((name.strip(), sol.strip()))
        return parsed

    submissions = parse_batch(raw)
    st.info(f"Detected **{len(submissions)}** submission{'s' if len(submissions) != 1 else ''}.")

    max_workers = min(8, len(submissions)) if submissions else 1
    st.caption(f"Will grade with up to {max_workers} parallel workers.")

    if st.button("Grade batch", use_container_width=True):
        if not submissions:
            st.warning("No valid submissions found. Check your format.")
            st.stop()

        progress_bar = st.progress(0, text="Starting...")
        completed = [0]

        args_list = [
            (student, sol, problem, correct_answer, subject)
            for student, sol in submissions
        ]

        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_student = {executor.submit(_grade_one_worker, args): args[0] for args in args_list}
            for future in concurrent.futures.as_completed(future_to_student):
                try:
                    student, result = future.result()
                    rows.append({
                        "Student": student,
                        "Score": result["score"],
                        "Confidence": result["confidence"],
                        "Error Type": result["error_type"],
                        "Human Review": result["needs_review"],
                        "Final Answer": result["final_answer"],
                        "Feedback": result["feedback"],
                    })
                except Exception as exc:
                    student = future_to_student[future]
                    logger.error("Grading failed for %s: %s", student, exc)
                    rows.append({
                        "Student": student,
                        "Score": 0,
                        "Confidence": 0.0,
                        "Error Type": "grading_error",
                        "Human Review": True,
                        "Final Answer": None,
                        "Feedback": f"Grading error — tutor review required. ({exc})",
                    })
                finally:
                    completed[0] += 1
                    pct = completed[0] / len(submissions)
                    progress_bar.progress(pct, text=f"Graded {completed[0]} of {len(submissions)}...")

        progress_bar.progress(1.0, text="Done!")

        # Sort by student name for consistent display
        rows.sort(key=lambda r: r["Student"])
        df = pd.DataFrame(rows)
        st.session_state["batch_results"] = df
        st.session_state["total_graded"] += len(df)
        st.session_state["batch_history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "subject": subject,
            "problem": problem,
            "count": len(df),
            "average": round(float(df["Score"].mean()), 2),
            "review_count": int(df["Human Review"].sum()),
        })
        st.success(f"Batch complete — {len(df)} students graded.")

    # ── Results display ──
    df = st.session_state.get("batch_results")
    if df is not None and not df.empty:
        st.markdown("---")
        avg = df["Score"].mean()
        max_s = df["Score"].max()
        top = df[df["Score"] == max_s]["Student"].tolist()
        rev = int(df["Human Review"].sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average", f"{avg:.1f}/10")
        m2.metric("Top score", f"{max_s}/10")
        m3.metric("Need review", f"{rev}/{len(df)}")
        m4.metric("Total graded", len(df))

        st.info(f"Top students: {', '.join(top)}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df.sort_values("Score"), x="Student", y="Score",
                         color="Score", color_continuous_scale="Blues",
                         title="Score by student", range_y=[0, 10])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              font_color="#0F172A", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            err = df["Error Type"].value_counts().reset_index()
            err.columns = ["Error Type", "Count"]
            fig2 = px.pie(err, values="Count", names="Error Type", title="Error distribution")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#0F172A")
            st.plotly_chart(fig2, use_container_width=True)

        # Downloads
        st.download_button(
            "Download CSV report",
            data=dataframe_to_csv_bytes(df),
            file_name=f"stemgrade_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        if REPORTLAB_AVAILABLE:
            pdf = create_pdf_report(
                title="STEMGrade AI Batch Report",
                subtitle=f"Subject: {subject} | Problem: {problem} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                df=df,
                notes="AI-assisted grading. Tutor review recommended for all flagged submissions.",
            )
            if pdf:
                st.download_button(
                    "Download PDF report",
                    data=pdf,
                    file_name=f"stemgrade_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning("PDF export requires reportlab. Install with: `pip install reportlab`")


# ══════════════════════════════════════════
# REPORTS
# ══════════════════════════════════════════
elif page == "Reports":
    st.title("Reports")
    st.caption("Export results and review your session history.")

    df = st.session_state.get("batch_results")
    if df is None or df.empty:
        st.info("No batch report yet. Go to Batch Grade first.")
    else:
        st.markdown("### Latest batch report")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download latest CSV",
            data=dataframe_to_csv_bytes(df),
            file_name="stemgrade_latest.csv",
            mime="text/csv",
        )
        if REPORTLAB_AVAILABLE:
            pdf = create_pdf_report(
                title="STEMGrade AI Latest Report",
                subtitle=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                df=df,
                notes="AI-assisted grading. Tutor review recommended for all flagged submissions.",
            )
            if pdf:
                st.download_button(
                    "Download latest PDF",
                    data=pdf,
                    file_name="stemgrade_latest.pdf",
                    mime="application/pdf",
                )
        else:
            st.warning("PDF export requires reportlab. Run: `pip install reportlab`")

    st.markdown("---")
    st.markdown("### Session history")
    history = st.session_state.get("batch_history", [])
    if not history:
        st.info("No batches graded in this session.")
    else:
        hdf = pd.DataFrame(history)
        st.dataframe(hdf, use_container_width=True, hide_index=True)
        st.download_button(
            "Download session history",
            data=dataframe_to_csv_bytes(hdf),
            file_name="stemgrade_history.csv",
            mime="text/csv",
        )
