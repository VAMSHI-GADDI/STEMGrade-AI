import os
import re
import io
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
from sympy import sympify, simplify, solve
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
# STEMGrade AI - Tutor-Focused SaaS MVP
# ============================================================
# What this app does:
# - Private tutor login with access code
# - Grade one STEM solution
# - Batch grade class submissions
# - Detect final answer correctness, reasoning issues, confidence, and review flags
# - Export CSV and PDF reports
#
# Important production note:
# This is still an MVP. Before public/commercial launch, move auth,
# database, billing, student data storage, audit logs, rate limits,
# and privacy controls to a real backend.
# ============================================================


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="STEMGrade AI | Tutor Grading Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "4.0 MVP"
DEFAULT_ACCESS_CODE = os.getenv("TUTOR_ACCESS_CODE", "demo-tutor")
STRIPE_PAYMENT_LINK = os.getenv("STRIPE_PAYMENT_LINK", "")
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@stemgrade.ai")


# -----------------------------
# Professional SaaS styling
# -----------------------------
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    :root {
        --primary: #2563EB;
        --primary-dark: #1E40AF;
        --bg: #F8FAFC;
        --card: #FFFFFF;
        --text: #0F172A;
        --muted: #64748B;
        --border: #E2E8F0;
        --success: #16A34A;
        --warning: #F59E0B;
        --danger: #DC2626;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #0F172A;
        padding: 1rem;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stSelectbox div,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #0F172A !important;
        background: #FFFFFF !important;
    }

    .hero {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #38BDF8 100%);
        border-radius: 24px;
        padding: 3rem 2.2rem;
        color: #FFFFFF;
        margin-bottom: 1.5rem;
        box-shadow: 0 18px 40px rgba(37, 99, 235, 0.20);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        line-height: 1.1;
    }

    .hero p {
        color: #E0F2FE !important;
        font-size: 1.2rem;
        max-width: 850px;
    }

    .app-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
        min-height: 120px;
    }

    .app-card h3 {
        color: #0F172A !important;
        font-size: 1.25rem;
        font-weight: 800;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .app-card p, .app-card li {
        color: #475569 !important;
        font-size: 0.98rem;
        line-height: 1.55;
    }

    .pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #DBEAFE;
        color: #1E40AF;
        font-weight: 800;
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
    }

    .step-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 12px;
        padding: 0.65rem 0.8rem;
        margin: 0.35rem 0;
        font-family: monospace;
        color: #0F172A;
    }

    .step-ok {
        border-left-color: #16A34A;
        background: #F0FDF4;
    }

    .step-bad {
        border-left-color: #DC2626;
        background: #FEF2F2;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px !important;
        border: 1px solid #1D4ED8 !important;
        background: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        min-height: 2.6rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #1D4ED8 !important;
        border-color: #1E40AF !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="select"] div {
        border-radius: 10px !important;
    }

    h1, h2, h3 {
        color: #0F172A !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Session state
# -----------------------------
DEFAULTS = {
    "authenticated": False,
    "tutor_name": "",
    "openai_key": "",
    "current_result": None,
    "batch_results": None,
    "batch_history": [],
    "extracted_solution": "",
    "usage_count": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# -----------------------------
# Config helpers
# -----------------------------
def get_openai_key() -> str:
    """Read API key from session, Streamlit secrets, or environment variable."""
    if st.session_state.get("openai_key"):
        return st.session_state["openai_key"]

    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            st.session_state["openai_key"] = key
            return key
    except Exception:
        pass

    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        st.session_state["openai_key"] = key
        return key

    return ""


def get_client() -> Optional[OpenAI]:
    key = get_openai_key()
    if not key:
        return None
    return OpenAI(api_key=key)


# -----------------------------
# Math and grading helpers
# -----------------------------
def normalize_math(text: str) -> str:
    text = text.strip()
    text = text.replace("^", "**")
    text = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1*\2", text)
    return text


def split_steps(solution: str) -> List[str]:
    if not solution:
        return []
    raw_steps = re.split(r";|\n", solution)
    return [normalize_math(step) for step in raw_steps if step.strip()]


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
    try:
        return simplify(sympify(str(predicted)) - sympify(str(correct))) == 0
    except Exception:
        return str(predicted).strip().lower() == str(correct).strip().lower()


def equation_expression(step: str):
    try:
        step = normalize_math(step)
        if "=" not in step:
            return None
        left, right = step.split("=", 1)
        return simplify(sympify(left) - sympify(right))
    except Exception:
        return None


def solve_equation_step(step: str):
    try:
        expr = equation_expression(step)
        if expr is None:
            return None, []
        symbols = sorted(list(expr.free_symbols), key=str)
        if len(symbols) != 1:
            return None, []
        variable = symbols[0]
        return variable, [simplify(x) for x in solve(expr, variable)]
    except Exception:
        return None, []


def valid_transition(previous_step: str, current_step: str) -> bool:
    try:
        prev_var, prev_solutions = solve_equation_step(previous_step)
        curr_var, curr_solutions = solve_equation_step(current_step)

        # If symbolic validation is not possible, do not automatically mark it wrong.
        if prev_var is None or curr_var is None:
            return True
        if not prev_solutions or not curr_solutions:
            return True
        if prev_var != curr_var:
            return False

        prev_set = {str(simplify(x)) for x in prev_solutions}
        curr_set = {str(simplify(x)) for x in curr_solutions}
        return prev_set == curr_set
    except Exception:
        return True


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
        student_expr = equation_expression(student_steps[0])
        expected_expr = equation_expression(expected_steps[0])
        if student_expr is None or expected_expr is None:
            return False
        return simplify(student_expr - expected_expr) == 0
    except Exception:
        return False


def inconsistent_steps(steps: List[str], answer: Optional[str]) -> List[int]:
    issues = []
    if not steps or answer is None:
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
    except Exception:
        pass
    return issues


SUBJECT_PROMPTS = {
    "Algebra": "Solve step by step. Return ONLY semicolon-separated steps. Use explicit multiplication like 3*x.",
    "Quadratics": "Solve the quadratic step by step. Return ONLY semicolon-separated steps.",
    "Systems of Equations": "Solve the system step by step. Return ONLY semicolon-separated steps.",
    "Physics": "Solve with formula, substitution, and answer. Return ONLY semicolon-separated steps.",
    "Chemistry": "Balance or solve the chemistry problem. Return ONLY semicolon-separated steps.",
    "Calculus": "Solve the calculus problem step by step. Return ONLY semicolon-separated steps.",
}


def generate_expected_steps(problem: str, subject: str) -> List[str]:
    client = get_client()
    if client is None:
        st.error("OpenAI API key is missing. Add it in Settings or Streamlit secrets.")
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
        st.error(f"Could not generate expected steps: {exc}")
        return []


def extract_handwritten_solution(image_bytes: bytes, problem: str, subject: str) -> str:
    client = get_client()
    if client is None:
        st.error("OpenAI API key is missing. Add it in Settings or Streamlit secrets.")
        return ""

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
                                f"Extract the handwritten {subject} solution for this problem: {problem}. "
                                "Return only the student's mathematical steps separated by semicolons. "
                                "Use explicit multiplication like 3*x. Do not add explanations."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                }
            ],
            max_tokens=600,
            timeout=30,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        st.error(f"Handwriting extraction failed: {exc}")
        return ""


def score_grading_result(
    final_correct: bool,
    problem_match: bool,
    wrong_step: Optional[int],
    inconsistencies: List[int],
) -> Tuple[int, str, str]:
    if final_correct and problem_match and wrong_step is None and not inconsistencies:
        return 10, "none", "All steps appear correct and the final answer matches."
    if final_correct and problem_match and wrong_step is not None:
        return 7, "reasoning_path_error", "Final answer is correct, but one reasoning step appears invalid."
    if final_correct and not problem_match:
        return 5, "problem_mismatch", "Final answer matches, but the solution may not match the assigned problem."
    if not final_correct and problem_match and wrong_step is None:
        return 6, "final_answer_error", "Reasoning appears mostly valid, but the final answer is incorrect."
    if not final_correct and wrong_step == 2:
        return 4, "early_reasoning_error", "A mistake appears early in the solution."
    if not final_correct and wrong_step is not None:
        return 5, "reasoning_error", "The solution contains a reasoning error."
    return 3, "needs_human_review", "The solution could not be graded reliably and needs tutor review."


def confidence_score(final_correct: bool, problem_match: bool, wrong_step: Optional[int], inconsistencies: List[int]) -> float:
    score = 0.35 * int(final_correct)
    score += 0.30 * int(problem_match)
    score += 0.20 * int(wrong_step is None)
    score += 0.15 * int(not inconsistencies)
    return round(score, 2)


def grade_solution(problem: str, solution: str, correct_answer: str, subject: str) -> Dict:
    expected = generate_expected_steps(problem, subject)
    steps = split_steps(solution)
    answer = final_answer(solution)
    final_correct = equivalent_answer(answer, correct_answer)
    problem_match = problem_matches_student_solution(steps, expected) if steps and expected else False
    wrong_step = first_wrong_step(steps)
    inconsistencies = inconsistent_steps(steps, answer)
    score, error_type, feedback = score_grading_result(final_correct, problem_match, wrong_step, inconsistencies)
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


# -----------------------------
# Report helpers
# -----------------------------
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def create_pdf_report(title: str, subtitle: str, df: pd.DataFrame, notes: str = "") -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(subtitle, styles["Normal"]))
    story.append(Spacer(1, 12))

    if notes:
        story.append(Paragraph(notes, styles["BodyText"]))
        story.append(Spacer(1, 12))

    display_columns = [
        col
        for col in ["Student", "Score", "Confidence", "Error Type", "Human Review", "Feedback"]
        if col in df.columns
    ]
    table_data = [display_columns] + df[display_columns].astype(str).values.tolist()

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# UI helpers
# -----------------------------
def html_card(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="app-card">
  <h3>{title}</h3>
  <p>{body}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def html_list_card(title: str, items: List[str]) -> None:
    list_items = "".join([f"<li>{item}</li>" for item in items])
    st.markdown(
        f"""
<div class="app-card">
  <h3>{title}</h3>
  <ul>{list_items}</ul>
</div>
""",
        unsafe_allow_html=True,
    )


def render_result(result: Dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score", f"{result['score']}/10")
    col2.metric("Confidence", f"{result['confidence']:.2f}")
    col3.metric("Tutor Review", "Yes" if result["needs_review"] else "No")
    col4.metric("Error Type", result["error_type"].replace("_", " ").title())

    st.progress(result["score"] / 10)

    if result["score"] >= 8 and not result["needs_review"]:
        st.success(result["feedback"])
    elif result["score"] >= 6:
        st.warning(result["feedback"])
    else:
        st.error(result["feedback"])

    st.markdown("### Step Review")
    left, right = st.columns(2)

    with left:
        st.markdown("**Expected Steps**")
        if result["expected_steps"]:
            for idx, step in enumerate(result["expected_steps"], start=1):
                st.markdown(f'<div class="step-box">{idx}. {step}</div>', unsafe_allow_html=True)
        else:
            st.info("Expected steps unavailable.")

    with right:
        st.markdown("**Student Steps**")
        if result["student_steps"]:
            first_wrong = result.get("first_wrong_step")
            for idx, step in enumerate(result["student_steps"], start=1):
                bad = first_wrong is not None and idx >= first_wrong
                css = "step-box step-bad" if bad else "step-box step-ok"
                label = "Review" if bad else "OK"
                st.markdown(f'<div class="{css}"><b>{label}</b> {idx}. {step}</div>', unsafe_allow_html=True)
        else:
            st.info("No student steps detected.")


SUBJECTS = [
    "Algebra",
    "Quadratics",
    "Systems of Equations",
    "Physics",
    "Chemistry",
    "Calculus",
]

SAMPLE_BY_SUBJECT = {
    "Algebra": {
        "problem": "Solve for x: 3*x - 6 = 9",
        "answer": "5",
        "solution": "3*x - 6 = 9; 3*x = 15; x = 5",
    },
    "Quadratics": {
        "problem": "Solve: x**2 - 5*x + 6 = 0",
        "answer": "2",
        "solution": "x**2 - 5*x + 6 = 0; (x-2)*(x-3) = 0; x = 2",
    },
    "Systems of Equations": {
        "problem": "Solve: x + y = 5 and x - y = 1",
        "answer": "3",
        "solution": "x + y = 5; x - y = 1; 2*x = 6; x = 3; y = 2",
    },
    "Physics": {
        "problem": "Find force: mass=5kg, acceleration=10 m/s^2",
        "answer": "50",
        "solution": "F = m*a; F = 5*10; F = 50",
    },
    "Chemistry": {
        "problem": "Balance: H2 + O2 = H2O",
        "answer": "2",
        "solution": "H2 + O2 = H2O; 2*H2 + O2 = 2*H2O",
    },
    "Calculus": {
        "problem": "Find derivative of x**2 + 3*x",
        "answer": "2*x + 3",
        "solution": "f(x) = x**2 + 3*x; f_prime(x) = 2*x + 3",
    },
}


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 📘 STEMGrade AI")
    st.caption(f"Tutor SaaS MVP · v{APP_VERSION}")
    st.markdown("---")

    if st.session_state["authenticated"]:
        st.success(f"Signed in as {st.session_state['tutor_name'] or 'Tutor'}")
        page = st.radio(
            "Workspace",
            ["Dashboard", "Grade One", "Batch Grade", "Reports", "Settings"],
            label_visibility="visible",
        )
        st.markdown("---")
        subject = st.selectbox("Subject", SUBJECTS)
        if st.button("Sign out"):
            for key, value in DEFAULTS.items():
                st.session_state[key] = value
            st.rerun()
    else:
        page = "Landing"
        subject = "Algebra"
        st.info("Sign in to access the tutor workspace.")


# -----------------------------
# Landing / login page
# -----------------------------
if page == "Landing":
    st.markdown(
        """
<div class="hero">
  <span class="pill">Tutor-focused MVP</span>
  <h1>Grade handwritten STEM work faster.</h1>
  <p>STEMGrade AI helps tutors upload student solutions, detect the first wrong step, and generate review-ready feedback reports.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        html_card("Step-by-step checking", "Evaluate the reasoning path, not just the final answer.")
    with c2:
        html_card("Human review flag", "Low-confidence and inconsistent work is routed back to the tutor.")
    with c3:
        html_card("Class reports", "Batch grade submissions and export CSV/PDF reports for review.")

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("## Tutor Login")
        tutor_name = st.text_input("Tutor or center name", placeholder="Example: Vamshi Tutoring")
        access_code = st.text_input("Access code", type="password", placeholder="Enter demo or pilot access code")
        st.caption(
            "For this MVP, set TUTOR_ACCESS_CODE in environment variables or Streamlit secrets. "
            "Default demo code: demo-tutor"
        )
        if st.button("Enter Workspace", use_container_width=True):
            if access_code == DEFAULT_ACCESS_CODE:
                st.session_state["authenticated"] = True
                st.session_state["tutor_name"] = tutor_name.strip() or "Tutor"
                st.rerun()
            else:
                st.error("Invalid access code.")

    with right:
        st.markdown("## Pilot offer")
        st.write("Use this MVP for tutor pilots before building the full SaaS backend.")
        st.markdown("- Upload or paste student solutions")
        st.markdown("- Review first wrong step and confidence")
        st.markdown("- Export CSV/PDF grading reports")
        st.markdown("- Keep human review in the loop")
        if STRIPE_PAYMENT_LINK:
            st.link_button("Start Paid Pilot", STRIPE_PAYMENT_LINK)


# -----------------------------
# Dashboard
# -----------------------------
elif page == "Dashboard":
    st.markdown(
        """
<div class="hero">
  <h1>Tutor grading workspace</h1>
  <p>Grade one solution, batch grade a class, and export review-ready reports.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    latest_df = st.session_state.get("batch_results")
    total_batches = len(st.session_state["batch_history"])
    total_submissions = 0 if latest_df is None else len(latest_df)
    avg_score = 0.0 if latest_df is None or latest_df.empty else latest_df["Score"].mean()
    review_count = 0 if latest_df is None or latest_df.empty else int(latest_df["Human Review"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Submissions", total_submissions)
    m2.metric("Latest Average", f"{avg_score:.1f}/10" if total_submissions else "—")
    m3.metric("Need Review", review_count if total_submissions else "—")
    m4.metric("Batches", total_batches)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        html_list_card(
            "Recommended selling flow",
            [
                "Grade 20-30 real anonymized submissions.",
                "Review low-confidence cases manually.",
                "Export report and ask tutor for feedback.",
                "Track accuracy and time saved.",
            ],
        )
    with c2:
        html_list_card(
            "Current MVP limits",
            [
                "Not a replacement for teacher judgment.",
                "Best for algebra-style symbolic work first.",
                "Needs production auth/database before public launch.",
                "Student data should be anonymized during pilots.",
            ],
        )


# -----------------------------
# Grade One
# -----------------------------
elif page == "Grade One":
    st.title("Grade One Student Submission")
    st.caption("Use this for live tutor demos and one-on-one grading.")

    if not get_openai_key():
        st.warning("OpenAI API key is missing. Add it in Settings or Streamlit secrets before grading.")

    sample = SAMPLE_BY_SUBJECT[subject]
    col1, col2 = st.columns([1, 1])
    with col1:
        problem = st.text_input("Problem", value=sample["problem"])
        correct_answer = st.text_input("Correct answer", value=sample["answer"])
    with col2:
        input_method = st.radio("Input method", ["Paste solution", "Upload handwritten image"], horizontal=True)

    solution = ""
    if input_method == "Paste solution":
        solution = st.text_area("Student solution", value=sample["solution"], height=150)
    else:
        uploaded = st.file_uploader("Upload handwritten solution", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, width=420)
            if st.button("Extract Handwriting"):
                with st.spinner("Extracting handwritten steps..."):
                    st.session_state["extracted_solution"] = extract_handwritten_solution(uploaded.read(), problem, subject)
            if st.session_state["extracted_solution"]:
                solution = st.text_area(
                    "Extracted solution - review/edit before grading",
                    st.session_state["extracted_solution"],
                    height=150,
                )

    if st.button("Grade Submission", use_container_width=True):
        if not solution.strip():
            st.warning("Please provide a student solution.")
            st.stop()
        with st.spinner("Grading reasoning steps..."):
            result = grade_solution(problem, solution, correct_answer, subject)
            st.session_state["current_result"] = result
            st.session_state["usage_count"] += 1

    if st.session_state.get("current_result"):
        st.markdown("---")
        render_result(st.session_state["current_result"])


# -----------------------------
# Batch Grade
# -----------------------------
elif page == "Batch Grade":
    st.title("Batch Grade a Class")
    st.caption("Best for tutor pilots: paste 5-30 anonymized submissions and export a report.")

    if not get_openai_key():
        st.warning("OpenAI API key is missing. Add it in Settings or Streamlit secrets before grading.")

    problem = st.text_input("Assignment problem", value="Solve for x: 2*x + 3 = 7")
    correct_answer = st.text_input("Correct answer", value="2")

    st.markdown("### Student submissions")
    st.caption("Format: StudentName: step 1; step 2; final answer. Use anonymous IDs for pilots.")
    raw = st.text_area(
        "Paste submissions",
        value=(
            "Student A: 2*x + 3 = 7; 2*x = 4; x = 2\n"
            "Student B: 2*x + 3 = 7; 2*x = 10; x = 5\n"
            "Student C: 2*x + 3 = 7; 2*x = 4; x = 2\n"
            "Student D: 2*x + 3 = 7; 2*x = 6; x = 3"
        ),
        height=210,
    )

    def parse_batch(text: str) -> List[Tuple[str, str]]:
        parsed = []
        for line in text.splitlines():
            if ":" in line:
                name, sol = line.split(":", 1)
                if name.strip() and sol.strip():
                    parsed.append((name.strip(), sol.strip()))
        return parsed

    submissions = parse_batch(raw)
    st.info(f"Detected {len(submissions)} submissions.")

    if st.button("Grade Batch", use_container_width=True):
        if not submissions:
            st.warning("No valid submissions found.")
            st.stop()

        rows = []
        progress = st.progress(0)
        status = st.empty()

        for index, (student, student_solution) in enumerate(submissions, start=1):
            status.text(f"Grading {student} ({index}/{len(submissions)})...")
            result = grade_solution(problem, student_solution, correct_answer, subject)
            rows.append(
                {
                    "Student": student,
                    "Score": result["score"],
                    "Confidence": result["confidence"],
                    "Error Type": result["error_type"],
                    "Human Review": result["needs_review"],
                    "Final Answer": result["final_answer"],
                    "Feedback": result["feedback"],
                }
            )
            progress.progress(index / len(submissions))

        df = pd.DataFrame(rows)
        st.session_state["batch_results"] = df
        st.session_state["batch_history"].append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "subject": subject,
                "problem": problem,
                "count": len(df),
                "average": float(df["Score"].mean()),
                "review_count": int(df["Human Review"].sum()),
            }
        )
        status.success("Batch grading complete.")

    df = st.session_state.get("batch_results")
    if df is not None and not df.empty:
        st.markdown("---")
        avg = df["Score"].mean()
        top_student = df.loc[df["Score"].idxmax(), "Student"]
        review_count = int(df["Human Review"].sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average", f"{avg:.1f}/10")
        m2.metric("Top Student", top_student)
        m3.metric("Need Review", f"{review_count}/{len(df)}")
        m4.metric("Total", len(df))

        st.dataframe(df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df, x="Student", y="Score", title="Scores", range_y=[0, 10])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#0F172A")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            error_counts = df["Error Type"].value_counts().reset_index()
            error_counts.columns = ["Error Type", "Count"]
            fig2 = px.pie(error_counts, values="Count", names="Error Type", title="Error Distribution")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#0F172A")
            st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "Download CSV Report",
            data=dataframe_to_csv_bytes(df),
            file_name=f"stemgrade_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        pdf_bytes = create_pdf_report(
            title="STEMGrade AI Batch Report",
            subtitle=(
                f"Subject: {subject} | Problem: {problem} | "
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            ),
            df=df,
            notes="AI-assisted grading report. Tutor review is recommended for all flagged submissions.",
        )
        if pdf_bytes:
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name=f"stemgrade_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("PDF export requires reportlab. Install it with: pip install reportlab")


# -----------------------------
# Reports
# -----------------------------
elif page == "Reports":
    st.title("Reports")
    st.caption("Export results and review pilot history.")

    df = st.session_state.get("batch_results")
    if df is None or df.empty:
        st.info("No batch report yet. Grade a batch first.")
    else:
        st.markdown("### Latest Batch Report")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Latest CSV",
            data=dataframe_to_csv_bytes(df),
            file_name="latest_stemgrade_report.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("### Pilot History")
    history = st.session_state.get("batch_history", [])
    if not history:
        st.info("No pilot batches recorded in this session.")
    else:
        hdf = pd.DataFrame(history)
        st.dataframe(hdf, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Pilot History",
            data=dataframe_to_csv_bytes(hdf),
            file_name="stemgrade_pilot_history.csv",
            mime="text/csv",
        )


# -----------------------------
# Settings
# -----------------------------
elif page == "Settings":
    st.title("Settings")
    st.caption("Configure the MVP for private pilots.")

    st.markdown("### API Configuration")
    existing_key = get_openai_key()
    st.write("OpenAI status:", "Connected" if existing_key else "Missing")
    key_input = st.text_input("OpenAI API key", type="password", placeholder="sk-...", value="")
    if st.button("Save API Key"):
        if key_input.strip():
            st.session_state["openai_key"] = key_input.strip()
            st.success("API key saved for this session.")
        else:
            st.warning("Enter a valid key.")

    st.markdown("---")
    st.markdown("### Billing")
    if STRIPE_PAYMENT_LINK:
        st.success("Stripe payment link configured.")
        st.link_button("Open Payment Link", STRIPE_PAYMENT_LINK)
    else:
        st.info("Set STRIPE_PAYMENT_LINK in environment variables when you are ready for paid pilots.")

    st.markdown("---")
    st.markdown("### Privacy Defaults")
    st.markdown(
        """
- Use anonymous student IDs during pilots.
- Do not upload sensitive personal student information.
- Review AI grades before sharing them with students.
- Add production database, encryption, audit logs, deletion workflow, and legal pages before public launch.
        """
    )

    st.markdown("---")
    st.markdown("### Deployment notes")
    st.code(
        """
# requirements.txt
streamlit
pandas
plotly
sympy
openai
reportlab

# Streamlit secrets example
OPENAI_API_KEY = "paste_your_key_in_streamlit_secrets_not_github"
TUTOR_ACCESS_CODE = "choose_private_demo_code"
STRIPE_PAYMENT_LINK = "optional_stripe_payment_link"
SUPPORT_EMAIL = "your_support_email"
        """.strip(),
        language="toml",
    )
