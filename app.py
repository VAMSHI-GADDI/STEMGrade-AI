import os, re, base64, time, smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import pandas as pd
import plotly.express as px
from sympy import sympify, simplify, solve
from openai import OpenAI

st.set_page_config(
    page_title="STEMGrade AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {background-color: #39FF14 !important;}

.stApp {background-color: #ffffff;}

[data-testid="stSidebar"] {
    background-color: #39FF14;
    padding: 10px;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #000000 !important;
    font-weight: 600;
}
[data-testid="stSidebar"] input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] div {
    background-color: #ffffff !important;
    color: #000000 !important;
}

.stButton > button {
    background-color: #39FF14 !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}
.stButton > button:hover {
    background-color: #28cc0f !important;
}

.stDownloadButton > button {
    background-color: #39FF14 !important;
    color: #000000 !important;
    border: 2px solid #000000 !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}

.stTextInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #39FF14 !important;
    border-radius: 8px !important;
}
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #39FF14 !important;
    border-radius: 8px !important;
}

[data-testid="stMetricValue"] {color: #000000 !important; font-weight: bold !important;}
[data-testid="stMetricLabel"] {color: #333333 !important;}

.stProgress > div > div > div {
    background-color: #39FF14 !important;
}

h1, h2, h3 {color: #000000 !important; font-weight: 700 !important;}
p, label, span {color: #000000 !important;}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──
DEFAULTS = {
    "logged_in": False,
    "user_role": None,
    "user_name": None,
    "api_key": "",
    "student_submissions": [],
    "class_results": None,
    "extracted_solution": "",
    "question_bank": [],
    "timer_running": False,
    "timer_start": None,
    "timer_duration": 600,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── API KEY ──
def get_api_key():
    if st.session_state["api_key"]:
        return st.session_state["api_key"]
    try:
        k = st.secrets["OPENAI_API_KEY"]
        if k:
            st.session_state["api_key"] = k
            return k
    except Exception:
        pass
    k = os.getenv("OPENAI_API_KEY", "")
    if k:
        st.session_state["api_key"] = k
        return k
    return ""

# ── USERS ──
USERS = {
    "teacher":  {"password": "teach@123",  "role": "teacher"},
    "vamshi":   {"password": "vamshi@123", "role": "teacher"},
    "student1": {"password": "study@123",  "role": "student"},
    "student2": {"password": "study@123",  "role": "student"},
    "student3": {"password": "study@123",  "role": "student"},
    "alice":    {"password": "alice@123",  "role": "student"},
    "bob":      {"password": "bob@123",    "role": "student"},
    "carol":    {"password": "carol@123",  "role": "student"},
    "david":    {"password": "david@123",  "role": "student"},
    "demo":     {"password": "demo@123",   "role": "student"},
}

def do_login(username, password):
    u = username.strip().lower()
    if u in USERS and USERS[u]["password"] == password:
        st.session_state["logged_in"] = True
        st.session_state["user_role"] = USERS[u]["role"]
        st.session_state["user_name"] = u
        return True
    return False

def do_logout():
    for k in ["logged_in", "user_role", "user_name"]:
        st.session_state[k] = DEFAULTS[k]

# ── SIDEBAR ──
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=60)
    st.title("STEMGrade AI")
    st.markdown("---")

    if st.session_state["logged_in"]:
        st.write(f"**{st.session_state['user_role'].title()}: {st.session_state['user_name'].title()}**")
        if st.button("Logout"):
            do_logout()
            st.rerun()
        st.markdown("---")

        api_key = get_api_key()
        if api_key:
            st.success("OpenAI connected!")
        else:
            st.warning("No API key found.")
            k = st.text_input("Enter OpenAI API Key", type="password", placeholder="sk-...")
            if k:
                st.session_state["api_key"] = k
                st.rerun()

        st.markdown("---")
        if st.session_state["user_role"] == "teacher":
            page = st.radio("Navigate", [
                "Home", "Single Student", "Teacher Dashboard",
                "Question Bank", "Student Submissions",
                "Leaderboard", "Analytics", "About"
            ])
        else:
            page = st.radio("Navigate", [
                "Home", "Submit Solution", "My Results", "Leaderboard", "About"
            ])
        st.markdown("---")
        subject = st.selectbox("Subject", [
            "Algebra", "Quadratics", "Systems of Equations",
            "Physics", "Chemistry", "Calculus"
        ])
    else:
        page = st.radio("Navigate", ["Home", "Login", "About"])
        subject = "Algebra"

    st.caption("STEMGrade AI v3.0")

# Build client AFTER sidebar so api_key is up to date
api_key = get_api_key()
client = OpenAI(api_key=api_key) if api_key else None

# ── GRADING FUNCTIONS ──
def norm(step):
    return re.sub(r"(\d)([a-zA-Z])", r"\1*\2", step)

def split_steps(sol):
    return [norm(s.strip()) for s in sol.split(";") if s.strip()]

def get_final_answer(sol):
    try:
        last = split_steps(sol)[-1]
        return last.split("=")[-1].strip() if "=" in last else last.strip()
    except Exception:
        return None

def check_answer(pred, correct):
    try:
        return simplify(sympify(str(pred)) - sympify(str(correct))) == 0
    except Exception:
        return str(pred).strip() == str(correct).strip()

def get_prompt(subj, problem):
    p = {
        "Algebra": f"Solve step by step. Return ONLY semicolon-separated steps. Write 3*x not 3x. Problem: {problem} Example: 2*x+3=7; 2*x=4; x=2",
        "Quadratics": f"Solve quadratic. Return ONLY semicolon-separated steps. Problem: {problem} Example: x**2-5*x+6=0; (x-2)*(x-3)=0; x=2",
        "Systems of Equations": f"Solve system. Return ONLY semicolon-separated steps. Problem: {problem} Example: x+y=5; x-y=1; 2*x=6; x=3; y=2",
        "Physics": f"Solve. Show formula, substitution, answer. Return ONLY semicolon-separated steps. Problem: {problem} Example: F=m*a; F=5*10; F=50",
        "Chemistry": f"Balance/solve. Return ONLY semicolon-separated steps. Problem: {problem} Example: 2*H2+O2=2*H2O",
        "Calculus": f"Solve. Return ONLY semicolon-separated steps. Problem: {problem} Example: f(x)=x**2+3*x; f_prime(x)=2*x+3",
    }
    return p.get(subj, p["Algebra"])

def gen_steps(problem, subj):
    if not client:
        st.error("No API key. Enter it in the sidebar.")
        return []
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a precise {subj} solver. Return ONLY semicolon-separated steps."},
                {"role": "user", "content": get_prompt(subj, problem)}
            ],
            temperature=0,
            timeout=30
        )
        return split_steps(r.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return []

def read_image(img_bytes, problem, subj):
    if not client:
        st.error("No API key. Enter it in the sidebar.")
        return ""
    try:
        b64 = base64.b64encode(img_bytes).decode()
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"Handwritten {subj} solution for: {problem}. Extract steps only. Separate with semicolons. Write 3*x not 3x."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]}],
            max_tokens=500,
            timeout=30
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Image reading failed: {e}")
        return ""

def norm_eq(step):
    try:
        step = norm(step)
        if "=" not in step:
            return None
        l, r = step.split("=", 1)
        return simplify(sympify(l) - sympify(r))
    except Exception:
        return None

def solve_step(step):
    try:
        expr = norm_eq(step)
        if expr is None:
            return None, []
        syms = sorted(list(expr.free_symbols), key=str)
        if len(syms) != 1:
            return None, []
        v = syms[0]
        return v, [simplify(s) for s in solve(expr, v)]
    except Exception:
        return None, []

def valid_trans(a, b):
    try:
        av, ap = solve_step(a)
        bv, bp = solve_step(b)
        if av is None or bv is None:
            return True
        if not ap or not bp:
            return True
        if av != bv:
            return False
        a_sols = {str(simplify(s)) for s in ap}
        b_sols = {str(simplify(s)) for s in bp}
        return a_sols == b_sols
    except Exception:
        return True

def first_wrong(steps):
    if len(steps) <= 1:
        return None

    last = steps[-1]
    final_val = None
    try:
        if "=" in last:
            final_val = sympify(last.split("=")[-1].strip())
    except Exception:
        pass

    for i in range(1, len(steps)):
        if not valid_trans(steps[i - 1], steps[i]):
            try:
                v, sols = solve_step(steps[i - 1])
                if v is not None and sols and final_val is not None:
                    if any(simplify(s - final_val) == 0 for s in sols):
                        continue
            except Exception:
                pass
            return i + 1
    return None

def match_prob(stu, exp):
    try:
        se = norm_eq(stu[0])
        ee = norm_eq(exp[0])
        if se is None or ee is None:
            return False
        return simplify(se - ee) == 0
    except Exception:
        return False

def incon(steps, ans):
    out = []
    try:
        fv, _ = solve_step(steps[0])
        if fv is None:
            return out
        fval = sympify(str(ans))
        for i, s in enumerate(steps, 1):
            e = norm_eq(s)
            if e is None or simplify(e.subs(fv, fval)) != 0:
                out.append(i)
    except Exception:
        pass
    return out

def score_result(fc, pm, fw, inc):
    if fc and pm and fw is None and not inc:
        return 10, "none", "All steps correct and final answer matches."
    if fc and pm and fw is not None:
        return 7, "reasoning_path_error", "Final answer correct but reasoning has an invalid step."
    if fc and not pm:
        return 5, "problem_mismatch", "Final answer correct but does not match the problem."
    if not fc and pm and fw is None:
        return 6, "final_answer_error", "Reasoning valid but final answer is incorrect."
    if not fc and fw == 2:
        return 4, "early_reasoning_error", "Mistake occurs early in the solution."
    if not fc and fw is not None:
        return 5, "reasoning_error", "Solution contains a reasoning error."
    return 3, "needs_human_review", "Could not be reliably graded automatically."

def calc_conf(fc, pm, fw, inc):
    return round(0.35 * fc + 0.30 * pm + 0.20 * (fw is None) + 0.15 * (not inc), 2)

def run_grade(problem, solution, correct, subj):
    exp = gen_steps(problem, subj)
    steps = split_steps(solution)
    ans = get_final_answer(solution)
    fc = check_answer(ans, correct)
    pm = match_prob(steps, exp) if exp and steps else False
    fw = first_wrong(steps)
    inc = incon(steps, ans) if steps else []
    sc, et, fb = score_result(fc, pm, fw, inc)
    conf = calc_conf(fc, pm, fw, inc)
    return {
        "score": sc,
        "error_type": et,
        "feedback": fb,
        "first_wrong": fw,
        "inconsistent": inc,
        "final_answer": ans,
        "final_correct": fc,
        "problem_match": pm,
        "confidence": conf,
        "needs_review": sc < 10 or conf < 0.9 or fw or inc or not fc or not pm,
        "expected": exp,
        "steps": steps
    }

def show_results(r):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{r['score']}/10")
    c2.metric("Confidence", r["confidence"])
    c3.metric("Review", "Yes" if r["needs_review"] else "No")
    c4.metric("Error", r["error_type"])
    st.progress(r["score"] / 10)

    if r["score"] == 10:
        st.success(r["feedback"])
    elif r["score"] >= 6:
        st.warning(r["feedback"])
    else:
        st.error(r["feedback"])

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Expected Steps:**")
        for i, s in enumerate(r["expected"], 1):
            st.markdown(
                f'''<div style="background:#f0f0f0;padding:7px 12px;border-radius:6px;
margin:3px 0;font-family:monospace;border-left:4px solid #39FF14;color:#000">{i}. {s}</div>''',
                unsafe_allow_html=True
            )
    with c2:
        st.markdown("**Student Steps:**")
        for i, s in enumerate(r["steps"], 1):
            wrong = r["first_wrong"] and i >= r["first_wrong"]
            bg = "#ffebee" if wrong else "#e8f5e9"
            bdr = "#f44336" if wrong else "#4caf50"
            lbl = "WRONG" if wrong else "OK"
            st.markdown(
                f'''<div style="background:{bg};padding:7px 12px;border-radius:6px;
margin:3px 0;font-family:monospace;border-left:4px solid {bdr};color:#000">{lbl} {i}. {s}</div>''',
                unsafe_allow_html=True
            )

SAMPLES = {
    "Algebra": ("Solve for x: 3*x - 6 = 9", "5", "3*x - 6 = 9; 3*x = 15; x = 5"),
    "Quadratics": ("Solve: x**2 - 5*x + 6 = 0", "2", "x**2 - 5*x + 6 = 0; (x-2)*(x-3) = 0; x = 2"),
    "Systems of Equations": ("Solve: x + y = 5 and x - y = 1", "3", "x + y = 5; x - y = 1; 2*x = 6; x = 3; y = 2"),
    "Physics": ("Find force: mass=5kg, acceleration=10 m/s^2", "50", "F = m*a; F = 5*10; F = 50"),
    "Chemistry": ("Balance: H2 + O2 = H2O", "2", "H2 + O2 = H2O; 2*H2 + O2 = 2*H2O"),
    "Calculus": ("Find derivative of x**2 + 3*x", "2*x + 3", "f(x) = x**2 + 3*x; f_prime(x) = 2*x + 3"),
}

# ══════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════

if page == "Home":
    st.markdown("""
<div style="background:linear-gradient(135deg,#39FF14,#00cc00);padding:48px 30px;
border-radius:20px;text-align:center;margin-bottom:24px">
<h1 style="color:#000;font-size:2.8em;font-weight:900;margin:0">STEMGrade AI</h1>
<p style="color:#000;font-size:1.2em;margin-top:10px">
AI-Powered STEM Homework Grader — Grade smarter, not harder.</p>
</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, title, sub in zip(
        [c1, c2, c3, c4],
        ["✍️", "🧮", "👨‍🏫", "📊"],
        ["Handwriting", "Step-by-Step", "Dashboard", "Analytics"],
        ["Read & Grade", "Analysis", "Whole Class", "& Reports"]
    ):
        with col:
            st.markdown(f"""
<div style="background:#fff;padding:24px 16px;border-radius:16px;text-align:center;
box-shadow:0 4px 16px rgba(0,0,0,0.08);border-top:4px solid #39FF14;border:1px solid #eee">
<div style="font-size:2em">{icon}</div>
<b style="color:#000;display:block;margin-top:8px">{title}</b>
<span style="color:#555;font-size:0.9em">{sub}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### How it works
1. Login with your account
2. Enter the problem and correct answer
3. Type or upload a photo of the solution
4. AI grades step by step instantly
5. Download reports for your whole class
        """)
    with c2:
        st.markdown("""
### Subjects Supported
- Algebra — Linear equations
- Quadratics — Factoring, quadratic formula
- Systems of Equations
- Physics — Force, velocity, energy
- Chemistry — Balancing equations
- Calculus — Derivatives, integrals
        """)
    st.markdown("---")
    if not st.session_state["logged_in"]:
        st.info("Please Login from the sidebar to start grading.")
    else:
        st.success(f"Welcome back, {st.session_state['user_name'].title()}!")

elif page == "Login":
    st.title("Login to STEMGrade AI")
    st.markdown("---")
    st.markdown("""
<div style="background:#f9f9f9;padding:32px;border-radius:16px;
border:2px solid #39FF14;max-width:480px;margin:0 auto">
<h3 style="color:#000;margin-top:0">Sign In</h3>
</div>""", unsafe_allow_html=True)
    uname = st.text_input("Username", placeholder="Enter your username")
    pword = st.text_input("Password", type="password", placeholder="Enter your password")
    clicked = st.button("Login", width="stretch")
    if clicked:
        if not uname or not pword:
            st.warning("Please enter both username and password.")
        elif do_login(uname, pword):
            st.success(f"Welcome, {uname.title()}!")
            st.rerun()
        else:
            st.error("Invalid username or password.")
    st.markdown("---")
    st.markdown("**Demo Credentials:**")
    st.markdown("""
| Role    | Username | Password    |
|---------|----------|-------------|
| Teacher | teacher  | teach@123   |
| Teacher | vamshi   | vamshi@123  |
| Student | alice    | alice@123   |
| Student | bob      | bob@123     |
| Student | carol    | carol@123   |
| Student | david    | david@123   |
| Student | student1 | study@123   |
| Demo    | demo     | demo@123    |
    """)

elif page == "About":
    st.title("About STEMGrade AI")
    st.markdown("""
<div style="background:linear-gradient(135deg,#39FF14,#00cc00);padding:40px 30px;
border-radius:20px;text-align:center;margin-bottom:24px">
<h1 style="color:#000;font-size:2.5em;font-weight:900;margin:0">STEMGrade AI</h1>
<p style="color:#000;font-size:1.1em;margin-top:10px">
Built to make STEM grading smarter, faster, and fairer.</p>
</div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### Mission
Most grading systems only check the final answer.
STEMGrade AI evaluates HOW students think — step by step.

### Tech Stack
- Python + Streamlit
- OpenAI GPT-4o
- SymPy for symbolic math
- Plotly for analytics
        """)
    with c2:
        st.markdown("""
### Subjects
Algebra, Quadratics, Systems of Equations,
Physics, Chemistry, Calculus

### Author
**Vamshi Gaddi**
github.com/VAMSHI-GADDI/STEMGrade-AI
        """)

elif page == "Submit Solution":
    st.title("Submit Your Solution")
    st.markdown(f"""<div style="display:inline-block;background:#39FF14;color:#000;
padding:5px 16px;border-radius:20px;font-weight:bold;border:1px solid #000;margin-bottom:12px">
Student: {st.session_state['user_name'].title()}</div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### Timer")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        dur = st.selectbox("Time Limit (min)", [5, 10, 15, 20, 30, 45, 60], index=1)
        st.session_state["timer_duration"] = dur * 60
    with tc2:
        if st.button("Start Timer"):
            st.session_state["timer_running"] = True
            st.session_state["timer_start"] = time.time()
    with tc3:
        if st.button("Stop Timer"):
            st.session_state["timer_running"] = False

    if st.session_state["timer_running"] and st.session_state["timer_start"]:
        rem = max(0, st.session_state["timer_duration"] - (time.time() - st.session_state["timer_start"]))
        col = "#39FF14" if rem > 60 else "#FF4444"
        st.markdown(f"""<div style="background:#000;color:{col};padding:14px;border-radius:10px;
font-size:2.4em;font-weight:bold;text-align:center;font-family:monospace;
border:3px solid {col};letter-spacing:4px;margin:8px 0">
{int(rem//60):02d}:{int(rem%60):02d}</div>""", unsafe_allow_html=True)
        if rem == 0:
            st.error("Time is up!")
            st.session_state["timer_running"] = False

    st.markdown("---")
    assigned = [q for q in st.session_state["question_bank"] if q.get("subject") == subject]
    if assigned:
        opts = [f"{i+1}. {q['problem']}" for i, q in enumerate(assigned)]
        sel = st.selectbox("Select assigned problem", opts)
        idx = int(sel.split(".")[0]) - 1
        problem = assigned[idx]["problem"]
        correct_answer = assigned[idx]["answer"]
        st.info(f"Problem: {problem}")
    else:
        smp = SAMPLES.get(subject, SAMPLES["Algebra"])
        problem = st.text_input("Problem", smp[0])
        correct_answer = st.text_input("Correct Answer", smp[1])

    method = st.radio("Input Method", ["Type solution", "Upload handwritten image"])
    sol = ""
    if method == "Type solution":
        smp = SAMPLES.get(subject, SAMPLES["Algebra"])
        sol = st.text_area("Your Solution", smp[2], height=130)
    else:
        f = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if f:
            st.image(f, width=380)
            if st.button("Read Handwriting"):
                with st.spinner("Reading..."):
                    result = read_image(f.read(), problem, subject)
                if result:
                    st.success("Done!")
                    st.session_state["extracted_solution"] = result
        if st.session_state["extracted_solution"]:
            sol = st.text_area("Extracted (edit if needed)", st.session_state["extracted_solution"], height=130)

    if st.button("Submit and Grade", width="stretch"):
        if not api_key:
            st.error("No API key.")
            st.stop()
        if not sol.strip():
            st.warning("Enter your solution.")
            st.stop()
        with st.spinner("Grading..."):
            r = run_grade(problem, sol, correct_answer, subject)
        st.session_state["student_submissions"].append({
            "student": st.session_state["user_name"],
            "subject": subject,
            "problem": problem,
            "solution": sol,
            "score": r["score"],
            "confidence": r["confidence"],
            "error_type": r["error_type"],
            "feedback": r["feedback"],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        st.markdown("---")
        show_results(r)
        st.session_state["timer_running"] = False

elif page == "My Results":
    st.title("My Results")
    mine = [s for s in st.session_state["student_submissions"] if s["student"] == st.session_state["user_name"]]
    if not mine:
        st.info("No submissions yet.")
    else:
        df = pd.DataFrame(mine)
        c1, c2, c3 = st.columns(3)
        c1.metric("Submissions", len(df))
        c2.metric("Average", f"{df['score'].mean():.1f}/10")
        c3.metric("Best", f"{df['score'].max()}/10")
        st.markdown("---")
        for _, row in df.iterrows():
            bc = "#c8e6c9" if row["score"] >= 8 else "#fff9c4" if row["score"] >= 5 else "#ffcdd2"
            st.markdown(f"""<div style="background:{bc};padding:14px 18px;border-radius:10px;
margin:6px 0;border-left:4px solid #39FF14;color:#000">
<b>{row['time']}</b> — {row['subject']} — Score: <b>{row['score']}/10</b><br>
Problem: {row['problem']}<br>Feedback: {row['feedback']}
</div>""", unsafe_allow_html=True)
        fig = px.line(df, x="time", y="score", markers=True, title="Score Progress", color_discrete_sequence=["#39FF14"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#000000", yaxis_range=[0, 10])
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download My Results", data=df.to_csv(index=False), file_name="my_results.csv", mime="text/csv")

elif page == "Leaderboard":
    st.title("Leaderboard")
    st.markdown("---")
    subs = st.session_state["student_submissions"]
    if not subs:
        st.info("No submissions yet.")
    else:
        df = pd.DataFrame(subs)
        lb = df.groupby("student")["score"].agg(["mean", "max", "count"]).reset_index()
        lb.columns = ["Student", "Average", "Best", "Submissions"]
        lb["Average"] = lb["Average"].round(1)
        lb = lb.sort_values("Average", ascending=False).reset_index(drop=True)
        medals = ["🥇", "🥈", "🥉"]
        for i, row in lb.iterrows():
            m = medals[i] if i < 3 else "🎖️"
            st.markdown(f"""<div style="background:#fff;padding:14px 20px;border-radius:12px;
margin:6px 0;border-left:5px solid #39FF14;box-shadow:0 2px 8px rgba(0,0,0,0.06);color:#000">
{m} <b>#{i+1} {row['Student'].title()}</b> &nbsp;|&nbsp;
Average: <b>{row['Average']}/10</b> &nbsp;|&nbsp;
Best: {row['Best']}/10 &nbsp;|&nbsp; {int(row['Submissions'])} submissions
</div>""", unsafe_allow_html=True)
        fig = px.bar(
            lb,
            x="Student",
            y="Average",
            color="Average",
            color_continuous_scale=["#f44336", "#FF9800", "#39FF14"],
            title="Average Scores",
            range_y=[0, 10]
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#000000")
        st.plotly_chart(fig, width="stretch")

elif page == "Question Bank":
    st.title("Question Bank")
    st.markdown("---")
    st.subheader("Add New Problem")
    c1, c2 = st.columns(2)
    with c1:
        qs = st.selectbox("Subject", list(SAMPLES.keys()), key="qb_s")
        qp = st.text_input("Problem Statement", key="qb_p")
        qa = st.text_input("Correct Answer", key="qb_a")
    with c2:
        qd = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="qb_d")
        qn = st.text_area("Notes", height=100, key="qb_n")
    if st.button("Add to Question Bank"):
        if qp and qa:
            st.session_state["question_bank"].append({
                "subject": qs,
                "problem": qp,
                "answer": qa,
                "difficulty": qd,
                "notes": qn,
                "added_by": st.session_state["user_name"],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success("Added!")
        else:
            st.warning("Fill in Problem and Answer.")
    st.markdown("---")
    st.subheader(f"Problems ({len(st.session_state['question_bank'])} total)")
    if not st.session_state["question_bank"]:
        st.info("No problems yet.")
    else:
        fs = st.selectbox("Filter", ["All"] + list(SAMPLES.keys()), key="qb_f")
        for i, q in enumerate([x for x in st.session_state["question_bank"] if fs == "All" or x["subject"] == fs]):
            dc = {"Easy": "#c8e6c9", "Medium": "#fff9c4", "Hard": "#ffcdd2"}.get(q["difficulty"], "#f0f0f0")
            st.markdown(f"""<div style="background:#fff;padding:16px;border-radius:10px;
border-left:4px solid #39FF14;margin:8px 0;box-shadow:0 2px 6px rgba(0,0,0,0.05);color:#000">
<b>#{i+1} [{q['subject']}]</b>
<span style="background:{dc};padding:2px 8px;border-radius:6px;font-size:0.8em;margin-left:6px">
{q['difficulty']}</span><br>
<b>Problem:</b> {q['problem']}<br>
<b>Answer:</b> {q['answer']}<br>
<small>Added by {q['added_by']} at {q['time']}</small>
</div>""", unsafe_allow_html=True)
        st.download_button(
            "Export Question Bank",
            data=pd.DataFrame(st.session_state["question_bank"]).to_csv(index=False),
            file_name="question_bank.csv",
            mime="text/csv"
        )

elif page == "Single Student":
    st.title("Single Student Grader")
    st.markdown("---")
    if not api_key:
        st.warning("No API key. Enter it in the sidebar.")
    smp = SAMPLES.get(subject, SAMPLES["Algebra"])
    c1, c2 = st.columns(2)
    with c1:
        problem = st.text_input("Problem", smp[0])
        correct_answer = st.text_input("Correct Answer", smp[1])
    with c2:
        method = st.radio("Input Method", ["Type solution", "Upload handwritten image"])
    sol = ""
    if method == "Type solution":
        sol = st.text_area("Student Solution", smp[2], height=130)
    else:
        f = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if f:
            st.image(f, width=380)
            if st.button("Read Handwriting"):
                with st.spinner("Reading..."):
                    result = read_image(f.read(), problem, subject)
                if result:
                    st.success("Extracted!")
                    st.session_state["extracted_solution"] = result
        if st.session_state["extracted_solution"]:
            sol = st.text_area("Extracted Solution", st.session_state["extracted_solution"], height=130)
    if st.button("Grade Now", width="stretch"):
        if not api_key:
            st.error("No API key.")
            st.stop()
        if not sol.strip():
            st.warning("Enter a solution.")
            st.stop()
        with st.spinner("Grading..."):
            r = run_grade(problem, sol, correct_answer, subject)
        st.markdown("---")
        show_results(r)

elif page == "Teacher Dashboard":
    st.title("Teacher Dashboard")
    st.markdown("---")
    if not api_key:
        st.warning("No API key. Enter it in the sidebar.")
    c1, c2 = st.columns(2)
    with c1:
        problem = st.text_input("Problem for whole class", "Solve for x: 2*x + 3 = 7")
        correct_answer = st.text_input("Correct Answer", "2")
    with c2:
        st.info("Upload up to 30 student images OR paste solutions below.")
    method = st.radio("Input Method", ["Paste solutions", "Upload student images"])
    names, solutions = [], []
    if method == "Paste solutions":
        raw = st.text_area(
            "Student solutions (Name: solution per line)",
            "Alice: 2*x + 3 = 7; 2*x = 4; x = 2\nBob: 2*x + 3 = 7; 2*x = 10; x = 5\nCarol: 2*x + 3 = 7; 2*x = 4; x = 2\nDavid: 2*x + 3 = 7; 2*x = 6; x = 3",
            height=180
        )
        for line in raw.strip().split("\n"):
            if ":" in line:
                n, s = line.split(":", 1)
                names.append(n.strip())
                solutions.append(s.strip())
    else:
        files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files:
            for f in files[:30]:
                names.append(f.name.rsplit(".", 1)[0])
                with st.spinner(f"Reading {f.name}..."):
                    solutions.append(read_image(f.read(), problem, subject))
            st.success(f"Read {len(files)} images!")

    if st.button("Grade All Students", width="stretch") and solutions:
        if not api_key:
            st.error("No API key.")
            st.stop()
        rows, prog, status = [], st.progress(0), st.empty()
        for i, (name, sol) in enumerate(zip(names, solutions)):
            status.text(f"Grading {name}... ({i+1}/{len(solutions)})")
            r = run_grade(problem, sol, correct_answer, subject)
            rows.append({
                "Student": name,
                "Score": r["score"],
                "Confidence": r["confidence"],
                "Error Type": r["error_type"],
                "Human Review": r["needs_review"],
                "Feedback": r["feedback"]
            })
            prog.progress((i + 1) / len(solutions))
        status.text("Done!")
        df = pd.DataFrame(rows)
        st.session_state["class_results"] = df
        avg = df["Score"].mean()
        top = df.loc[df["Score"].idxmax(), "Student"]
        rev = int(df["Human Review"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average", f"{avg:.1f}/10")
        c2.metric("Top Student", top)
        c3.metric("Need Review", f"{rev}/{len(df)}")
        c4.metric("Total", len(df))
        st.markdown("---")

        # Simple dataframe is used here to avoid pandas Styler/applymap issues on Streamlit Cloud.
        st.dataframe(df, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                df,
                x="Student",
                y="Score",
                color="Score",
                color_continuous_scale=["#f44336", "#FF9800", "#39FF14"],
                title="Scores",
                range_y=[0, 10]
            )
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#000000")
            st.plotly_chart(fig, width="stretch")
        with c2:
            ec = df["Error Type"].value_counts().reset_index()
            ec.columns = ["Error Type", "Count"]
            fig2 = px.pie(ec, values="Count", names="Error Type", title="Error Distribution")
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#000000")
            st.plotly_chart(fig2, width="stretch")
        st.markdown("---")
        st.subheader("Email Report")
        c1, c2 = st.columns(2)
        with c1:
            gmail = st.text_input("Your Gmail", placeholder="you@gmail.com")
            gpwd = st.text_input("Gmail App Password", type="password")
            to = st.text_input("Send to", placeholder="teacher@school.com")
        with c2:
            body = f"STEMGrade Report\nProblem: {problem}\nSubject: {subject}\nTotal: {len(df)}\nAverage: {avg:.1f}/10\nTop: {top}\n\n"
            for _, row in df.iterrows():
                body += f"{row['Student']}: {row['Score']}/10 - {row['Feedback']}\n"
            st.text_area("Preview", body, height=150)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Send Email Report"):
                if gmail and gpwd and to:
                    try:
                        msg = MIMEMultipart()
                        msg["From"] = gmail
                        msg["To"] = to
                        msg["Subject"] = f"STEMGrade Report ({subject})"
                        msg.attach(MIMEText(body, "plain"))
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                            s.login(gmail, gpwd)
                            s.sendmail(gmail, to, msg.as_string())
                        st.success(f"Sent to {to}!")
                    except Exception as e:
                        st.error(f"Email failed: {e}")
                else:
                    st.warning("Fill all email fields.")
        with c2:
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False),
                file_name="class_grades.csv",
                mime="text/csv"
            )

elif page == "Student Submissions":
    st.title("Student Submissions")
    st.markdown("---")
    subs = st.session_state["student_submissions"]
    if not subs:
        st.info("No submissions yet.")
    else:
        df = pd.DataFrame(subs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(df))
        c2.metric("Average", f"{df['score'].mean():.1f}/10")
        c3.metric("Students", df["student"].nunique())
        fs = st.selectbox("Filter", ["All"] + sorted(df["student"].unique().tolist()))
        st.dataframe(df if fs == "All" else df[df["student"] == fs], width="stretch")
        fig = px.bar(
            df,
            x="student",
            y="score",
            color="score",
            color_continuous_scale=["#f44336", "#FF9800", "#39FF14"],
            title="Scores",
            range_y=[0, 10]
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#000000")
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download All", data=df.to_csv(index=False), file_name="submissions.csv", mime="text/csv")

elif page == "Analytics":
    st.title("Analytics")
    st.markdown("---")
    if st.session_state["class_results"] is None:
        st.info("Grade a class first in Teacher Dashboard.")
    else:
        df = st.session_state["class_results"]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="Score", nbins=10, title="Score Distribution", color_discrete_sequence=["#39FF14"])
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#000000")
            st.plotly_chart(fig, width="stretch")
        with c2:
            fig2 = px.box(df, y="Score", title="Score Spread", color_discrete_sequence=["#000000"])
            fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#000000")
            st.plotly_chart(fig2, width="stretch")
        fig3 = px.bar(
            df,
            x="Student",
            y="Confidence",
            color="Confidence",
            color_continuous_scale=["#f44336", "#FF9800", "#39FF14"],
            title="Confidence by Student"
        )
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#000000")
        st.plotly_chart(fig3, width="stretch")
        rev = df[df["Human Review"] == True][["Student", "Score", "Error Type", "Feedback"]]
        st.subheader("Students Needing Review")
        if len(rev) > 0:
            st.dataframe(rev, width="stretch")
        else:
            st.success("No students need review!")
