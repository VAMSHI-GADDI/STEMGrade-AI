# STEMGrade AI — Handwritten STEM Reasoning Evaluator

> AI that reads handwritten math, verifies step-by-step reasoning, and grades like a human teacher.

🔗 **Live Demo: (https://stemgrade-ai-vamshi-gaddi.streamlit.app)**

**Teacher login:** `teacher` / `teach@123`

---

## What It Does

Most grading systems only check the final answer. STEMGrade AI evaluates **how** a student thinks — step by step.

Given a handwritten photo or typed solution, the system:

- Reads handwritten STEM solutions using **GPT-4o vision**
- Generates a reference solution using an LLM
- Verifies each student step **symbolically** using SymPy
- Detects exactly **where reasoning breaks**
- Classifies the **error type**
- Assigns a **confidence score**
- Flags cases needing **human review**

---

## Demo

![STEMGrade AI Demo](demo.png)

---

## Error Classification

| Error Type | Description |
|---|---|
| `none` | All steps correct, 10/10 |
| `reasoning_path_error` | Right answer, wrong reasoning |
| `final_answer_error` | Right reasoning, wrong answer |
| `early_reasoning_error` | Mistake in first two steps |
| `reasoning_error` | Mistake later in solution |
| `problem_mismatch` | Solution doesn't match the problem |
| `needs_human_review` | Cannot be reliably auto-graded |

---

## Subjects Supported

- Algebra, Quadratics, Systems of Equations
- Physics, Chemistry, Calculus

---

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| Handwriting reading | OpenAI GPT-4o Vision |
| LLM reasoning | OpenAI GPT-4o-mini |
| Symbolic verification | SymPy |
| Analytics | Plotly |
| Deployment | Streamlit Cloud |

---

## Features

- **Handwriting reading** — upload a photo, GPT-4o extracts the steps
- **Teacher dashboard** — grade entire class, download CSV/text reports
- **Student portal** — submit solutions, track score history
- **Leaderboard** — class rankings
- **Question bank** — teachers upload problems for students
- **Submission timer** — countdown for timed assignments
- **Email reports** — send grading results directly to teacher
- **Analytics** — score distribution, confidence charts, review flags

---

## Why This Matters

This system moves beyond answer-checking into **reasoning path evaluation** — critical for:

- AI education tools
- Automated grading at scale
- LLM mathematical reasoning research
- Benchmark construction for structured reasoning

---

## Author

**Vamshi Gaddi**  
M.S. Information Science and Technology, Missouri S&T  
[github.com/VAMSHI-GADDI](https://github.com/VAMSHI-GADDI)
