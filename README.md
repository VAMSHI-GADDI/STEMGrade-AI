# STEMGrade AI — Handwritten STEM Reasoning Evaluator

> AI that reads handwritten math, verifies step-by-step reasoning, and grades like a human teacher.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-39FF14?style=for-the-badge)](https://stemgrade-ai-vamshi-gaddi.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-VAMSHI--GADDI-black?style=for-the-badge&logo=github)](https://github.com/VAMSHI-GADDI/STEMGrade-AI)

**🔗 App: https://stemgrade-ai-vamshi-gaddi.streamlit.app**

# STEMGrade AI

**STEMGrade AI** is an AI-assisted grading tool designed to help tutors review STEM solutions faster, detect step-by-step reasoning mistakes, flag answers that need human review, and export CSV/PDF reports.

**Built by:** Vamshi  
**Demo Video:** [Watch here](https://drive.google.com/file/d/1TNnBRtTrAitZ7InbGyMqldjjehWaMq7y/view?usp=sharing)  
**Live App:** [Open Streamlit App](https://stemgrade-ai-vamshi-gaddi.streamlit.app/)

---

## 📸 Demo

![STEMGrade AI Demo](demo.png)

---

## What It Does

Most grading systems only check the final answer. STEMGrade AI evaluates **how** a student thinks — step by step.

Given a handwritten photo or typed solution, the system:

- 📷 Reads handwritten STEM solutions using **GPT-4o vision**
- 🧠 Generates a reference solution using an LLM
- ✅ Verifies each student step **symbolically** using SymPy
- ❌ Detects exactly **where reasoning breaks**
- 🏷️ Classifies the **error type**
- 📊 Assigns a **confidence score**
- 👁️ Flags cases needing **human review**

---

## End-to-End Pipeline

No separate OCR step — one end-to-end pipeline from image to structured reasoning evaluation.

---

## Error Classification

| Error Type | Description | Score |
|---|---|---|
| `none` | All steps correct | 10/10 |
| `reasoning_path_error` | Right answer, wrong reasoning | 7/10 |
| `final_answer_error` | Right reasoning, wrong answer | 6/10 |
| `reasoning_error` | Mistake in solution | 5/10 |
| `early_reasoning_error` | Mistake in first two steps | 4/10 |
| `needs_human_review` | Cannot be reliably auto-graded | 3/10 |

---

## Subjects Supported

Algebra · Quadratics · Systems of Equations · Physics · Chemistry · Calculus

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

- 📷 **Handwriting reading** — upload a photo, GPT-4o extracts and evaluates the steps
- 👨‍🏫 **Teacher dashboard** — grade entire class, download CSV/text reports
- 👤 **Student portal** — submit solutions, track score history
- 🏆 **Leaderboard** — class rankings by average score
- 📚 **Question bank** — teachers upload and assign problems
- ⏱️ **Submission timer** — countdown for timed assignments
- 📧 **Email reports** — send grading results directly to teacher
- 📊 **Analytics** — score distribution, confidence charts, review flags

---

## Why This Matters

This system moves beyond answer-checking into **reasoning path evaluation** — critical for:

- AI education tools
- Automated grading at scale
- LLM mathematical reasoning research
- Benchmark construction for structured reasoning failure analysis

---

## Author

**Vamshi Gaddi**
M.S. Information Science and Technology, Missouri S&T
[github.com/VAMSHI-GADDI](https://github.com/VAMSHI-GADDI)
