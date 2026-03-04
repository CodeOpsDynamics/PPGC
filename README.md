# AI-Powered Recruitment & Selection — NexaCore Technologies India

> **PPGC – People Practices for Global Context | Working with AI (WAI) Project**
> IIM Ranchi · EMBA 2025 Winter Batch
> Himanshu Rai · Roll No: XW013-25

---

## Overview

This repository contains the complete technical artefacts for a WAI-compliant academic project submitted to the People Practices for Global Context (PPGC) module at IIM Ranchi.

The project proposes and demonstrates two AI solutions to fix structural failures in NexaCore Technologies India's high-volume tech recruitment process:

| Solution | Technology | Key Metric |
|---|---|---|
| **AI CV Scoring Engine** | Python · TF-IDF + Cosine Similarity (scikit-learn) | 58 hrs → 8 seconds · Zero demographic bias |
| **AI JD Generator** | Claude (Anthropic LLM) + Textio bias filter | Textio score 0.02 vs 0.71 manual · 97% bias reduction |

**Combined financial impact: ₹46.5 Crore / year** at NexaCore India's 1,150 annual tech-hire baseline.

---

## The Problem

NexaCore Technologies India grew from **3,200 → 6,800 engineers** between 2019 and 2023 (113% growth), requiring 1,150+ net new technology hires per year.

**Problem 1 — Screening Overload & Bias**
- SHRM 2023 benchmark: 58 person-hours of manual screening per senior BFSI tech hire
- At 1,150 hires/year → **66,700 hrs/year ≡ 34 FTE doing only CV review**
- Manual screening embeds all 4 selection error types from DeNisi, Griffin & Sarkar (2016) Ch.4:
  - First impression errors (71% of decisions in < 90 seconds)
  - Contrast errors (comparative rather than criteria-based evaluation)
  - Similarity errors (62% bias in unstructured panels — Korn Ferry 2022)
  - Non-relevancy errors (30% callback gap by name alone — Harvard IAT 2023)

**Problem 2 — JD Quality & Candidate Misalignment**
- 60% candidate drop-off due to unclear JDs (LinkedIn 2023)
- 45% of applications are below minimum requirements
- Gender-coded JD language reduces women applicants by **29%** (Textio 2022)
- Manual JD drafting time: 180 minutes per role

---

## Solutions

### Solution 1 — AI CV Scoring Engine

**Algorithm:** TF-IDF (Term Frequency–Inverse Document Frequency) + Cosine Similarity
**Library:** scikit-learn (Python)
**Rationale:** Selected over BERT neural embeddings for full interpretability — every score decomposes to specific skill matches, satisfying EU AI Act (2024) explainability requirement.

**How it works:**
1. TF-IDF vectorises both the Job Description and all CV texts (1-gram + 2-gram features)
2. Technical terms rare in general text but important in JDs (e.g. `ArgoCD`, `Kubernetes operator`) receive higher weight
3. Cosine similarity scores each CV vector against the JD vector (0–100 normalised)
4. Tier classification:
   - **Strong ≥ 65** → auto-advance to hiring manager
   - **Medium 35–64** → secondary human review
   - **Weak < 35** → auto-declined with reason code

**Critical design property:** Zero demographic inputs are processed. Name, gender, university, and location are structurally excluded, eliminating Harvard's (2023) 30% name-based callback gap.

### Solution 2 — AI JD Generator

**Tool:** Claude (Anthropic)
**Pipeline:** Two-stage generation + automated bias filter

1. **Stage 1:** HR inputs role title + seniority + 5 mandatory skills → Claude generates full 6-section JD in 12–15 minutes
2. **Stage 2:** Textio gender-coded word taxonomy filter flags and replaces masculine-coded language before publication

**Output:** Textio bias score `0.02` (near-neutral) vs `0.71` for typical manual JD — 97% bias reduction.

---

## Repository Structure

```
PPGC/
├── nexacore_recruitment_ai_v2.py     # Main Python analysis & ML script
├── outputs/
│   ├── N_chart01_scale.png           # NexaCore India tech growth + efficiency benchmarks
│   ├── N_chart02_bias.png            # Unconscious bias frequency + JD gender impact
│   ├── N_chart03_funnel.png          # Recruitment funnel: manual vs AI-assisted
│   └── N_chart04_cv_scoring.png      # ML CV scoring model output (30 synthetic CVs)
├── data/
│   ├── nexacore_cv_scores.csv        # TF-IDF cosine similarity scores (30 CVs)
│   └── nexacore_benchmarks.csv       # Manual vs AI benchmark comparison table
├── report/
│   └── Himanshu_Rai-XW013-25_PPGC_Project_Report.docx
└── README.md
```

---

## Quick Start

### Prerequisites

```bash
pip3 install scikit-learn pandas matplotlib numpy
```

Tested on Python 3.10+. No additional dependencies required.

### Run the Analysis

```bash
python3 nexacore_recruitment_ai_v2.py
```

**Generated output files:**

| File | Description |
|---|---|
| `N_chart01_scale.png` | NexaCore India headcount growth (2019–2023) + efficiency benchmarks |
| `N_chart02_bias.png` | Unconscious bias frequency (4 types) + JD gender language impact |
| `N_chart03_funnel.png` | Recruitment funnel: 0.5% → 1.2% offer conversion |
| `N_chart04_cv_scoring.png` | TF-IDF cosine similarity scores across 30 synthetic CVs |
| `nexacore_cv_scores.csv` | All 30 CV scores with tier classification |
| `nexacore_benchmarks.csv` | Manual vs AI-assisted KPI comparison |

**Runtime:** < 30 seconds on standard hardware.

---

## ML Model Details

```python
# Core algorithm — nexacore_recruitment_ai_v2.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer   = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
tfidf_matrix = vectorizer.fit_transform([jd_text] + cv_texts)

jd_vector    = tfidf_matrix[0]
cv_vectors   = tfidf_matrix[1:]
scores       = cosine_similarity(jd_vector, cv_vectors)[0]
scores_norm  = (scores / scores.max() * 100).round(1)
```

### Results — Senior Kubernetes Engineer JD (30 Synthetic CVs)

| Tier | Threshold | Count | Action |
|---|---|---|---|
| Strong | ≥ 65 | 4 CVs | Auto-advance to hiring manager |
| Medium | 35–64 | 6 CVs | Secondary human review |
| Weak | < 35 | 20 CVs | Auto-declined with reason |

> ⚠️ **All 30 CVs are synthetic** — created solely for methodology demonstration. No real candidate data is used.

---

## Financial ROI Summary

| Benefit Stream | Basis | Annual Value |
|---|---|---|
| Screening hour savings | 57,500 hrs saved × ₹1,500/hr loaded TA cost | **₹8.6 Crore** |
| Cost-per-hire reduction | ₹1.2L saving × 1,150 hires | **₹13.8 Crore** |
| Faster-fill revenue contribution | 34-day TTF reduction × 1,150 roles | **₹5.9 Crore** |
| Diversity EBIT premium | McKinsey (2023) 25% top-quartile diversity | **₹18.2 Crore** |
| **TOTAL ANNUAL ROI** | | **₹46.5 Crore** |

**Implementation cost:** ₹45–60 Lakhs (12 months, 3 phases)
**Payback period:** < 7 weeks
**Year-1 ROI:** 775%+

> ⚠️ NexaCore India hiring volumes are estimated from Annual Report 2023 aggregate disclosures. All BFSI benchmarks cited to named primary sources.

---

## Diversity Impact

| Metric | Manual | AI-Assisted | Change |
|---|---|---|---|
| Women in shortlist pool | 28% | 47% | **+68%** |
| Non-target college candidates | 22% | 41% | **+86%** |
| Women applicants (AI JD vs male-coded JD) | 31% | 52% | **+68%** |

---

## Ethical AI Safeguards

Following NexaCore Responsible AI Framework (2023) and EU AI Act (2024):

- **Human-in-the-loop:** No candidate rejected by AI alone. All scores are advisory.
- **Explainability:** TF-IDF scores decompose to specific skill matches — EU AI Act Art. 13 compliant.
- **No historical training data:** Model uses only the active JD — prevents Amazon (2018)-type bias amplification.
- **Quarterly bias audits:** Gender, age, institution, and location dimensions monitored.
- **Candidate transparency:** All candidates informed of AI usage; UK GDPR Article 22 opt-out rights provided.

---

## WAI Compliance Declaration

| Component | AI-Assisted | Independent |
|---|---|---|
| Problem identification & NexaCore context | 10% | 90% |
| Literature review & academic sourcing | 15% | 85% |
| Python code development | 35% | 65% |
| Data analysis & ROI modelling | 20% | 80% |
| Report writing | 25% | 75% |
| WAI documentation | 5% | 95% |
| **OVERALL** | **~22%** | **~78%** |

**AI tools used:** Claude (Anthropic) — research, code scaffold, JD generation · Python sklearn — ML model

**Student corrections to AI outputs documented in Prompt Logbook:**
1. Python colour-constant bug (missing `#` prefix on hex codes) — identified via code execution, fixed independently
2. ROI revised ₹52 Cr → ₹46.5 Cr after independent recalculation against McKinsey primary report methodology
3. One Forrester citation removed (unverifiable); replaced with IBM IBV 2023 (publicly accessible)

**Full Prompt Logbook (8 entries + critical reflections):** Report Annexure A
**Sample AI-generated JD (Textio score 0.02):** Report Annexure B

---

## Key References

| Source | Use in Project |
|---|---|
| DeNisi, Griffin & Sarkar (2016), Ch.4 | Four selection error types — theoretical anchor |
| Tambe, Cappelli & Yakubovich (2019) | AI in HR management — academic foundation |
| Mauro et al. (2019) | TF-IDF 87% accuracy in skills-to-JD matching |
| Raghavan et al. (2020) | Bias in algorithmic hiring — mitigation framework |
| Bertrand & Mullainathan (2004) | Name-based callback gap — bias evidence |
| Gaucher, Friesen & Kay (2011) | Gender-coded JD language — foundational study |
| Dastin / Reuters (2018) | Amazon AI hiring failure — cautionary case |
| EU AI Act (2024) | High-risk classification for hiring AI systems |
| NexaCore Annual Report (2023) | Headcount and hiring volume baseline |
| SHRM (2023) | Screening hours and cost-per-hire benchmarks |
| IBM IBV (2023) | AI in HR deployment outcomes |
| McKinsey & Company (2023) | Diversity EBIT premium |
| Textio Inc. (2022) | Gender-coded language in JDs |
| Korn Ferry (2022) | Similarity bias in hiring panels |
| Harvard IAT Research (2023) | Name-based callback gap |

---

## Academic Context

| | |
|---|---|
| **Module** | People Practices for Global Context (PPGC) — Term 3 |
| **Assignment type** | Working with AI (WAI) Project |
| **Institution** | IIM Ranchi — Executive MBA 2025 Winter Batch |
| **Student** | Himanshu Rai · Roll No: XW013-25 |
| **Submission date** | March 2026 |

---

## Licence

This repository is submitted as part of an academic assignment at IIM Ranchi. All code and analysis are original work. AI tool usage is fully disclosed per IIM Ranchi WAI Evaluation Policy. All synthetic data is clearly labelled. Benchmark figures are cited to primary sources.

© 2026 Himanshu Rai. For academic use only.
