# Smart Respondent Recruitment System — 3.9× Conversion Lift

> AI-powered automation pipeline for B2B market research recruitment. Built to solve the "hard-to-reach audience" problem at RONIN International scale.

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

<p align="center">
  <img src="./dashboard/funnel_comparison.html" width="900" alt="Funnel Comparison: Manual vs AI-Powered">
</p>
<p align="center">
  <b>Left: Traditional Manual Process (11.1% participation) | Right: AI-Powered (43.1%)</b>
</p>

---

## 🎯 Executive Summary

- **3.9× participation lift** — from 11.1% (manual) to 43.1% (AI-powered) through intelligent prioritization and multi-channel nurture sequences.
- **$1.95M projected annual value** — at 4 projects × 250 leads/month, driven by faster scheduling and reduced no-shows.
- **40% cost reduction** — automation cuts manual research/drafting time from 15 min/lead to human-review-only workflows, with break-even in 0.5 days.

## 🏢 Business Context

RONIN International conducts B2B qualitative and quantitative research across 80+ countries. Recruiting C-Suite executives, healthcare professionals, and cybersecurity leaders is expensive and time-intensive:

- **Generic outreach** is ignored by high-value respondents.
- **Manual follow-up** stops after 1–2 attempts, even though 80% of conversions require 5+ touchpoints.
- **No prioritization** means warm leads are treated the same as cold prospects.

This system replaces the manual workflow with an end-to-end pipeline: **score → personalize → nurture → trigger → measure ROI**.

## 📊 Key Results

| Metric | Manual (Baseline) | AI-Powered | Lift |
|--------|-------------------|------------|------|
| Contact → Open | 61.0% | 79.3% | +1.3× |
| Open → Reply | 51.4% | 71.9% | +1.4× |
| Reply → Schedule | 15.9% | 50.8% | +3.2× |
| Schedule → Participate | 11.1% | 43.1% | **+3.9×** |
| AUC (Lead Scoring) | — | **0.93** | — |
| Monthly Labour Cost | $6,250 | $3,750 | **−40%** |
| Break-Even Period | — | **0.5 days** | — |
| Annual Revenue Uplift | — | **$1.92M** | — |

*Based on 1,000-respondent simulation with 5,000 historical interactions.*

## 🛠️ Technical Architecture

| Module | Technology | Purpose |
|--------|------------|---------|
| **Data Layer** | pandas, NumPy | 1,000 respondent profiles + 5,000 interaction records |
| **Lead Scoring** | LightGBM, scikit-learn | Binary classifier predicting participation probability (AUC 0.93) |
| **Personalization** | FAISS, Sentence-Transformers, LangChain | RAG-driven email generation tailored to industry and seniority |
| **Nurture Sequences** | Python state machine | 7-state lifecycle (NEW → INITIAL → ENGAGED → RESPONDED → SCHEDULED → NURTURE → UNSUBSCRIBED) |
| **Trigger Engine** | Event-driven rules | 5 intelligent rules across Email, LinkedIn, WhatsApp with cooldown and priority logic |
| **ROI Dashboard** | Plotly, HTML | 4 interactive charts: funnel, timeline, lead score distribution, channel effectiveness |

## 📈 Live Demo

Open the dashboards directly in your browser:

- [Funnel Comparison](./dashboard/funnel_comparison.html)
- [ROI Timeline](./dashboard/roi_timeline.html)
- [Lead Score Distribution](./dashboard/lead_score_distribution.html)
- [Channel Effectiveness](./dashboard/channel_effectiveness.html)

Or view the project showcase page: `./index.html`

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Keith-hot/ronin-recruitment-automation.git
cd ronin-recruitment-automation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the lead scoring model
python src/lead_scoring.py

# 4. Run the personalization engine
python src/personalization_engine.py

# 5. Run the nurture sequence simulation
python src/nurture_sequences.py

# 6. Run the trigger engine test
python src/trigger_manager.py

# 7. Generate ROI dashboards
python dashboard/roi_dashboard.py
```

All outputs are saved to `./data/` and `./dashboard/`.

## 📁 Project Structure

```
.
├── data/
│   ├── raw/               # respondents.csv, interactions.csv
│   ├── knowledge_base/    # Healthcare, Finance, Tech research briefs
│   ├── simulation_log.csv # 14-day nurture simulation
│   └── trigger_log.csv    # Event-driven trigger audit
├── dashboard/             # Plotly HTML charts + roi_dashboard.py
├── models/                # lgbm_lead_scorer_v1.pkl, faiss_index
├── src/
│   ├── lead_scoring.py           # LightGBM pipeline
│   ├── personalization_engine.py # RAG email generator
│   ├── nurture_sequences.py      # State-machine manager
│   └── trigger_manager.py        # Event-driven rules + adapters
├── docs/
│   ├── follow_up_email.md        # Job application follow-up
│   └── ronin_proposal.md         # Internal strategy document
├── index.html             # GitHub Pages landing page
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

## 💼 Author

**Keith** — Data Science & Machine Learning Engineering

- Email: 1720261562@qq.com
- GitHub: [@Keith-hot](https://github.com/Keith-hot)
- Target: Digital Research and B2B Market Research roles at RONIN International

Built as a hands-on prototype to demonstrate how data-driven automation can transform respondent recruitment workflows.
