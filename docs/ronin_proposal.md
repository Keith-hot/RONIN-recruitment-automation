# RONIN International — Smart Respondent Recruitment Strategy

**Document Type:** Internal Strategy Proposal  
**Author:** Keith  
**Date:** April 2026  
**Target:** Digital Research Operations, RONIN International (Hong Kong)

---

## 1. Observation: Current Pain Points

### Organic Recruitment Challenges
- **High researcher overhead:** Each lead requires 15+ minutes of manual research, drafting, and follow-up scheduling.
- **Inconsistent touchpoints:** Warm leads (those who opened emails or clicked links) often receive no further engagement beyond a single reminder.
- **No prioritization:** All respondents are approached with the same urgency, diluting researcher bandwidth on low-probability prospects.

### Campaign-Driven Recruitment Challenges
- **Generic messaging:** Mass outreach to C-Suite and specialist audiences yields single-digit reply rates because content lacks sector-specific relevance.
- **Channel rigidity:** Email-first approaches ignore the fact that APAC executives are highly responsive on LinkedIn, while WhatsApp is more effective for urgent reactivation in LATAM.
- **No closed-loop feedback:** There is no systematic way to learn which subject lines, send times, or incentives drive the highest conversion per segment.

### Financial Impact
- At 4 projects × 250 leads/month with a manual 11.1% participation rate, RONIN leaves significant revenue on the table.
- Researcher time spent on administrative follow-up is time not spent on interview design, client reporting, or methodology innovation.

---

## 2. Proposed Solution: 3-Phase Implementation

### Phase 1 — Lead Scoring & Prioritization (Weeks 1–2)
- Train a LightGBM classifier on historical respondent data (demographics, past participation, interaction history).
- Output: a 0–1 probability score for each new lead.
- Immediate action: route high-intent leads (>0.8) to senior researchers for white-glove treatment; route low-scoring leads to automated nurture pools.

### Phase 2 — RAG-Driven Personalization (Weeks 3–4)
- Build a vector knowledge base of research project briefs (Healthcare Digital Transformation, Fintech AI Adoption, Enterprise Cybersecurity).
- Use FAISS + Sentence-Transformers to retrieve the most relevant project context for each respondent's industry and seniority.
- Generate personalized email/LinkedIn/WhatsApp content that references specific pain points (e.g., "AI-assisted diagnosis" for Healthcare CIOs).

### Phase 3 — Full Automation & Triggers (Weeks 5–6)
- Deploy a state-machine nurture manager (7 states: NEW → INITIAL_CONTACT → ENGAGED → RESPONDED → SCHEDULED → PARTICIPATED → NURTURE).
- Implement 5 intelligent trigger rules:
  1. Email opened → LinkedIn follow-up after 6 hours.
  2. Link clicked → fast-track scheduling reminder.
  3. Positive reply → stop automation, notify sales team.
  4. 7-day silence → switch to WhatsApp reactivation.
  5. Multiple opens, no clicks → flag content fatigue, shorten next subject line.
- Add cooldown logic to prevent over-messaging.

---

## 3. Integration Points

### CATI Systems (Computer-Assisted Telephone Interviewing)
- The `SCHEDULED` state in the nurture manager can push confirmed appointments directly to CATI scheduling APIs.
- Pre-interview reminders (24h, 1h) can be triggered automatically via the same channel that secured the booking (Email, WhatsApp, or calendar invite).

### CRM (Salesforce / HubSpot)
- Lead scores and engagement flags (e.g., `content_fatigue`, `high_priority`) should be written back to CRM contact records.
- Trigger events (`email_opened`, `link_clicked`) can be ingested via CRM webhooks to keep the nurture engine in sync with real-world behaviour.

### Reporting & Client Dashboards
- Aggregate funnel metrics (contact → open → reply → schedule → participate) can be exposed to client-facing dashboards, proving recruitment efficiency and ROI.

---

## 4. Compliance: GDPR & Data Privacy

### Consent & Legitimate Interest
- All automated emails must include a one-click unsubscribe mechanism (`UNSUBSCRIBED` state in the nurture manager).
- For EU respondents, lawful basis should be documented as "legitimate interest" (B2B research) with an easy opt-out.

### Data Minimization
- The lead scoring model uses only professional attributes (industry, job level, company size, past participation). No sensitive personal data is required.
- FAISS vector indices store only text embeddings, not raw respondent PII.

### Retention & Anonymization
- Interaction logs should be retained for 12 months for model retraining, then anonymized.
- Respondent identifiers should be hashed before being used in aggregate reporting.

### Cross-Border Transfers
- APAC and EMEA data should be processed in region-appropriate cloud regions to comply with local data residency requirements.

---

## 5. Expected Outcomes

| KPI | Current (Manual) | Target (AI-Powered) | Timeline |
|-----|------------------|---------------------|----------|
| Participation Rate | 11.1% | 40%+ | 30 days |
| Cost per Qualified Interview | ~$45 | ~$18 | 30 days |
| Researcher Hours / 1,000 Leads | 250h | 100h | 30 days |
| Response Rate (Hard-to-Reach) | ~3% | ~12% | 60 days |

---

*This document is intended for internal discussion and does not constitute a formal business proposal.*
