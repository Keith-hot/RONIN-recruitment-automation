"""
RONIN Digital Research – State-Machine Driven Nurture Sequence Manager
Manages the full respondent journey from first contact to interview completion.
"""

import os
import sys
import csv
import pandas as pd
from datetime import date, timedelta
from typing import Optional

from lead_scoring import LeadScoringModel
from personalization_engine import PersonalizationEngine


class SequenceManager:
    """Finite-state-machine nurture orchestrator."""

    # ------------------------------------------------------------------
    # State constants
    # ------------------------------------------------------------------
    NEW = "new"
    INITIAL_CONTACT = "initial_contact"
    ENGAGED = "engaged"
    RESPONDED = "responded"
    SCHEDULED = "scheduled"
    PARTICIPATED = "participated"
    NURTURE = "nurture"
    UNSUBSCRIBED = "unsubscribed"

    # ------------------------------------------------------------------
    # Time-rule parameters (class-level for easy tuning)
    # ------------------------------------------------------------------
    DAYS_TO_NURTURE = 7          # Rule C: NEW -> NURTURE (no response)
    DAYS_ENGAGED_TO_NURTURE = 3  # Rule E: ENGAGED -> NURTURE (no further activity)
    DAYS_REACTIVATION = 30       # Rule G: NURTURE -> INITIAL_CONTACT

    def __init__(
        self,
        lead_scorer: LeadScoringModel,
        content_engine: PersonalizationEngine,
        start_date: Optional[date] = None,
    ):
        self.lead_scorer = lead_scorer
        self.content_engine = content_engine
        self.respondents: dict[str, dict] = {}
        self.action_log: list[dict] = []
        self.current_date = start_date or date.today()
        self._sim_start_date = self.current_date

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _log_action(self, respondent_id: str, state: str, action: str, notes: str = "") -> None:
        day = (self.current_date - self._sim_start_date).days
        entry = {
            "day": day,
            "respondent_id": respondent_id,
            "state": state,
            "action_triggered": action,
            "notes": notes,
        }
        self.action_log.append(entry)

    def _transition(self, respondent_id: str, new_state: str, action: str, notes: str = "") -> None:
        r = self.respondents[respondent_id]
        old_state = r["state"]
        r["state"] = new_state
        r["state_entered_at"] = self.current_date
        r["history"].append(
            {
                "date": self.current_date.isoformat(),
                "from": old_state,
                "to": new_state,
                "action": action,
            }
        )
        self._log_action(respondent_id, new_state, action, notes)
        print(f"  [{respondent_id}] {old_state:20s} -> {new_state:20s} | {action} | {notes}")

    def _get_project_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _generate_email_subject(self, profile: dict) -> str:
        """Safely generate an email subject via the content engine."""
        try:
            email = self.content_engine.generate_email(profile)
            return email["subject"]
        except Exception as exc:
            return f"[Email generation failed: {exc}]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_respondent(self, respondent_id: str, profile: dict, priority_score: float) -> None:
        """Ingest a new lead into the nurture system."""
        self.respondents[respondent_id] = {
            "profile": profile,
            "state": self.NEW,
            "state_entered_at": self.current_date,
            "priority_score": priority_score,
            "high_priority": priority_score > 0.8,
            "history": [],
        }
        self._log_action(respondent_id, self.NEW, "added_to_system", f"priority_score={priority_score:.3f}")
        print(f"  [SYSTEM] Added {respondent_id} (priority={priority_score:.3f}, high_priority={priority_score > 0.8})")

    def get_state(self, respondent_id: str) -> tuple[str, int]:
        """Return (current_state, days_in_state)."""
        r = self.respondents[respondent_id]
        days = (self.current_date - r["state_entered_at"]).days
        return r["state"], days

    # ------------------------------------------------------------------
    # Daily trigger processor
    # ------------------------------------------------------------------
    def process_daily_triggers(self) -> None:
        """Scan all respondents and fire time-based state transitions."""
        for rid, r in self.respondents.items():
            state = r["state"]
            days_in_state = (self.current_date - r["state_entered_at"]).days
            profile = r["profile"]

            # ----------------------------------------------------------
            # Rule A: NEW -> INITIAL_CONTACT (Day 0)
            # ----------------------------------------------------------
            if state == self.NEW:
                region = profile.get("region", "North America")
                send_hour = {
                    "APAC": "09:00",
                    "EMEA": "14:00",
                    "North America": "10:00",
                    "LATAM": "10:00",
                }.get(region, "10:00")
                subject = self._generate_email_subject(profile)
                self._transition(
                    rid,
                    self.INITIAL_CONTACT,
                    "sent_initial_email",
                    f"Send at {send_hour} local | Subject: {subject}",
                )
                continue

            # ----------------------------------------------------------
            # Rule C: INITIAL_CONTACT -> NURTURE (7 days no response)
            # ----------------------------------------------------------
            if state == self.INITIAL_CONTACT and days_in_state >= self.DAYS_TO_NURTURE:
                subject = self._generate_email_subject(profile)
                self._transition(
                    rid,
                    self.NURTURE,
                    "moved_to_nurture_pool",
                    f"Sent nurture whitepaper | Subject: {subject}",
                )
                continue

            # ----------------------------------------------------------
            # Rule E: ENGAGED -> NURTURE (3 days no further interaction)
            # ----------------------------------------------------------
            if state == self.ENGAGED and days_in_state >= self.DAYS_ENGAGED_TO_NURTURE:
                channel = profile.get("preferred_contact", "Email")
                escalation = (
                    "LinkedIn connection request"
                    if channel != "LinkedIn"
                    else "Last-chance email via alternate channel"
                )
                self._transition(
                    rid,
                    self.NURTURE,
                    "escalation_attempt",
                    f"{escalation} sent",
                )
                continue

            # ----------------------------------------------------------
            # Rule G: NURTURE -> INITIAL_CONTACT (30 days + new project)
            # ----------------------------------------------------------
            if state == self.NURTURE and days_in_state >= self.DAYS_REACTIVATION:
                subject = self._generate_email_subject(profile)
                self._transition(
                    rid,
                    self.INITIAL_CONTACT,
                    "reactivation",
                    f"New project match | Subject: {subject}",
                )
                continue

    # ------------------------------------------------------------------
    # External event ingestion
    # ------------------------------------------------------------------
    def record_event(self, respondent_id: str, event_type: str, metadata: Optional[str] = None) -> None:
        """Record inbound behavioural events (email open, click, reply, etc.)."""
        if respondent_id not in self.respondents:
            print(f"  [WARN] Unknown respondent: {respondent_id}")
            return

        r = self.respondents[respondent_id]
        state = r["state"]
        notes = metadata or ""

        # --------------------------------------------------------------
        # Rule B: INITIAL_CONTACT -> ENGAGED (email opened)
        # --------------------------------------------------------------
        if event_type == "email_opened" and state == self.INITIAL_CONTACT:
            self._transition(respondent_id, self.ENGAGED, "detected_engagement", "Email opened within 24h")
            return

        # --------------------------------------------------------------
        # Rule D: ENGAGED -> RESPONDED (reply received)
        # --------------------------------------------------------------
        if event_type == "reply_received" and state in (self.INITIAL_CONTACT, self.ENGAGED):
            self._transition(
                respondent_id,
                self.RESPONDED,
                "human_handoff",
                "ALERT: Sales team notified – stop all automation",
            )
            return

        # --------------------------------------------------------------
        # Rule F: RESPONDED -> SCHEDULED (interview confirmed)
        # --------------------------------------------------------------
        if event_type == "scheduled" and state == self.RESPONDED:
            self._transition(
                respondent_id,
                self.SCHEDULED,
                "interview_scheduled",
                "Calendar invite sent + 24h reminder queued",
            )
            return

        # --------------------------------------------------------------
        # Explicit decline -> UNSUBSCRIBED
        # --------------------------------------------------------------
        if event_type == "declined":
            self._transition(respondent_id, self.UNSUBSCRIBED, "unsubscribed", notes or "Respondent declined")
            return

        # --------------------------------------------------------------
        # Manual reactivation trigger (simulates external CRM signal)
        # --------------------------------------------------------------
        if event_type == "new_project_match" and state == self.NURTURE:
            subject = self._generate_email_subject(r["profile"])
            self._transition(
                respondent_id,
                self.INITIAL_CONTACT,
                "reactivation",
                f"New project match (external trigger) | Subject: {subject}",
            )
            return

        # Fallback: log the event without state change
        self._log_action(respondent_id, state, f"event_{event_type}", notes)
        print(f"  [{respondent_id}] Event '{event_type}' logged in state '{state}' (no transition)")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_simulation_log(self, filename: str = "simulation_log.csv") -> None:
        out_path = os.path.join(self._get_project_root(), "data", filename)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["day", "respondent_id", "state", "action_triggered", "notes"]
            )
            writer.writeheader()
            for entry in self.action_log:
                writer.writerow(entry)
        print(f"\nSimulation log saved to: {out_path}")

    def print_summary(self) -> None:
        """Print end-of-simulation summary statistics."""
        counts = {
            self.SCHEDULED: 0,
            self.NURTURE: 0,
            self.RESPONDED: 0,
            self.UNSUBSCRIBED: 0,
            self.INITIAL_CONTACT: 0,
            self.ENGAGED: 0,
            self.PARTICIPATED: 0,
            self.NEW: 0,
        }
        human_handoffs = 0
        total_touches = 0

        for rid, r in self.respondents.items():
            counts[r["state"]] += 1
            for h in r["history"]:
                total_touches += 1
                if h["action"] == "human_handoff":
                    human_handoffs += 1

        print("\n" + "=" * 60)
        print("NURTURE SEQUENCE SIMULATION – FINAL REPORT")
        print("=" * 60)
        for st, c in counts.items():
            if c:
                print(f"  {st:20s}: {c} respondents")
        print(f"\n  Total state transitions (touches): {total_touches}")
        print(f"  Human handoffs triggered:          {human_handoffs}")
        print(
            f"\n培育序列模拟完成：{counts[self.SCHEDULED]} 人完成预约，"
            f"{counts[self.NURTURE]} 人进入长期培育，"
            f"{human_handoffs} 次人工接管"
        )
        print("=" * 60)


# =====================================================================
# 14-Day Simulation
# =====================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Initialise engines
    # ------------------------------------------------------------------
    print("Loading lead scorer ...")
    scorer = LeadScoringModel()
    try:
        scorer.load_model()
    except FileNotFoundError:
        print("[ERROR] Model not found. Please run lead_scoring.py first.")
        sys.exit(1)

    print("Loading content engine ...")
    engine = PersonalizationEngine()
    if not engine.load_vector_store():
        docs = engine.load_knowledge_base()
        engine.build_vector_store(docs)

    # ------------------------------------------------------------------
    # 2. Create SequenceManager with simulated start date
    # ------------------------------------------------------------------
    sim_start = date(2024, 6, 1)
    manager = SequenceManager(scorer, engine, start_date=sim_start)

    # ------------------------------------------------------------------
    # 3. Load 3 real respondents from CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(manager._get_project_root(), "data", "raw", "respondents.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] {csv_path} not found.")
        sys.exit(1)

    # Pick 3 diverse profiles
    selected = df.iloc[[0, 250, 500]].copy()  # R001, R251, R501
    print(f"\nLoaded {len(selected)} respondent profiles for simulation.\n")

    for _, row in selected.iterrows():
        profile = {
            "industry": row["industry"],
            "job_level": row["job_level"],
            "region": row["region"],
            "company_size": row["company_size"],
            "past_participation_count": int(row["past_participation_count"]),
            "preferred_contact": row["preferred_contact"],
            "research_topic_match_score": int(row["research_topic_match_score"]),
            "is_hard_to_reach": int(row["is_hard_to_reach"]),
            "avg_historical_response_time": None,
        }
        score = scorer.predict_lead_score(profile)
        manager.add_respondent(row["respondent_id"], profile, score)

    # Hard-code respondent IDs for deterministic simulation
    RID_FAST = "R001"
    RID_ENGAGED_NURTURE = "R251"
    RID_NO_RESPONSE = "R501"
    rids = [RID_FAST, RID_ENGAGED_NURTURE, RID_NO_RESPONSE]

    # Verify IDs exist
    for rid in rids:
        if rid not in manager.respondents:
            print(f"[ERROR] {rid} not found in respondents. Available: {list(manager.respondents.keys())}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Run 14-day simulation loop
    # ------------------------------------------------------------------
    for day_offset in range(15):
        print(f"\n{'='*60}")
        print(f"DAY {day_offset:2d} | {manager.current_date} | Simulated Clock")
        print("=" * 60)

        # --- Morning: process automated triggers ---
        print("-- Trigger processing --")
        manager.process_daily_triggers()

        # --- Afternoon: inject mock external events ---
        print("-- Event injection --")

        if day_offset == 1:
            manager.record_event(RID_FAST, "email_opened")
            manager.record_event(RID_ENGAGED_NURTURE, "email_opened")

        if day_offset == 2:
            manager.record_event(RID_FAST, "reply_received")

        if day_offset == 3:
            manager.record_event(RID_FAST, "scheduled")

        if day_offset == 10:
            manager.record_event(RID_ENGAGED_NURTURE, "new_project_match", "CRM flagged new fintech AI study")

        # --- Evening: state snapshot ---
        print("-- End-of-day snapshot --")
        for rid in rids:
            st, days = manager.get_state(rid)
            hp = "★HIGH-PRIORITY" if manager.respondents[rid]["high_priority"] else "  standard"
            print(f"  {rid}: {st:20s} ({days}d) {hp}")

        # Advance clock
        manager.current_date += timedelta(days=1)

    # ------------------------------------------------------------------
    # 5. Finalise
    # ------------------------------------------------------------------
    manager.print_summary()
    manager.save_simulation_log()
    print("\nSimulation complete.")
