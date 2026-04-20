"""
RONIN Digital Research – Event-Driven Trigger Manager
Intelligent rule engine with multi-channel adapters and cooldown logic.
"""

import os
import sys
import json
import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Dict, List, Any

from nurture_sequences import SequenceManager
from lead_scoring import LeadScoringModel
from personalization_engine import PersonalizationEngine


# ------------------------------------------------------------------
# Channel Type Enum
# ------------------------------------------------------------------
class ChannelType(Enum):
    EMAIL = "email"
    LINKEDIN = "linkedin"
    WHATSAPP = "whatsapp"


# ------------------------------------------------------------------
# Base Channel Adapter (Abstract)
# ------------------------------------------------------------------
class BaseChannel(ABC):
    @abstractmethod
    def send_message(self, respondent_id: str, content: str, metadata: Optional[dict] = None) -> str:
        """Send a message and return a message ID."""
        pass

    @abstractmethod
    def get_delivery_status(self, message_id: str) -> str:
        """Query delivery status."""
        pass


# ------------------------------------------------------------------
# Concrete Adapters
# ------------------------------------------------------------------
class EmailAdapter(BaseChannel):
    def __init__(self):
        self.sent_messages: dict[str, dict] = {}
        self._counter = 0

    def send_message(self, respondent_id: str, content: str, metadata: Optional[dict] = None) -> str:
        self._counter += 1
        msg_id = f"email_{respondent_id}_{self._counter}"
        self.sent_messages[msg_id] = {
            "respondent_id": respondent_id,
            "content": content,
            "sent_time": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        print(f"    [EMAIL -> {respondent_id}] {content[:90]}...")
        return msg_id

    def get_delivery_status(self, message_id: str) -> str:
        return "delivered" if message_id in self.sent_messages else "unknown"


class LinkedInAdapter(BaseChannel):
    def __init__(self):
        self.sent_messages: dict[str, dict] = {}
        self._counter = 0

    def send_message(self, respondent_id: str, content: str, metadata: Optional[dict] = None) -> str:
        self._counter += 1
        msg_id = f"linkedin_{respondent_id}_{self._counter}"

        # Enforce < 100 words
        words = content.split()
        if len(words) > 100:
            content = " ".join(words[:95]) + " ..."

        # Casual tone hint
        content = content.replace("Dear ", "Hi ").replace("Best regards", "Cheers")

        # Social proof for hard-to-reach
        profile = (metadata or {}).get("profile", {})
        if profile.get("is_hard_to_reach") == 1:
            content += "\n    [LinkedIn Hint: Mention shared connection Dr. Sarah Chen]"

        self.sent_messages[msg_id] = {
            "respondent_id": respondent_id,
            "content": content,
            "sent_time": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        print(f"    [LINKEDIN -> {respondent_id}] {content[:90]}...")
        return msg_id

    def get_delivery_status(self, message_id: str) -> str:
        return "delivered" if message_id in self.sent_messages else "unknown"


class WhatsAppAdapter(BaseChannel):
    def __init__(self):
        self.sent_messages: dict[str, dict] = {}
        self._counter = 0

    def send_message(self, respondent_id: str, content: str, metadata: Optional[dict] = None) -> str:
        self._counter += 1
        msg_id = f"whatsapp_{respondent_id}_{self._counter}"

        # Enforce < 50 words
        words = content.split()
        if len(words) > 50:
            content = " ".join(words[:45]) + " ..."

        # Urgent tone
        content = f"[URGENT] {content}"

        self.sent_messages[msg_id] = {
            "respondent_id": respondent_id,
            "content": content,
            "sent_time": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        print(f"    [WHATSAPP -> {respondent_id}] {content[:90]}...")
        return msg_id

    def get_delivery_status(self, message_id: str) -> str:
        return "delivered" if message_id in self.sent_messages else "unknown"


# ------------------------------------------------------------------
# Trigger Rule Dataclass
# ------------------------------------------------------------------
@dataclass
class TriggerRule:
    name: str
    condition_func: Callable[[Any, str, str, Optional[dict]], bool]
    action_func: Callable[[Any, str, str, Optional[dict]], None]
    cooldown_hours: float = 24.0
    priority: int = 1
    channel: str = ""
    last_triggered: dict[str, datetime] = field(default_factory=dict, repr=False)


# ------------------------------------------------------------------
# Trigger Engine
# ------------------------------------------------------------------
class TriggerEngine:
    """
    Event-driven trigger engine that evaluates rules against inbound events
    and orchestrates multi-channel responses.
    """

    def __init__(self, sequence_manager: SequenceManager, channel_adapters: dict[str, BaseChannel]):
        self.sequence_manager = sequence_manager
        self.channel_adapters = channel_adapters
        self.rules: list[TriggerRule] = []
        self.event_history: list[dict] = []
        self.trigger_log: list[dict] = []
        self.respondent_events: dict[str, list[dict]] = {}
        self.failure_counts: dict[tuple[str, str], int] = {}
        self.manual_review_flags: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Rule registration
    # ------------------------------------------------------------------
    def register_trigger(
        self,
        name: str,
        condition_func: Callable,
        action_func: Callable,
        cooldown_hours: float = 24.0,
        priority: int = 1,
        channel: str = "",
    ) -> None:
        rule = TriggerRule(
            name=name,
            condition_func=condition_func,
            action_func=action_func,
            cooldown_hours=cooldown_hours,
            priority=priority,
            channel=channel,
        )
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
        print(f"[TRIGGER] Registered '{name}' (priority={priority}, cooldown={cooldown_hours}h)")

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------
    def process_event(
        self,
        respondent_id: str,
        event_type: str,
        metadata: Optional[dict] = None,
        timestamp: Optional[datetime] = None,
    ) -> list[str]:
        metadata = metadata or {}
        ts = timestamp or datetime.now()

        print(f"\n[EVENT] {ts.isoformat()} | {respondent_id} | {event_type}")

        # 1. Forward to SequenceManager if it is a state-changing event
        sm_events = {"email_opened", "link_clicked", "reply_received", "scheduled", "declined"}
        if event_type in sm_events:
            self.sequence_manager.record_event(respondent_id, event_type, metadata)

        # 2. Track locally
        entry = {
            "timestamp": ts.isoformat(),
            "respondent_id": respondent_id,
            "event_type": event_type,
            "metadata": metadata,
        }
        self.event_history.append(entry)
        self.respondent_events.setdefault(respondent_id, []).append({"time": ts, "type": event_type})

        # 3. Evaluate rules
        triggered: list[str] = []
        for rule in self.rules:
            # Condition check
            if not rule.condition_func(self, respondent_id, event_type, metadata):
                continue

            # Cooldown check (per-respondent)
            last_ts = rule.last_triggered.get(respondent_id)
            if last_ts is not None:
                hours_since = (ts - last_ts).total_seconds() / 3600.0
                if hours_since < rule.cooldown_hours:
                    print(f"  [SKIP] '{rule.name}' is in cooldown for {respondent_id} ({hours_since:.1f}h < {rule.cooldown_hours}h)")
                    continue

            # Execute action with failure tracking
            try:
                rule.action_func(self, respondent_id, event_type, metadata)
                rule.last_triggered[respondent_id] = ts
                triggered.append(rule.name)
                self._log_trigger(ts, respondent_id, event_type, rule.name, "executed", rule.channel)
                # Reset failures on success
                self.failure_counts.pop((rule.name, respondent_id), None)
            except Exception as exc:
                key = (rule.name, respondent_id)
                self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
                fail_count = self.failure_counts[key]
                self._log_trigger(ts, respondent_id, event_type, rule.name, f"failed ({fail_count}/3): {exc}", rule.channel)
                if fail_count >= 3:
                    self.manual_review_flags[respondent_id] = True
                    print(f"  [ALERT] {respondent_id} flagged for MANUAL REVIEW after 3 consecutive failures on '{rule.name}'")

        if not triggered:
            print(f"  [INFO] No rules triggered for this event")
        return triggered

    def _log_trigger(
        self,
        ts: datetime,
        respondent_id: str,
        event_type: str,
        rule_name: str,
        action_taken: str,
        channel: str,
    ) -> None:
        self.trigger_log.append({
            "timestamp": ts.isoformat(),
            "respondent_id": respondent_id,
            "event_type": event_type,
            "rule_triggered": rule_name,
            "action_taken": action_taken,
            "channel": channel,
        })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_trigger_log(self, filename: str = "trigger_log.csv") -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(root, "data", filename)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "respondent_id", "event_type", "rule_triggered", "action_taken", "channel"]
            )
            writer.writeheader()
            for row in self.trigger_log:
                writer.writerow(row)
        print(f"\nTrigger log saved to: {out_path}")

    def save_event_stream(self, events: list[dict], filename: str = "event_stream_test.json") -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_path = os.path.join(root, "data", filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        print(f"Event stream saved to: {out_path}")

    def print_summary(self) -> None:
        total_events = len(self.event_history)
        triggered_rules = len([r for r in self.trigger_log if r["action_taken"] == "executed"])
        channel_switches = len([r for r in self.trigger_log if r["channel"] != "" and "switch" in r["rule_triggered"].lower()])
        manual_reviews = len(self.manual_review_flags)

        print("\n" + "=" * 60)
        print("TRIGGER ENGINE TEST SUMMARY")
        print("=" * 60)
        print(f"  Total events processed:   {total_events}")
        print(f"  Rules triggered:          {triggered_rules}")
        print(f"  Channel switches:         {channel_switches}")
        print(f"  Manual review flags:      {manual_reviews}")
        print(
            f"\n触发器系统测试完成：共处理 {total_events} 个事件，"
            f"触发 {triggered_rules} 条规则，"
            f"{channel_switches} 次渠道切换"
        )
        print("=" * 60)


# =====================================================================
# Built-in Rule Factories
# =====================================================================
def build_rule_linkedin_followup():
    """Rule 1: Email opened but no reply -> LinkedIn follow-up after 6h."""
    def condition(engine, rid, event_type, metadata):
        if event_type != "email_opened":
            return False
        state, _ = engine.sequence_manager.get_state(rid)
        # Trigger after the first open (state may already be ENGAGED because
        # SequenceManager transitioned it) as long as no reply has been received.
        if state not in (SequenceManager.INITIAL_CONTACT, SequenceManager.ENGAGED):
            return False
        for e in engine.respondent_events.get(rid, []):
            if e["type"] == "reply_received":
                return False
        return True

    def action(engine, rid, event_type, metadata):
        adapter = engine.channel_adapters.get(ChannelType.LINKEDIN.value)
        if adapter:
            profile = engine.sequence_manager.respondents.get(rid, {}).get("profile", {})
            adapter.send_message(
                rid,
                "Hi there, I noticed you opened our research invitation. Would love to connect briefly here on LinkedIn to see if the study aligns with your priorities. Best, RONIN Team",
                metadata={"profile": profile},
            )
    return TriggerRule(
        name="linkedin_followup",
        condition_func=condition,
        action_func=action,
        cooldown_hours=48,
        priority=3,
        channel=ChannelType.LINKEDIN.value,
    )


def build_rule_fast_track():
    """Rule 2: Link clicked but not scheduled -> fast-track email."""
    def condition(engine, rid, event_type, metadata):
        if event_type != "link_clicked":
            return False
        state, _ = engine.sequence_manager.get_state(rid)
        if state != SequenceManager.ENGAGED:
            return False
        # Ensure not already scheduled
        for e in engine.respondent_events.get(rid, []):
            if e["type"] == "scheduled":
                return False
        return True

    def action(engine, rid, event_type, metadata):
        adapter = engine.channel_adapters.get(ChannelType.EMAIL.value)
        if adapter:
            adapter.send_message(
                rid,
                "Thanks for your interest! We've opened a fast-track lane for you – confirm your slot in just 2 clicks and skip the queue.",
            )
    return TriggerRule(
        name="fast_track_reminder",
        condition_func=condition,
        action_func=action,
        cooldown_hours=24,
        priority=5,
        channel=ChannelType.EMAIL.value,
    )


def build_rule_human_handoff():
    """Rule 3: Positive reply -> stop automation + alert researcher."""
    def condition(engine, rid, event_type, metadata):
        if event_type != "reply_received":
            return False
        return metadata.get("sentiment") == "positive"

    def action(engine, rid, event_type, metadata):
        print(f"    [ALERT] AUTOMATION STOPPED for {rid}. Sales team notified immediately.")
        # Also push through SequenceManager human_handoff
        engine.sequence_manager.record_event(rid, "reply_received", metadata)
    return TriggerRule(
        name="human_handoff",
        condition_func=condition,
        action_func=action,
        cooldown_hours=0,
        priority=10,
        channel="internal_alert",
    )


def build_rule_channel_switch():
    """Rule 4: 7 days no response -> switch to WhatsApp or last-chance email."""
    def condition(engine, rid, event_type, metadata):
        return event_type == "no_response_7days"

    def action(engine, rid, event_type, metadata):
        profile = engine.sequence_manager.respondents.get(rid, {}).get("profile", {})
        preferred = profile.get("preferred_contact", "Email")
        if preferred != "WhatsApp" and ChannelType.WHATSAPP.value in engine.channel_adapters:
            adapter = engine.channel_adapters[ChannelType.WHATSAPP.value]
            adapter.send_message(
                rid,
                "Quick reminder: your exclusive research invitation is still open. Reply YES to reserve your slot within 24h.",
            )
        else:
            adapter = engine.channel_adapters.get(ChannelType.EMAIL.value)
            if adapter:
                adapter.send_message(
                    rid,
                    "Last chance – your research invitation expires in 48 hours. Confirm now to secure your exclusive benchmark report.",
                )
    return TriggerRule(
        name="channel_switch_whatsapp",
        condition_func=condition,
        action_func=action,
        cooldown_hours=72,
        priority=2,
        channel=ChannelType.WHATSAPP.value,
    )


def build_rule_content_fatigue():
    """Rule 5: Multiple opens but no clicks -> flag content fatigue."""
    def condition(engine, rid, event_type, metadata):
        if event_type != "email_opened":
            return False
        events = engine.respondent_events.get(rid, [])
        if len(events) < 2:
            return False
        # Look back 3 days from latest event
        latest_ts = events[-1]["time"]
        window_start = latest_ts - timedelta(days=3)
        opens = [e for e in events if e["type"] == "email_opened" and e["time"] >= window_start]
        clicks = [e for e in events if e["type"] == "link_clicked" and e["time"] >= window_start]
        return len(opens) >= 2 and len(clicks) == 0

    def action(engine, rid, event_type, metadata):
        print(f"    [FLAG] {rid} marked as CONTENT FATIGUE. Next email will use shorter subject + urgency wording.")
        # Store flag in sequence manager metadata for future use
        if rid in engine.sequence_manager.respondents:
            engine.sequence_manager.respondents[rid].setdefault("flags", {})["content_fatigue"] = True
    return TriggerRule(
        name="content_fatigue_flag",
        condition_func=condition,
        action_func=action,
        cooldown_hours=0,
        priority=1,
        channel="",
    )


# =====================================================================
# Main execution – Event Stream Simulation
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RONIN Trigger Engine – Event Stream Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Initialise downstream systems
    # ------------------------------------------------------------------
    scorer = LeadScoringModel()
    scorer.load_model()

    engine_content = PersonalizationEngine()
    if not engine_content.load_vector_store():
        docs = engine_content.load_knowledge_base()
        engine_content.build_vector_store(docs)

    seq_mgr = SequenceManager(scorer, engine_content, start_date=datetime(2024, 6, 1).date())

    # ------------------------------------------------------------------
    # 2. Add 3 test respondents
    # ------------------------------------------------------------------
    test_profiles = {
        "R001": {
            "industry": "Healthcare",
            "job_level": "Manager",
            "region": "EMEA",
            "company_size": "Mid-market",
            "past_participation_count": 0,
            "preferred_contact": "Email",
            "research_topic_match_score": 73,
            "is_hard_to_reach": 0,
            "avg_historical_response_time": None,
        },
        "R251": {
            "industry": "Healthcare",
            "job_level": "C-Suite",
            "region": "North America",
            "company_size": "Mid-market",
            "past_participation_count": 0,
            "preferred_contact": "Email",
            "research_topic_match_score": 70,
            "is_hard_to_reach": 1,
            "avg_historical_response_time": None,
        },
        "R501": {
            "industry": "Technology",
            "job_level": "Director",
            "region": "EMEA",
            "company_size": "Enterprise",
            "past_participation_count": 0,
            "preferred_contact": "LinkedIn",
            "research_topic_match_score": 44,
            "is_hard_to_reach": 0,
            "avg_historical_response_time": None,
        },
    }

    for rid, profile in test_profiles.items():
        score = scorer.predict_lead_score(profile)
        seq_mgr.add_respondent(rid, profile, score)
        # Pre-seed to INITIAL_CONTACT (simulating that first email was already sent)
        seq_mgr.respondents[rid]["state"] = seq_mgr.INITIAL_CONTACT
        seq_mgr.respondents[rid]["state_entered_at"] = datetime(2024, 6, 1).date()

    # ------------------------------------------------------------------
    # 3. Initialise TriggerEngine + adapters
    # ------------------------------------------------------------------
    adapters = {
        ChannelType.EMAIL.value: EmailAdapter(),
        ChannelType.LINKEDIN.value: LinkedInAdapter(),
        ChannelType.WHATSAPP.value: WhatsAppAdapter(),
    }
    trigger_engine = TriggerEngine(seq_mgr, adapters)

    # Register built-in rules
    trigger_engine.register_trigger(
        **{k: v for k, v in vars(build_rule_linkedin_followup()).items() if k != "last_triggered"}
    )
    trigger_engine.register_trigger(
        **{k: v for k, v in vars(build_rule_fast_track()).items() if k != "last_triggered"}
    )
    trigger_engine.register_trigger(
        **{k: v for k, v in vars(build_rule_human_handoff()).items() if k != "last_triggered"}
    )
    trigger_engine.register_trigger(
        **{k: v for k, v in vars(build_rule_channel_switch()).items() if k != "last_triggered"}
    )
    trigger_engine.register_trigger(
        **{k: v for k, v in vars(build_rule_content_fatigue()).items() if k != "last_triggered"}
    )

    # ------------------------------------------------------------------
    # 4. Replay deterministic event stream
    # ------------------------------------------------------------------
    event_stream = [
        # R001: opens email, clicks link, replies positively
        {"time": "2024-06-01T09:00:00", "rid": "R001", "event": "email_sent"},
        {"time": "2024-06-01T11:00:00", "rid": "R001", "event": "email_opened"},
        {"time": "2024-06-01T16:00:00", "rid": "R001", "event": "link_clicked"},
        {"time": "2024-06-02T10:00:00", "rid": "R001", "event": "reply_received", "metadata": {"sentiment": "positive"}},

        # R251: opens email, then 7 days silence -> channel switch
        {"time": "2024-06-01T09:00:00", "rid": "R251", "event": "email_sent"},
        {"time": "2024-06-01T10:00:00", "rid": "R251", "event": "email_opened"},
        {"time": "2024-06-08T10:00:00", "rid": "R251", "event": "no_response_7days"},

        # R501: opens 3 times, never clicks -> content fatigue
        {"time": "2024-06-01T09:00:00", "rid": "R501", "event": "email_sent"},
        {"time": "2024-06-01T12:00:00", "rid": "R501", "event": "email_opened"},
        {"time": "2024-06-02T12:00:00", "rid": "R501", "event": "email_opened"},
        {"time": "2024-06-03T12:00:00", "rid": "R501", "event": "email_opened"},
    ]

    trigger_engine.save_event_stream(event_stream)

    for evt in event_stream:
        ts = datetime.fromisoformat(evt["time"])
        trigger_engine.process_event(
            respondent_id=evt["rid"],
            event_type=evt["event"],
            metadata=evt.get("metadata"),
            timestamp=ts,
        )

    # ------------------------------------------------------------------
    # 5. Finalise
    # ------------------------------------------------------------------
    trigger_engine.print_summary()
    trigger_engine.save_trigger_log()
    print("\nTrigger engine test complete.")
