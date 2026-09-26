"""
Signal Ranker — scores and ranks available triggers deterministically.
Selects the best trigger(s) to act on for each tick.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .state import suppression_store, conversation_store


# ── Strategy families ──────────────────────────────────────────────────────

STRATEGY_MAP = {
    # External merchant triggers
    "research_digest": "RESEARCH_INSIGHT",
    "regulation_change": "COMPLIANCE_ALERT",
    "festival_upcoming": "OFFER_OPPORTUNITY",
    "weather_heatwave": "SEASONAL_REFRAME",
    "local_news_event": "OFFER_OPPORTUNITY",
    "category_research_digest_release": "RESEARCH_INSIGHT",
    "competitor_opened": "COMPETITOR_CHANGE",
    "category_trend_movement": "RESEARCH_INSIGHT",
    "ipl_match_today": "OFFER_OPPORTUNITY",
    "category_seasonal": "SEASONAL_REFRAME",
    "cde_opportunity": "RESEARCH_INSIGHT",

    # Internal merchant triggers
    "perf_spike": "PERFORMANCE_SPIKE",
    "perf_dip": "PERFORMANCE_DIP",
    "seasonal_perf_dip": "SEASONAL_REFRAME",
    "milestone_reached": "MILESTONE",
    "dormant_with_vera": "DORMANT_REACTIVATION",
    "review_theme_emerged": "REVIEW_PROBLEM",
    "supply_alert": "COMPLIANCE_ALERT",
    "scheduled_recurring": "CURIOSITY_ASK",
    "curious_ask_due": "CURIOSITY_ASK",
    "renewal_due": "SUBSCRIPTION_RENEWAL",
    "gbp_unverified": "GBP_ACTION",
    "winback_eligible": "DORMANT_REACTIVATION",
    "active_planning_intent": "ACTIVE_PLANNING",

    # Customer triggers
    "recall_due": "CUSTOMER_RECALL",
    "customer_lapsed_soft": "CUSTOMER_WINBACK",
    "customer_lapsed_hard": "CUSTOMER_WINBACK",
    "appointment_tomorrow": "CUSTOMER_APPOINTMENT",
    "chronic_refill_due": "CUSTOMER_REFILL",
    "trial_followup": "CUSTOMER_TRIAL_FOLLOWUP",
    "wedding_package_followup": "CUSTOMER_EVENT",
}

URGENCY_BOOST = {
    "COMPLIANCE_ALERT": 0.30,
    "ACTIVE_PLANNING": 0.25,
    "SUBSCRIPTION_RENEWAL": 0.20,
    "CUSTOMER_REFILL": 0.20,
    "CUSTOMER_RECALL": 0.15,
    "PERFORMANCE_DIP": 0.10,
    "PERFORMANCE_SPIKE": 0.05,
    "CUSTOMER_WINBACK": 0.10,
    "CUSTOMER_TRIAL_FOLLOWUP": 0.10,
    "COMPETITOR_CHANGE": 0.10,
    "REVIEW_PROBLEM": 0.10,
    "GBP_ACTION": 0.05,
    "SUBSCRIPTION_RENEWAL": 0.15,
    "RESEARCH_INSIGHT": 0.05,
    "SEASONAL_REFRAME": 0.00,
    "DORMANT_REACTIVATION": 0.05,
    "MILESTONE": 0.00,
    "OFFER_OPPORTUNITY": 0.05,
    "CURIOSITY_ASK": 0.00,
    "CUSTOMER_APPOINTMENT": 0.20,
    "CUSTOMER_EVENT": 0.10,
}


@dataclass
class RankedTrigger:
    trigger_id: str
    trigger: Dict
    merchant: Dict
    category: Dict
    customer: Optional[Dict]
    priority: float
    strategy: str
    mode: str
    reason: str
    evidence_ids: List[str] = field(default_factory=list)
    desired_outcome: str = ""
    should_send: bool = True
    suppress_reason: str = ""

    def to_plan(self) -> dict:
        return {
            "trigger_id": self.trigger_id,
            "priority": round(self.priority, 3),
            "strategy": self.strategy,
            "mode": self.mode,
            "reason": self.reason,
            "evidence_ids": self.evidence_ids,
            "desired_outcome": self.desired_outcome,
            "should_send": self.should_send,
        }


class SignalRanker:
    """
    Deterministic trigger scorer.
    Evaluates every available trigger against merchant context and conversation state.
    """

    def rank(
        self,
        available_trigger_ids: List[str],
        context_store,
        now_str: str,
    ) -> List[RankedTrigger]:
        """
        Rank all available triggers. Returns sorted list (highest priority first).
        Only includes triggers where should_send=True.
        """
        results = []
        now = _parse_dt(now_str)

        for trg_id in available_trigger_ids:
            trg = context_store.get("trigger", trg_id)
            if not trg:
                continue

            # Basic expiry check
            expires = trg.get("expires_at")
            if expires and now:
                exp_dt = _parse_dt(expires)
                if exp_dt and now > exp_dt:
                    continue  # expired

            merchant_id = trg.get("merchant_id")
            if not merchant_id:
                continue
            merchant = context_store.get("merchant", merchant_id)
            if not merchant:
                continue

            cat_slug = merchant.get("category_slug", "")
            category = context_store.get("category", cat_slug) or {}

            customer_id = trg.get("customer_id")
            customer = None
            if customer_id:
                customer = context_store.get("customer", customer_id)

            ranked = self._score_trigger(
                trg_id, trg, merchant, category, customer, now
            )
            results.append(ranked)

        results.sort(key=lambda r: -r.priority)
        return results

    def _score_trigger(
        self,
        trg_id: str,
        trg: Dict,
        merchant: Dict,
        category: Dict,
        customer: Optional[Dict],
        now: Optional[datetime],
    ) -> RankedTrigger:
        kind = trg.get("kind", "unknown")
        urgency = trg.get("urgency", 1)  # 1-5
        strategy = STRATEGY_MAP.get(kind, "RESEARCH_INSIGHT")
        trg_scope = trg.get("scope", "merchant")

        # Start with base score from urgency (0.1 to 0.5)
        score = urgency * 0.10

        # Strategy boost
        score += URGENCY_BOOST.get(strategy, 0.0)

        # Suppress check
        sup_key = trg.get("suppression_key", "")
        if sup_key and suppression_store.is_suppressed(sup_key):
            return RankedTrigger(
                trigger_id=trg_id, trigger=trg, merchant=merchant,
                category=category, customer=customer,
                priority=0.0, strategy=strategy, mode="suppressed",
                reason="Already sent — suppression key active",
                should_send=False, suppress_reason="suppression_key_active"
            )

        # Conversation opt-out check
        conv_id = _stable_conv_id(merchant.get("merchant_id", ""), trg_id, strategy, trg.get("scope", "merchant"))
        if conversation_store.is_opted_out(conv_id):
            return RankedTrigger(
                trigger_id=trg_id, trigger=trg, merchant=merchant,
                category=category, customer=customer,
                priority=0.0, strategy=strategy, mode="opted_out",
                reason="Merchant opted out",
                should_send=False, suppress_reason="opted_out"
            )

        # Customer consent check for customer-scoped triggers
        if trg_scope == "customer" and customer:
            consent = customer.get("consent", {})
            consent_scope = consent.get("scope", [])
            opted_in = bool(consent.get("opted_in_at"))
            if not opted_in or not consent_scope:
                return RankedTrigger(
                    trigger_id=trg_id, trigger=trg, merchant=merchant,
                    category=category, customer=customer,
                    priority=0.0, strategy=strategy, mode="no_consent",
                    reason="Customer has not opted in",
                    should_send=False, suppress_reason="no_consent"
                )
            # Validate consent scope matches trigger kind
            kind_to_consent = {
                "recall_due": "recall_reminders",
                "chronic_refill_due": "refill_reminders",
                "appointment_tomorrow": "appointment_reminders",
                "trial_followup": "kids_program_updates",
                "customer_lapsed_soft": "promotional_offers",
                "customer_lapsed_hard": "winback_offers",
                "wedding_package_followup": "bridal_package_followup",
            }
            required = kind_to_consent.get(kind)
            # Accept if required scope is present OR consent is broad
            if required and required not in consent_scope:
                broad_scopes = {"promotional_offers", "recall_reminders", "refill_reminders"}
                if not any(s in consent_scope for s in broad_scopes):
                    # Still allow if reminder_opt_in is True in preferences
                    if not customer.get("preferences", {}).get("reminder_opt_in", False):
                        return RankedTrigger(
                            trigger_id=trg_id, trigger=trg, merchant=merchant,
                            category=category, customer=customer,
                            priority=0.0, strategy=strategy, mode="consent_mismatch",
                            reason=f"Consent scope '{required}' not found",
                            should_send=False, suppress_reason="consent_mismatch"
                        )

        # Evidence richness bonus
        payload = trg.get("payload", {})
        evidence_count = _count_evidence(payload)
        score += min(0.10, evidence_count * 0.02)

        # Conversation history relevance
        conv_history = merchant.get("conversation_history", [])
        signals = merchant.get("signals", [])

        # ACTIVE_PLANNING: highest effective urgency
        if strategy == "ACTIVE_PLANNING":
            # Check if merchant already expressed explicit intent
            last_merchant_msg = ""
            for turn in reversed(conv_history):
                if turn.get("from") == "merchant":
                    last_merchant_msg = turn.get("body", "").lower()
                    break
            if any(w in last_merchant_msg for w in [
                "yes", "good idea", "what would", "let's do", "go ahead",
                "send it", "what next", "proceed"
            ]):
                score += 0.20  # very high boost for explicit intent

        # Seasonal dip reframe: reduce alarm if marked expected
        if strategy == "SEASONAL_REFRAME":
            if payload.get("is_expected_seasonal"):
                score *= 0.8  # still worth sending but not urgent

        # Novelty: check if this topic was already sent recently
        mode = _infer_mode(strategy, conv_history, signals, payload)

        # Subscription expiry: boost if near deadline
        if strategy == "SUBSCRIPTION_RENEWAL":
            days_left = payload.get("days_remaining", 100)
            if days_left <= 7:
                score += 0.20
            elif days_left <= 14:
                score += 0.10

        # Engagement quality of last conversation
        for turn in conv_history:
            eng = turn.get("engagement", "")
            if eng == "merchant_replied":
                score += 0.05
            elif eng == "intent_action":
                score += 0.10

        # Recency penalty: penalise if there was a very recent message on same topic
        # (checked via signals)
        if "engaged_in_last_24h" in signals and strategy == "CURIOSITY_ASK":
            score *= 0.6  # don't spam recently-engaged merchants with curiosity asks

        # Clamp score
        score = max(0.01, min(0.99, score))

        # Build evidence IDs for explanation
        evidence_ids = [
            f"trigger.payload.{k}" for k in payload if payload[k] is not None
        ]
        if trg.get("suppression_key"):
            evidence_ids.append("trigger.suppression_key")

        reason = _build_reason(strategy, merchant, trg, customer, payload)
        desired_outcome = _desired_outcome(strategy, mode)

        return RankedTrigger(
            trigger_id=trg_id, trigger=trg, merchant=merchant,
            category=category, customer=customer,
            priority=score, strategy=strategy, mode=mode,
            reason=reason, evidence_ids=evidence_ids,
            desired_outcome=desired_outcome, should_send=True
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _count_evidence(payload: dict, depth: int = 0) -> int:
    if depth > 3:
        return 0
    count = 0
    for v in payload.values():
        if v is not None and not isinstance(v, bool):
            if isinstance(v, dict):
                count += _count_evidence(v, depth + 1)
            elif isinstance(v, list):
                count += len(v)
            else:
                count += 1
    return count


def _infer_mode(
    strategy: str,
    conv_history: List[dict],
    signals: List[str],
    payload: dict,
) -> str:
    """Infer the conversational mode from context."""
    # Check for active planning intent
    for turn in reversed(conv_history):
        if turn.get("from") == "merchant":
            body = turn.get("body", "").lower()
            eng = turn.get("engagement", "")
            if eng in ("intent_action", "intent_planning", "intent_question"):
                return "action_continuation"
            if any(w in body for w in ["yes", "go ahead", "do it", "let's", "what would"]):
                return "action_continuation"

    if strategy == "ACTIVE_PLANNING":
        return "action_continuation"
    if strategy in ("COMPLIANCE_ALERT",):
        return "urgent_action"
    if strategy in ("CUSTOMER_RECALL", "CUSTOMER_REFILL", "CUSTOMER_WINBACK"):
        return "customer_outreach"
    if strategy in ("RESEARCH_INSIGHT",):
        return "knowledge_share"
    if strategy in ("SEASONAL_REFRAME",):
        return "reframe"
    if strategy in ("DORMANT_REACTIVATION",):
        return "re_engagement"
    return "proactive_nudge"


def _build_reason(
    strategy: str,
    merchant: Dict,
    trg: Dict,
    customer: Optional[Dict],
    payload: Dict,
) -> str:
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("name", "merchant"))
    kind = trg.get("kind", "")
    urgency = trg.get("urgency", 1)

    templates = {
        "RESEARCH_INSIGHT": f"New research/digest item relevant to {owner}'s practice (urgency {urgency})",
        "COMPLIANCE_ALERT": f"Compliance requirement affecting {owner} — deadline-bound (urgency {urgency})",
        "PERFORMANCE_SPIKE": f"{owner}'s performance spiked — acknowledgment + capitalisation opportunity",
        "PERFORMANCE_DIP": f"{owner}'s metrics dipped — intervention recommended",
        "SEASONAL_REFRAME": f"Expected seasonal pattern; reframe rather than alarm for {owner}",
        "ACTIVE_PLANNING": f"{owner} expressed explicit planning intent — switch to execution mode",
        "OFFER_OPPORTUNITY": f"External event creates offer opportunity for {owner}",
        "MILESTONE": f"{owner} is approaching/reached a milestone",
        "REVIEW_PROBLEM": f"Review pattern emerged for {owner} — needs attention",
        "COMPETITOR_CHANGE": f"Competitor change detected near {owner}",
        "CURIOSITY_ASK": f"Cadence curiosity-ask due for {owner}",
        "DORMANT_REACTIVATION": f"{owner} has been dormant — re-engagement opportunity",
        "CUSTOMER_RECALL": f"Recall window open for customer of {owner}",
        "CUSTOMER_WINBACK": f"Lapsed customer of {owner} — warm winback",
        "CUSTOMER_TRIAL_FOLLOWUP": f"Trial followup for {owner}'s customer",
        "CUSTOMER_REFILL": f"Chronic refill due for {owner}'s customer",
        "CUSTOMER_APPOINTMENT": f"Appointment reminder for {owner}'s customer",
        "CUSTOMER_EVENT": f"Customer event trigger for {owner}",
        "GBP_ACTION": f"GBP verification/action needed for {owner}",
        "SUBSCRIPTION_RENEWAL": f"{owner}'s subscription needs renewal",
    }
    return templates.get(strategy, f"Trigger '{kind}' for {owner}")


def _desired_outcome(strategy: str, mode: str) -> str:
    if mode == "action_continuation":
        return "move directly to execution — merchant already committed"
    outcomes = {
        "RESEARCH_INSIGHT": "merchant curiosity triggered → request abstract or draft",
        "COMPLIANCE_ALERT": "merchant acknowledges and initiates compliance action",
        "PERFORMANCE_SPIKE": "merchant attributes and capitalises on spike",
        "PERFORMANCE_DIP": "merchant understands cause and takes corrective action",
        "SEASONAL_REFRAME": "merchant reframes expectation → plans appropriately",
        "ACTIVE_PLANNING": "merchant confirms plan and next step",
        "OFFER_OPPORTUNITY": "merchant activates or creates an offer",
        "MILESTONE": "merchant acknowledges and leverages milestone",
        "REVIEW_PROBLEM": "merchant commits to addressing review theme",
        "DORMANT_REACTIVATION": "merchant re-engages with Vera",
        "CUSTOMER_RECALL": "customer books appointment",
        "CUSTOMER_WINBACK": "lapsed customer returns",
        "CUSTOMER_REFILL": "customer confirms refill",
        "GBP_ACTION": "merchant completes GBP verification",
        "SUBSCRIPTION_RENEWAL": "merchant renews subscription",
    }
    return outcomes.get(strategy, "merchant responds and takes next step")


def _stable_conv_id(merchant_id: str, trigger_id: str, strategy: str, scope: str) -> str:
    """Generate a stable conversation ID."""
    # Sanitise components
    mid = re.sub(r"[^a-z0-9_]", "_", merchant_id.lower())[:20]
    strat = strategy.lower()[:15]
    # Use a key portion of the trigger id
    trg_key = re.sub(r"[^a-z0-9_]", "_", trigger_id.lower())[-20:]
    return f"conv_{mid}_{strat}_{trg_key}"
