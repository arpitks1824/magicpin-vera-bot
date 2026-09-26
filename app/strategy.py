"""
Message Strategy Layer — maps trigger + context into a structured strategy object.
The strategy determines what the message should achieve, not just what to say.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MessageStrategy:
    strategy_family: str
    mode: str
    why_now: str
    message_angle: str
    strongest_evidence_ids: List[str]
    desired_merchant_action: str
    cta_type: str  # "none" | "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot"
    send_as: str   # "vera" | "merchant_on_behalf"
    language_mode: str  # "en" | "hi" | "hi-en mix" | "te-en mix" etc.
    template_name: str
    should_send: bool = True
    suppression_reason: str = ""
    tone_notes: str = ""
    hook_type: str = ""  # "loss_aversion" | "curiosity" | "social_proof" | "reciprocity" | "specificity"
    max_length_words: int = 80


TEMPLATE_NAMES = {
    "RESEARCH_INSIGHT": "vera_research_v1",
    "COMPLIANCE_ALERT": "vera_compliance_v1",
    "PERFORMANCE_SPIKE": "vera_performance_v1",
    "PERFORMANCE_DIP": "vera_performance_v1",
    "SEASONAL_REFRAME": "vera_performance_v1",
    "ACTIVE_PLANNING": "vera_action_v1",
    "OFFER_OPPORTUNITY": "vera_action_v1",
    "MILESTONE": "vera_performance_v1",
    "REVIEW_PROBLEM": "vera_performance_v1",
    "COMPETITOR_CHANGE": "vera_performance_v1",
    "CURIOSITY_ASK": "vera_action_v1",
    "DORMANT_REACTIVATION": "vera_action_v1",
    "SUBSCRIPTION_RENEWAL": "vera_action_v1",
    "GBP_ACTION": "vera_action_v1",
    "CUSTOMER_RECALL": "merchant_recall_reminder_v1",
    "CUSTOMER_WINBACK": "vera_customer_winback_v1",
    "CUSTOMER_TRIAL_FOLLOWUP": "vera_customer_winback_v1",
    "CUSTOMER_REFILL": "vera_customer_recall_v1",
    "CUSTOMER_APPOINTMENT": "merchant_recall_reminder_v1",
    "CUSTOMER_EVENT": "merchant_recall_reminder_v1",
}

CTA_MAP = {
    "RESEARCH_INSIGHT": "open_ended",
    "COMPLIANCE_ALERT": "binary_yes_no",
    "PERFORMANCE_SPIKE": "open_ended",
    "PERFORMANCE_DIP": "open_ended",
    "SEASONAL_REFRAME": "open_ended",
    "ACTIVE_PLANNING": "binary_confirm_cancel",
    "OFFER_OPPORTUNITY": "binary_yes_no",
    "MILESTONE": "open_ended",
    "REVIEW_PROBLEM": "binary_yes_no",
    "COMPETITOR_CHANGE": "open_ended",
    "CURIOSITY_ASK": "open_ended",
    "DORMANT_REACTIVATION": "binary_yes_no",
    "SUBSCRIPTION_RENEWAL": "binary_confirm_cancel",
    "GBP_ACTION": "binary_yes_no",
    "CUSTOMER_RECALL": "multi_choice_slot",
    "CUSTOMER_WINBACK": "binary_yes_no",
    "CUSTOMER_TRIAL_FOLLOWUP": "binary_yes_no",
    "CUSTOMER_REFILL": "binary_confirm_cancel",
    "CUSTOMER_APPOINTMENT": "binary_confirm_cancel",
    "CUSTOMER_EVENT": "binary_confirm_cancel",
}

HOOK_MAP = {
    "RESEARCH_INSIGHT": "curiosity",
    "COMPLIANCE_ALERT": "loss_aversion",
    "PERFORMANCE_DIP": "loss_aversion",
    "PERFORMANCE_SPIKE": "reciprocity",
    "SEASONAL_REFRAME": "social_proof",
    "ACTIVE_PLANNING": "reciprocity",
    "OFFER_OPPORTUNITY": "loss_aversion",
    "COMPETITOR_CHANGE": "loss_aversion",
    "CUSTOMER_WINBACK": "reciprocity",
    "SUBSCRIPTION_RENEWAL": "loss_aversion",
    "DORMANT_REACTIVATION": "curiosity",
    "CURIOSITY_ASK": "curiosity",
}


def build_strategy(
    strategy_family: str,
    mode: str,
    merchant: Dict,
    category: Dict,
    trigger: Dict,
    customer: Optional[Dict],
    evidence_ids: List[str],
) -> MessageStrategy:
    """
    Build a complete MessageStrategy from the ranked trigger + context.
    """
    identity = merchant.get("identity", {})
    trg_scope = trigger.get("scope", "merchant")
    kind = trigger.get("kind", "")

    # Language mode
    if customer:
        lang_pref = customer.get("identity", {}).get("language_pref", "en")
    else:
        languages = identity.get("languages", ["en"])
        lang_pref = _infer_language_mode(languages)

    # send_as
    if trg_scope == "customer" and customer:
        send_as = "merchant_on_behalf"
    else:
        send_as = "vera"

    # CTA
    cta_type = CTA_MAP.get(strategy_family, "open_ended")
    if mode == "action_continuation":
        cta_type = "binary_confirm_cancel"

    # Hook
    hook_type = HOOK_MAP.get(strategy_family, "specificity")

    # Why now
    why_now = _why_now(strategy_family, trigger, merchant, customer)
    message_angle = _message_angle(strategy_family, mode, merchant, trigger, customer)
    desired_action = _desired_action(strategy_family, mode, merchant)
    tone_notes = _tone_notes(strategy_family, category, customer)

    # Template
    template_name = TEMPLATE_NAMES.get(strategy_family, "vera_action_v1")

    # Strongest evidence
    strong_ids = _strongest_evidence(strategy_family, evidence_ids, trigger)

    return MessageStrategy(
        strategy_family=strategy_family,
        mode=mode,
        why_now=why_now,
        message_angle=message_angle,
        strongest_evidence_ids=strong_ids,
        desired_merchant_action=desired_action,
        cta_type=cta_type,
        send_as=send_as,
        language_mode=lang_pref,
        template_name=template_name,
        hook_type=hook_type,
        tone_notes=tone_notes,
    )


def _infer_language_mode(languages: List[str]) -> str:
    has_hi = "hi" in languages
    has_en = "en" in languages
    has_te = "te" in languages
    has_ta = "ta" in languages
    has_kn = "kn" in languages
    has_mr = "mr" in languages

    if has_hi and has_en:
        return "hi-en mix"
    if has_te and has_en:
        return "te-en mix"
    if has_ta and has_en:
        return "ta-en mix"
    if has_kn and has_en:
        return "kn-en mix"
    if has_mr and has_en:
        return "mr-en mix"
    if has_en:
        return "en"
    return "en"


def _why_now(
    strategy: str, trigger: Dict, merchant: Dict, customer: Optional[Dict]
) -> str:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})

    if strategy == "RESEARCH_INSIGHT":
        return f"New research/compliance item available (trigger: {kind})"
    if strategy == "COMPLIANCE_ALERT":
        deadline = payload.get("deadline_iso", payload.get("deadline", ""))
        return f"Compliance action required" + (f" by {deadline}" if deadline else "")
    if strategy == "PERFORMANCE_DIP":
        metric = payload.get("metric", "views")
        delta = payload.get("delta_pct", 0)
        return f"{metric} dropped {abs(delta)*100:.0f}% this week"
    if strategy == "PERFORMANCE_SPIKE":
        metric = payload.get("metric", "views")
        delta = payload.get("delta_pct", 0)
        return f"{metric} spiked +{delta*100:.0f}% this week"
    if strategy == "SEASONAL_REFRAME":
        return "Expected seasonal pattern — reframe needed to avoid unnecessary worry"
    if strategy == "ACTIVE_PLANNING":
        topic = payload.get("intent_topic", "business expansion")
        return f"Merchant expressed explicit planning intent on: {topic}"
    if strategy == "OFFER_OPPORTUNITY":
        return f"External event creates engagement window: {kind}"
    if strategy == "CUSTOMER_RECALL":
        due = payload.get("due_date", "")
        return f"6-month recall due" + (f" — {due}" if due else "")
    if strategy == "CUSTOMER_REFILL":
        runs_out = payload.get("stock_runs_out_iso", "")
        return f"Chronic prescription running out" + (f" on {runs_out[:10]}" if runs_out else "")
    if strategy == "CUSTOMER_WINBACK":
        days = payload.get("days_since_last_visit", 0)
        return f"Customer lapsed {days} days ago"
    if strategy == "SUBSCRIPTION_RENEWAL":
        days = payload.get("days_remaining", 0)
        return f"Subscription expires in {days} days"
    if strategy == "DORMANT_REACTIVATION":
        days = payload.get("days_since_last_merchant_message", 0)
        return f"No merchant engagement for {days} days"
    return f"Trigger: {kind}"


def _message_angle(
    strategy: str, mode: str, merchant: Dict, trigger: Dict, customer: Optional[Dict]
) -> str:
    if mode == "action_continuation":
        return "Continue from merchant's explicit commitment — show artifact / execution step"

    angles = {
        "RESEARCH_INSIGHT": "Share verifiable finding + relevance to their specific patient/customer profile + offer to prepare an artifact",
        "COMPLIANCE_ALERT": "State exact compliance requirement + deadline + practical next step + offer to draft workflow",
        "PERFORMANCE_DIP": "Anchor on specific metric + compare to baseline + propose concrete action",
        "PERFORMANCE_SPIKE": "Acknowledge win + identify likely driver + propose capitalisation action",
        "SEASONAL_REFRAME": "Pre-empt merchant anxiety + explain seasonal norm + redirect to retention/planning",
        "OFFER_OPPORTUNITY": "Connect external event to merchant's active offer + concrete execution path",
        "CUSTOMER_RECALL": "Personal recall with exact slot options matching customer's preferences",
        "CUSTOMER_WINBACK": "Warm, no-shame reconnect using customer's prior goal + specific new offering",
        "CUSTOMER_REFILL": "Exact medicines + exact due date + active offer + simple confirmation path",
        "SUBSCRIPTION_RENEWAL": "Show what they'll lose if not renewed + make renewal single action",
        "DORMANT_REACTIVATION": "Light, curious re-engagement — reference something new since last interaction",
        "GBP_ACTION": "Specific GBP gap + estimated uplift + clear single action",
    }
    return angles.get(strategy, "Specific data point + why it matters + one clear next step")


def _desired_action(strategy: str, mode: str, merchant: Dict) -> str:
    if mode == "action_continuation":
        return "Merchant confirms execution"
    actions = {
        "RESEARCH_INSIGHT": "Merchant requests abstract or draft patient content",
        "COMPLIANCE_ALERT": "Merchant acknowledges and begins compliance workflow",
        "PERFORMANCE_DIP": "Merchant commits to one corrective action",
        "PERFORMANCE_SPIKE": "Merchant capitalises with a follow-up campaign",
        "ACTIVE_PLANNING": "Merchant confirms plan and next step",
        "OFFER_OPPORTUNITY": "Merchant activates or approves offer",
        "CUSTOMER_RECALL": "Customer books slot (Reply 1/2 or gives availability)",
        "CUSTOMER_WINBACK": "Customer agrees to try again",
        "CUSTOMER_REFILL": "Customer confirms delivery",
        "SUBSCRIPTION_RENEWAL": "Merchant confirms renewal",
    }
    return actions.get(strategy, "Merchant replies with intent")


def _tone_notes(strategy: str, category: Dict, customer: Optional[Dict]) -> str:
    cat_voice = category.get("voice", {})
    tone = cat_voice.get("tone", "professional")
    taboos = cat_voice.get("vocab_taboo", [])

    note = f"Tone: {tone}."
    if taboos:
        note += f" Avoid: {', '.join(taboos[:3])}."
    if strategy == "COMPLIANCE_ALERT":
        note += " Be calm and precise — no alarm language."
    if strategy == "CUSTOMER_WINBACK":
        note += " No shame, no guilt. Warm and non-judgmental."
    if strategy == "SEASONAL_REFRAME":
        note += " Reassure first, then advise."
    if customer:
        age = customer.get("identity", {}).get("age_band", "")
        if "65" in age or "senior" in str(customer.get("identity", {}).get("senior_citizen", "")):
            note += " Message via son/caregiver WhatsApp — respectful tone."
    return note


def _strongest_evidence(
    strategy: str, evidence_ids: List[str], trigger: Dict
) -> List[str]:
    """Identify the 3-5 most critical evidence IDs for this strategy."""
    payload = trigger.get("payload", {})
    strong = []

    # Always include trigger payload evidence
    for k in ["top_item_id", "metric", "delta_pct", "days_remaining",
              "affected_batches", "molecule_list", "molecule", "available_slots",
              "deadline_iso", "intent_topic", "days_since_last_visit"]:
        eid = f"trigger.payload.{k}"
        if payload.get(k) is not None and eid in evidence_ids:
            strong.append(eid)

    # Strategy-specific
    if strategy == "RESEARCH_INSIGHT":
        strong += [e for e in evidence_ids if "digest" in e and "source" in e][:2]
    elif strategy == "COMPLIANCE_ALERT":
        strong += [e for e in evidence_ids if "deadline" in e or "batch" in e][:2]
    elif strategy in ("PERFORMANCE_DIP", "PERFORMANCE_SPIKE"):
        strong += [e for e in evidence_ids if "perf" in e or "delta" in e][:2]

    return list(dict.fromkeys(strong))[:5]  # deduplicate, keep order
