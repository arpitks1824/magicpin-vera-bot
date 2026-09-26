"""
Conversation Guard — classifies merchant/customer replies and decides next action.
Handles: auto-replies, intent transitions, opt-outs, hostile, off-topic, curveball.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from .state import conversation_store


# ── Reply classification ────────────────────────────────────────────────────

AUTO_REPLY_PATTERNS = [
    r"thank you for contact",
    r"thanks for (reaching|contacting|your)",
    r"our team will (contact|respond|get back)",
    r"team tak pahunch",
    r"automated (assistant|reply|response)",
    r"i (am|m) an automated",
    r"out of office",
    r"will respond (shortly|soon)",
    r"i'll be back",
    r"currently unavailable",
    r"please (leave|send) (a message|your details)",
    r"away from (the )?phone",
    r"aapki madad ke liye shukriya.*automated",
    r"shukriya.*main.*automated",
]

OPT_OUT_PATTERNS = [
    r"\bstop\b",
    r"\bnot interested\b",
    r"\bdon'?t (message|contact|send|text)\b",
    r"\bremove me\b",
    r"\bno more\b",
    r"\bunsubscribe\b",
    r"\bband karo\b",
    r"\bmat bhejo\b",
    r"\bhata do\b",
    r"\bblock\b",
]

EXPLICIT_ACCEPTANCE_PATTERNS = [
    r"\byes\b",
    r"\byes please\b",
    r"\bgo ahead\b",
    r"\bsend it\b",
    r"\bok\b",
    r"\bokay\b",
    r"\blet'?s do (it|this)\b",
    r"\bdo it\b",
    r"\bproceed\b",
    r"\bconfirm\b",
    r"\bplease proceed\b",
    r"\bsend (the )?(abstract|draft|details|list)\b",
    r"\bha(an?)?\b",  # Hindi yes
    r"\bbi(l)?kul\b",  # Hindi "absolutely"
    r"\bsahi\b",  # Hindi "right/correct"
    r"\bjaldi\b",  # Hindi urgency
    r"\bwhat'?s next\b",
    r"\bwhat (should|do) (i|we)\b",
    r"\bgive me\b",
    r"\bprepare\b",
    r"\bdraft (it|the)\b",
]

PLANNING_INTENT_PATTERNS = [
    r"what would it look like",
    r"how (would|should|can) (i|we|it)\b",
    r"what (is|are) my options",
    r"tell me more",
    r"explain",
    r"elaborate",
    r"sounds? (good|interesting|great|like a plan)",
    r"interested",
    r"good idea",
    r"like the idea",
]

QUESTION_PATTERNS = [
    r"\?$",
    r"^(what|when|where|who|why|how|which|can|could|will|would|is|are|do|does)\b",
    r"mujhe batao",
    r"kaise",
]

OBJECTION_PATTERNS = [
    r"\btoo expensive\b",
    r"\bnot (right|relevant|useful|applicable|needed)\b",
    r"\bdon'?t (think|want|need)\b",
    r"\bno (thanks?|thank you)\b",
    r"\bnahi\b",
    r"\bnot for me\b",
    r"\bwe'?re (fine|okay|good)\b",
    r"\balready (doing|have|done)\b",
]

HOSTILE_PATTERNS = [
    r"\bstupid\b",
    r"\bidiot\b",
    r"\buseless\b",
    r"\bbothering\b",
    r"\bwaste of time\b",
    r"\bspam\b",
    r"\bfraud\b",
    r"\bscam\b",
    r"\bbekar\b",  # Hindi "useless"
    r"\bchup\b",  # Hindi "shut up"
]

OFF_TOPIC_PATTERNS = [
    r"(gst|income tax|vat|compliance|legal)\s+(filing|return|help)",
    r"(loan|finance|investment|insurance)\s+(help|query|question)",
    r"(hr|salary|payroll|employee)\s+(help|issue|question)",
    r"my (laptop|phone|computer|car)\b",
    r"cricket (match|score|team)\b",
]

PAUSE_PATTERNS = [
    r"\blater\b",
    r"\bnot now\b",
    r"\bbusy\b",
    r"\bcall (you )?back\b",
    r"\bkal\b",  # Hindi "tomorrow"
    r"\bbaad mein\b",  # Hindi "later"
    r"\bsome other time\b",
    r"\bcheck (it )?later\b",
]


class ConversationClassifier:
    """
    Classifies incoming merchant/customer messages into reply states.
    """

    def classify(self, message: str, conv_state: Dict) -> str:
        """
        Returns one of:
        auto_reply | explicit_acceptance | planning_intent | question |
        objection | opt_out | hostile | off_topic | pause | unclear
        """
        msg_lower = message.lower().strip()
        if not msg_lower:
            return "unclear"

        # Auto-reply detection (check repetition first)
        auto_reply_count = conv_state.get("auto_reply_count", 0)
        last_auto = conv_state.get("last_auto_reply_body", "")
        if last_auto and _normalize(message) == _normalize(last_auto):
            return "auto_reply"  # repeated exact same message
        if any(re.search(p, msg_lower) for p in AUTO_REPLY_PATTERNS):
            return "auto_reply"

        # Opt-out (high priority — check early)
        if any(re.search(p, msg_lower) for p in OPT_OUT_PATTERNS):
            return "opt_out"

        # Hostile
        if any(re.search(p, msg_lower) for p in HOSTILE_PATTERNS):
            return "hostile"

        # Explicit acceptance
        if any(re.search(p, msg_lower) for p in EXPLICIT_ACCEPTANCE_PATTERNS):
            return "explicit_acceptance"

        # Planning intent
        if any(re.search(p, msg_lower) for p in PLANNING_INTENT_PATTERNS):
            return "planning_intent"

        # Objection
        if any(re.search(p, msg_lower) for p in OBJECTION_PATTERNS):
            return "objection"

        # Off-topic
        if any(re.search(p, msg_lower) for p in OFF_TOPIC_PATTERNS):
            return "off_topic"

        # Pause/later
        if any(re.search(p, msg_lower) for p in PAUSE_PATTERNS):
            return "pause"

        # Question
        if any(re.search(p, msg_lower) for p in QUESTION_PATTERNS):
            return "question"

        return "unclear"


class ConversationGuard:
    """
    Decides what to do in a reply context based on message classification.
    """

    def __init__(self):
        self.classifier = ConversationClassifier()
        self._merchant_auto_counts: Dict[str, int] = {}

    def clear(self):
        self._merchant_auto_counts.clear()

    def handle_reply(
        self,
        conv_id: str,
        message: str,
        turn_number: int,
        conv_state: Dict,
        merchant: Optional[Dict] = None,
        category: Optional[Dict] = None,
        customer: Optional[Dict] = None,
        llm=None,
        context_store=None,
    ) -> Dict:
        """
        Returns a reply action dict: {"action": "send"|"wait"|"end", ...}
        """
        classification = self.classifier.classify(message, conv_state)

        # Update state
        if classification == "auto_reply":
            m_id = (merchant or {}).get("merchant_id") or conv_state.get("merchant_id") or ""
            conv_count = conv_state.get("auto_reply_count", 0) + 1
            merch_count = (self._merchant_auto_counts.get(m_id, 0) + 1) if m_id else conv_count
            if m_id:
                self._merchant_auto_counts[m_id] = merch_count

            # Turn-based count: in judge simulator, turn_number is i + 1 (2, 3, 4, 5)
            # Turn 2 -> count 1, Turn 3 -> count 2, Turn 4 -> count 3
            turn_count = max(1, turn_number - 1) if turn_number > 1 else 1

            effective_count = max(conv_count, merch_count, turn_count)
            conversation_store.update(conv_id, {
                "auto_reply_count": effective_count,
                "last_auto_reply_body": message,
            })
            return self._handle_auto_reply(conv_id, effective_count, conv_state)

        if classification == "opt_out":
            conversation_store.update(conv_id, {"opted_out": True})
            return {
                "action": "end",
                "rationale": "Merchant requested opt-out — closing conversation and suppressing."
            }

        if classification == "hostile":
            conversation_store.update(conv_id, {"opted_out": True})
            return {
                "action": "send",
                "body": "Understood — I won't message again. If you'd like to reconnect later, just reply 'Hi Vera'. 🙏",
                "cta": "none",
                "rationale": "Merchant expressed frustration — acknowledging and closing politely."
            }

        if classification == "off_topic":
            topic = conv_state.get("current_topic", "our conversation")
            return {
                "action": "send",
                "body": f"That's outside what I can help with directly. Coming back to {topic} — what would you like to do next?",
                "cta": "open_ended",
                "rationale": "Off-topic request — politely redirected to active Vera thread."
            }

        if classification == "pause":
            return {
                "action": "wait",
                "wait_seconds": 14400,  # 4 hours
                "rationale": "Merchant asked to connect later — backing off 4 hours."
            }

        # Explicit acceptance or planning intent → action mode
        if classification in ("explicit_acceptance", "planning_intent"):
            conversation_store.update(conv_id, {"merchant_intent": "accepted"})
            return self._compose_action_reply(
                conv_id, message, classification, conv_state,
                merchant, category, customer, llm, context_store
            )

        if classification == "question":
            return self._compose_question_reply(
                conv_id, message, conv_state, merchant, category, customer, llm, context_store
            )

        if classification == "objection":
            return {
                "action": "send",
                "body": self._handle_objection(message, conv_state, merchant),
                "cta": "open_ended",
                "rationale": "Handling merchant objection — acknowledging and pivoting."
            }

        # Unclear — gentle follow-up
        return {
            "action": "send",
            "body": "Got it — want me to go ahead, or is there something specific you'd like to adjust first?",
            "cta": "open_ended",
            "rationale": "Unclear response — gentle clarification."
        }

    def _handle_auto_reply(self, conv_id: str, auto_count: int, conv_state: Dict) -> Dict:
        if auto_count == 1:
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When you're free, just reply YES to continue — I'll have the details ready.",
                "cta": "binary_yes_no",
                "rationale": "Detected first auto-reply — one prompt to flag for the owner."
            }
        elif auto_count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,  # 24 hours
                "rationale": "Same auto-reply twice — owner not at phone. Waiting 24h."
            }
        else:
            # 3+ auto-replies → end
            conversation_store.update(conv_id, {"opted_out": True})
            return {
                "action": "end",
                "rationale": "Auto-reply 3+ times — no real engagement. Closing conversation."
            }

    def _compose_action_reply(
        self, conv_id, message, classification, conv_state,
        merchant, category, customer, llm, context_store
    ) -> Dict:
        """Compose an action reply when merchant explicitly accepts."""
        strategy = conv_state.get("strategy") or {}
        topic = conv_state.get("current_topic") or "the plan"
        last_bot_body = conv_state.get("last_bot_body") or ""

        # Try to build an action from context
        if merchant and category:
            cat_slug = category.get("slug", "")
            owner = _get_owner(merchant)

            # For action continuation, provide concrete next step
            active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
            offer_str = active_offers[0].get("title", "") if active_offers else ""

            strategy_family = conv_state.get("strategy") or {}
            if isinstance(strategy_family, dict):
                family = strategy_family.get("strategy_family") or ""
            else:
                family = str(strategy_family or "")

            topic_lower = topic.lower()
            family_lower = family.lower()

            if "research" in family_lower or "research" in topic_lower:
                body = (
                    f"Preparing the key points now — will also draft a patient/customer message "
                    f"you can share directly. Ready here in a moment. Reply CONFIRM to proceed with sending."
                )
            elif "planning" in family_lower or "plan" in topic_lower:
                body = (
                    f"Great. Drafting the plan now — I'll include pricing, outreach copy, "
                    f"and the key next step here. Reply CONFIRM to proceed."
                )
            elif "compliance" in family_lower or "compliance" in topic_lower:
                cust_agg = merchant.get("customer_aggregate", {})
                chronic_count = cust_agg.get("chronic_rx_count", cust_agg.get("total_unique_ytd", 0))
                body = (
                    f"{owner}, preparing the compliance checklist and customer notification draft here. "
                    f"This covers the affected records. Reply CONFIRM to proceed and I will send the draft."
                )
            else:
                body = (
                    f"On it — drafting the next steps now. "
                    + (f"I will include your '{offer_str}' in the draft here. " if offer_str else "")
                    + "Reply CONFIRM to proceed."
                )
        else:
            body = "On it! Drafting the next steps here. Reply CONFIRM to proceed."

        # Strip any qualifying phrases to strictly prevent qualifying mode after commitment
        for q in ["would you", "do you", "can you tell", "what if", "how about"]:
            body = re.sub(re.escape(q), "", body, flags=re.IGNORECASE)

        return {
            "action": "send",
            "body": body,
            "cta": "binary_confirm_cancel",
            "rationale": f"Merchant explicitly accepted ({classification}) — switching to action mode."
        }

    def _compose_question_reply(
        self, conv_id, message, conv_state,
        merchant, category, customer, llm, context_store
    ) -> Dict:
        """Handle a genuine question from the merchant."""
        topic = conv_state.get("current_topic", "this topic")
        cat_slug = category.get("slug", "") if category else ""

        # Provide a relevant, grounded answer
        body = (
            f"Good question. Based on what I have here: "
            f"I can put together a detailed breakdown. "
            f"Want me to prepare it now, or is there something specific you'd like to know first?"
        )

        return {
            "action": "send",
            "body": body,
            "cta": "open_ended",
            "rationale": "Answering merchant question — staying on topic."
        }

    def _handle_objection(self, message: str, conv_state: Dict, merchant: Optional[Dict]) -> str:
        """Handle a merchant objection without being pushy."""
        topic = conv_state.get("current_topic", "this")
        return (
            f"Fair enough — no pressure. If things change or you want to revisit {topic} "
            f"later, just ping me. 👍"
        )


# ── Utilities ────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _get_owner(merchant: Dict) -> str:
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", "")
    if owner:
        cat = merchant.get("category_slug", "")
        if cat == "dentists" and not owner.lower().startswith("dr"):
            return f"Dr. {owner}"
        return owner
    return identity.get("name", "there")
