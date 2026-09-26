"""
Main message composer.
Orchestrates: Evidence Ledger → Strategy → LLM/Fallback → Validator.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from .evidence import EvidenceLedger, build_ledger
from .fallback import FallbackComposer
from .llm import LLMProvider, NullProvider
from .strategy import MessageStrategy, build_strategy
from .validators import MessageValidator

logger = logging.getLogger(__name__)

_fallback = FallbackComposer()
_validator = MessageValidator()


# ── System prompt for LLM composer ─────────────────────────────────────────

SYSTEM_PROMPT = """You are Vera, magicpin's merchant AI assistant. You compose WhatsApp messages.

CRITICAL RULES:
1. ONLY use facts from the EVIDENCE LEDGER provided. Do NOT invent any numbers, names, dates, prices, or statistics.
2. Every factual claim must be traceable to an evidence item.
3. One primary CTA — last sentence contains the action.
4. No URLs unless in the evidence.
5. No internal jargon: never say "trigger", "payload", "context", "suppression key", "JSON", "ledger" to the merchant.
6. No long preambles. No "Hi, hope you're doing well."
7. Match language_mode: hi-en mix = natural Hinglish, not every sentence in Hindi.
8. Merchant-facing: use owner first name or clinic/business name. Customer-facing: use customer first name.
9. For CUSTOMER messages: send_as must be "merchant_on_behalf".
10. Keep body under 80 words unless the content genuinely requires more (e.g., a quote or price table).

RESPOND ONLY WITH THIS JSON:
{
  "body": "<the WhatsApp message>",
  "cta": "<none|open_ended|binary_yes_no|binary_confirm_cancel|multi_choice_slot>",
  "send_as": "<vera|merchant_on_behalf>",
  "rationale": "<1-2 sentences: which signal, which evidence, what outcome expected>",
  "used_evidence": ["<evidence_id_1>", "<evidence_id_2>"]
}"""


def compose_message(
    strategy: MessageStrategy,
    ledger: EvidenceLedger,
    trigger: Dict,
    merchant: Dict,
    category: Dict,
    customer: Optional[Dict],
    conversation_history: List[Dict],
    llm: Optional[LLMProvider],
    now_str: str = "",
) -> Dict:
    """
    Full composition pipeline:
    1. Build user prompt from strategy + evidence
    2. Call LLM (with timeout)
    3. Parse + validate
    4. Fall back to deterministic if LLM fails
    """
    # Try LLM first
    if llm and not isinstance(llm, NullProvider):
        try:
            user_prompt = _build_prompt(
                strategy, ledger, trigger, merchant, category, customer,
                conversation_history, now_str
            )
            raw = llm.complete(user_prompt, SYSTEM_PROMPT)
            result = _parse_llm_output(raw)

            # Validate
            valid, issues = _validator.validate(
                result, ledger, strategy, merchant, customer,
                conversation_history
            )
            if valid:
                logger.debug(f"LLM compose succeeded for {trigger.get('id', '')}")
                return result
            else:
                logger.warning(f"LLM output failed validation: {issues}")
                # Attempt constrained regeneration
                try:
                    constrained_prompt = user_prompt + f"\n\nPrevious attempt had issues: {issues}. Fix them strictly."
                    raw2 = llm.complete(constrained_prompt, SYSTEM_PROMPT)
                    result2 = _parse_llm_output(raw2)
                    valid2, issues2 = _validator.validate(
                        result2, ledger, strategy, merchant, customer,
                        conversation_history
                    )
                    if valid2:
                        return result2
                except Exception:
                    pass
                # Fall through to fallback
        except Exception as e:
            logger.warning(f"LLM error: {e} — using fallback")

    # Deterministic fallback
    result = _fallback.compose(strategy, ledger, trigger, merchant, category, customer)
    return result


def _build_prompt(
    strategy: MessageStrategy,
    ledger: EvidenceLedger,
    trigger: Dict,
    merchant: Dict,
    category: Dict,
    customer: Optional[Dict],
    history: List[Dict],
    now_str: str,
) -> str:
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("name", ""))
    cat_slug = category.get("slug", "")
    cat_voice = category.get("voice", {})

    # Build evidence section (only allowed items)
    evidence_lines = []
    for item in ledger.all_allowed()[:40]:  # cap at 40 items
        evidence_lines.append(
            f"  [{item.id}] = {item.display_value} (source: {item.source})"
        )

    # Conversation history
    hist_lines = []
    for turn in history[-4:]:  # last 4 turns
        role = turn.get("role", turn.get("from", "?"))
        body = turn.get("body", turn.get("msg", ""))[:200]
        hist_lines.append(f"  [{role}]: {body}")

    # Customer summary
    cust_summary = ""
    if customer:
        c_id = customer.get("identity", {})
        c_rel = customer.get("relationship", {})
        cust_summary = f"""
CUSTOMER:
  name: {c_id.get("name", "")}
  language_pref: {c_id.get("language_pref", "en")}
  age_band: {c_id.get("age_band", "")}
  state: {customer.get("state", "")}
  last_visit: {c_rel.get("last_visit", "")}
  visits_total: {c_rel.get("visits_total", "")}
  consent_scope: {customer.get("consent", {}).get("scope", [])}
  preferred_slots: {customer.get("preferences", {}).get("preferred_slots", "")}
"""

    prompt = f"""TASK: Compose a WhatsApp message for the following scenario.

STRATEGY:
  family: {strategy.strategy_family}
  mode: {strategy.mode}
  why_now: {strategy.why_now}
  angle: {strategy.message_angle}
  cta_type: {strategy.cta_type}
  send_as: {strategy.send_as}
  language_mode: {strategy.language_mode}
  hook: {strategy.hook_type}
  tone_notes: {strategy.tone_notes}
  desired_outcome: {strategy.desired_merchant_action}

CATEGORY: {cat_slug}
  tone: {cat_voice.get("tone", "")}
  vocab_allowed: {cat_voice.get("vocab_allowed", [])[:6]}
  vocab_taboo: {cat_voice.get("vocab_taboo", [])}

MERCHANT:
  owner_first_name: {owner}
  business_name: {identity.get("name", "")}
  city: {identity.get("city", "")}
  locality: {identity.get("locality", "")}
  languages: {identity.get("languages", ["en"])}
  category: {cat_slug}
  subscription_status: {merchant.get("subscription", {}).get("status", "")}
  signals: {merchant.get("signals", [])[:5]}
{cust_summary}
TRIGGER:
  kind: {trigger.get("kind", "")}
  urgency: {trigger.get("urgency", 1)}
  suppression_key: {trigger.get("suppression_key", "")}

EVIDENCE LEDGER (ONLY use facts from here):
{chr(10).join(evidence_lines)}

CONVERSATION HISTORY (last turns):
{chr(10).join(hist_lines) if hist_lines else "  (new conversation)"}

NOW: {now_str}

Write the message. Remember: HOOK + VERIFIABLE CONTEXT + WHY THIS MATTERS + ONE NEXT STEP.
"""
    return prompt


def _parse_llm_output(raw: str) -> Dict:
    """Parse LLM JSON output robustly."""
    # Try direct JSON parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Return error dict
    raise ValueError(f"Could not parse LLM output: {raw[:200]}")
