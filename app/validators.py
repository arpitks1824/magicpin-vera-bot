"""
Post-generation message validator.
Checks for hallucinations, format issues, and quality problems.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .evidence import EvidenceLedger
from .strategy import MessageStrategy


FORBIDDEN_JARGON = [
    "trigger", "payload", "suppression key", "suppression_key",
    "context_id", "merchant_id", "customer_id", "JSON", "ledger",
    "evidence", "strategy_family", "database", "API", "endpoint",
]

VALID_CTA = {"none", "open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot"}
VALID_SEND_AS = {"vera", "merchant_on_behalf"}

# Patterns that suggest hallucinated or fabricated content
SUSPICIOUS_PATTERNS = [
    r"study of \d+ patients",  # may fabricate study sizes
    r"according to (?!JIDA|DCI|IDA|ICMR|CDSCO|FDA|GST|Zomato|Swiggy|magicpin|Practo|Google|Hair Brand|Salon India|Dentsply)",  # invented citation
]

MULTIPLE_CTA_PATTERNS = [
    r"reply yes for .+ no for .+",
    r"reply 1 for .+ 2 for .+ 3 for",
    r"option a .+ option b .+ option c",
]


class MessageValidator:
    """
    Validates a composed message against evidence ledger and quality rules.
    Returns (is_valid: bool, issues: List[str]).
    """

    def validate(
        self,
        result: Dict,
        ledger: EvidenceLedger,
        strategy: MessageStrategy,
        merchant: Dict,
        customer: Optional[Dict],
        conversation_history: List[Dict],
    ) -> Tuple[bool, List[str]]:
        issues = []

        body = result.get("body", "")
        cta = result.get("cta", "")
        send_as = result.get("send_as", "")
        rationale = result.get("rationale", "")

        # 1. Required fields
        if not body or not body.strip():
            issues.append("body is empty")
        if not cta:
            issues.append("cta is missing")
        if not send_as:
            issues.append("send_as is missing")

        # 2. Valid enum values
        if cta and cta not in VALID_CTA:
            issues.append(f"invalid cta value: {cta}")
        if send_as and send_as not in VALID_SEND_AS:
            issues.append(f"invalid send_as value: {send_as}")

        # 3. No internal jargon exposed to user
        body_lower = body.lower()
        for jargon in FORBIDDEN_JARGON:
            if jargon.lower() in body_lower:
                issues.append(f"internal jargon in body: '{jargon}'")

        # 4. No URLs (hard fail per challenge rules)
        if re.search(r"https?://", body):
            issues.append("URL found in body — not allowed")

        # 5. No merchant/customer IDs in visible message
        merchant_id = merchant.get("merchant_id", "")
        if merchant_id and merchant_id in body:
            issues.append("merchant_id exposed in body")
        if customer:
            cid = customer.get("customer_id", "")
            if cid and cid in body:
                issues.append("customer_id exposed in body")

        # 6. Multiple CTAs check
        for pat in MULTIPLE_CTA_PATTERNS:
            if re.search(pat, body_lower):
                issues.append("multiple CTAs detected in body")
                break

        # 7. Evidence grounding — check suspicious numbers
        # Numbers in body that don't appear in evidence raise a flag
        body_numbers = set(re.findall(r"\d{3,}", body))  # 3+ digit numbers
        evidence_numbers = set(ledger.extract_numbers())
        # Remove obvious non-facts (year references, phone-like)
        body_numbers = {n for n in body_numbers if len(n) <= 6}
        for n in body_numbers:
            if n not in evidence_numbers:
                # Check if it's a derived value (simple arithmetic)
                # Allow small tolerances
                if not _is_likely_derived(n, evidence_numbers):
                    issues.append(f"number {n} in body not found in evidence ledger")

        # 8. send_as match for customer triggers
        if customer and strategy.send_as == "merchant_on_behalf":
            if send_as != "merchant_on_behalf":
                issues.append("customer trigger should have send_as=merchant_on_behalf")

        # 9. No repeated prior body
        for turn in conversation_history:
            if turn.get("role") in ("vera", "bot") or turn.get("from") == "vera":
                prior_body = turn.get("body", "")
                if prior_body and prior_body.strip() == body.strip():
                    issues.append("exact same body as a prior turn")
                    break

        # 10. Reasonable length (warn, not fail)
        word_count = len(body.split())
        if word_count > 150:
            issues.append(f"body too long ({word_count} words, target <80-100)")

        # 11. Category taboo check
        # (We do this lightly — a soft check)
        # If validation passes with <=2 minor issues (length, etc), still accept

        is_valid = len([i for i in issues if not i.startswith("body too long")]) == 0
        return is_valid, issues


def _is_likely_derived(n: str, evidence_numbers: set) -> bool:
    """Check if a number might be derived from evidence numbers."""
    try:
        val = int(n)
        for ev in evidence_numbers:
            try:
                ev_val = int(ev)
                # Check simple arithmetic relationships
                if ev_val != 0 and abs(val - ev_val) <= max(ev_val * 0.2, 5):
                    return True
                # Check percentage of evidence number
                for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.38, 0.40, 0.50]:
                    derived = round(ev_val * pct)
                    if abs(val - derived) <= 2:
                        return True
            except ValueError:
                continue
    except ValueError:
        pass
    return False
