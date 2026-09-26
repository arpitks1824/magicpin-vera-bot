"""
Evidence Ledger — extracts and validates factual claims from context.
The LLM is ONLY allowed to use facts present in this ledger.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceItem:
    id: str
    value: Any
    display_value: str
    source: str
    sensitivity: str = "normal"  # "normal" | "sensitive" | "derived"
    allowed: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "value": self.value,
            "display_value": self.display_value,
            "source": self.source,
            "sensitivity": self.sensitivity,
            "allowed": self.allowed,
        }


class EvidenceLedger:
    """
    Builds and maintains the set of allowable facts for message composition.
    """

    def __init__(self):
        self.items: List[EvidenceItem] = []
        self._by_id: Dict[str, EvidenceItem] = {}

    def add(self, item: EvidenceItem):
        self.items.append(item)
        self._by_id[item.id] = item

    def get(self, eid: str) -> Optional[EvidenceItem]:
        return self._by_id.get(eid)

    def all_allowed(self) -> List[EvidenceItem]:
        return [i for i in self.items if i.allowed]

    def to_dict(self) -> dict:
        return {"items": [i.to_dict() for i in self.all_allowed()]}

    def extract_numbers(self) -> List[str]:
        """All numeric values in the ledger (for hallucination detection)."""
        nums = []
        for item in self.all_allowed():
            v = str(item.display_value)
            # extract all numeric portions
            found = re.findall(r"\d+(?:[.,]\d+)?", v)
            nums.extend(found)
        return nums

    def all_strings(self) -> List[str]:
        """All string evidence values for hallucination detection."""
        return [str(i.display_value) for i in self.all_allowed()]


def build_ledger(
    category: Dict,
    merchant: Dict,
    trigger: Dict,
    customer: Optional[Dict],
    now_str: str = None,
) -> EvidenceLedger:
    """
    Main factory: extract all allowable evidence from the four context layers.
    """
    ledger = EvidenceLedger()
    now_str = now_str or datetime.now(timezone.utc).isoformat()

    # ── CATEGORY evidence ────────────────────────────────────────────────────
    cat_slug = category.get("slug", "")
    cat_name = category.get("display_name", cat_slug)

    ledger.add(EvidenceItem(
        id="cat.slug", value=cat_slug,
        display_value=cat_slug, source="category.slug"
    ))
    ledger.add(EvidenceItem(
        id="cat.name", value=cat_name,
        display_value=cat_name, source="category.display_name"
    ))

    # Peer stats
    peer = category.get("peer_stats", {})
    if peer.get("avg_ctr"):
        ledger.add(EvidenceItem(
            id="cat.peer_avg_ctr",
            value=peer["avg_ctr"],
            display_value=f"{peer['avg_ctr']*100:.1f}%",
            source="category.peer_stats.avg_ctr"
        ))
    if peer.get("avg_rating"):
        ledger.add(EvidenceItem(
            id="cat.peer_avg_rating",
            value=peer["avg_rating"],
            display_value=str(peer["avg_rating"]),
            source="category.peer_stats.avg_rating"
        ))
    if peer.get("avg_review_count"):
        ledger.add(EvidenceItem(
            id="cat.peer_avg_reviews",
            value=peer["avg_review_count"],
            display_value=str(peer["avg_review_count"]),
            source="category.peer_stats.avg_review_count"
        ))
    if peer.get("avg_views_30d"):
        ledger.add(EvidenceItem(
            id="cat.peer_avg_views_30d",
            value=peer["avg_views_30d"],
            display_value=str(peer["avg_views_30d"]),
            source="category.peer_stats.avg_views_30d"
        ))
    if peer.get("avg_calls_30d"):
        ledger.add(EvidenceItem(
            id="cat.peer_avg_calls_30d",
            value=peer["avg_calls_30d"],
            display_value=str(peer["avg_calls_30d"]),
            source="category.peer_stats.avg_calls_30d"
        ))

    # Digest items (key source for research/compliance facts)
    digest = category.get("digest", [])
    for item in digest:
        did = item.get("id", "")
        title = item.get("title", "")
        source = item.get("source", "")
        summary = item.get("summary", "")
        ledger.add(EvidenceItem(
            id=f"cat.digest.{did}.title",
            value=title, display_value=title,
            source=f"category.digest.{did}"
        ))
        if source:
            ledger.add(EvidenceItem(
                id=f"cat.digest.{did}.source",
                value=source, display_value=source,
                source=f"category.digest.{did}.source"
            ))
        if summary:
            ledger.add(EvidenceItem(
                id=f"cat.digest.{did}.summary",
                value=summary, display_value=summary,
                source=f"category.digest.{did}.summary"
            ))
        # Specific fields
        for fld in ["trial_n", "patient_segment", "actionable", "date", "credits",
                    "deadline", "molecule", "manufacturer"]:
            if item.get(fld):
                ledger.add(EvidenceItem(
                    id=f"cat.digest.{did}.{fld}",
                    value=item[fld], display_value=str(item[fld]),
                    source=f"category.digest.{did}.{fld}"
                ))

    # Offer catalog
    for offer in category.get("offer_catalog", []):
        oid = offer.get("id", "")
        ledger.add(EvidenceItem(
            id=f"cat.offer.{oid}",
            value=offer.get("title", ""),
            display_value=offer.get("title", ""),
            source=f"category.offer_catalog.{oid}"
        ))

    # Seasonal beats
    for sb in category.get("seasonal_beats", []):
        ledger.add(EvidenceItem(
            id=f"cat.seasonal.{sb.get('month_range', '')}",
            value=sb.get("note", ""),
            display_value=sb.get("note", ""),
            source="category.seasonal_beats"
        ))

    # Trend signals
    for ts in category.get("trend_signals", []):
        query = ts.get("query", "")
        delta = ts.get("delta_yoy")
        if delta is not None:
            pct_str = f"+{int(delta*100)}%" if delta >= 0 else f"{int(delta*100)}%"
            ledger.add(EvidenceItem(
                id=f"cat.trend.{query[:30]}",
                value=delta, display_value=f"{query}: {pct_str} YoY",
                source="category.trend_signals"
            ))

    # ── MERCHANT evidence ────────────────────────────────────────────────────
    mid = merchant.get("merchant_id", "")
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    cust_agg = merchant.get("customer_aggregate", {})

    ledger.add(EvidenceItem(
        id="merchant.id", value=mid, display_value=mid, source="merchant.merchant_id"
    ))
    ledger.add(EvidenceItem(
        id="merchant.name", value=identity.get("name", ""),
        display_value=identity.get("name", ""), source="merchant.identity.name"
    ))
    owner = identity.get("owner_first_name", "")
    if owner:
        ledger.add(EvidenceItem(
            id="merchant.owner_first_name", value=owner,
            display_value=owner, source="merchant.identity.owner_first_name"
        ))
    ledger.add(EvidenceItem(
        id="merchant.city", value=identity.get("city", ""),
        display_value=identity.get("city", ""), source="merchant.identity.city"
    ))
    ledger.add(EvidenceItem(
        id="merchant.locality", value=identity.get("locality", ""),
        display_value=identity.get("locality", ""), source="merchant.identity.locality"
    ))
    ledger.add(EvidenceItem(
        id="merchant.verified", value=identity.get("verified", False),
        display_value=str(identity.get("verified", False)),
        source="merchant.identity.verified"
    ))
    ledger.add(EvidenceItem(
        id="merchant.languages", value=identity.get("languages", ["en"]),
        display_value=", ".join(identity.get("languages", ["en"])),
        source="merchant.identity.languages"
    ))

    # Subscription
    ledger.add(EvidenceItem(
        id="merchant.sub_status", value=sub.get("status", ""),
        display_value=sub.get("status", ""), source="merchant.subscription.status"
    ))
    if sub.get("days_remaining") is not None:
        ledger.add(EvidenceItem(
            id="merchant.sub_days_remaining", value=sub["days_remaining"],
            display_value=str(sub["days_remaining"]),
            source="merchant.subscription.days_remaining"
        ))
    if sub.get("days_since_expiry") is not None:
        ledger.add(EvidenceItem(
            id="merchant.sub_days_since_expiry", value=sub["days_since_expiry"],
            display_value=str(sub["days_since_expiry"]),
            source="merchant.subscription.days_since_expiry"
        ))
    if sub.get("plan"):
        ledger.add(EvidenceItem(
            id="merchant.sub_plan", value=sub["plan"],
            display_value=sub["plan"], source="merchant.subscription.plan"
        ))

    # Performance
    if perf:
        for metric in ["views", "calls", "directions", "ctr", "leads"]:
            v = perf.get(metric)
            if v is not None:
                if metric == "ctr":
                    display = f"{v*100:.1f}%"
                else:
                    display = str(v)
                ledger.add(EvidenceItem(
                    id=f"merchant.perf.{metric}", value=v,
                    display_value=display, source=f"merchant.performance.{metric}"
                ))

        delta = perf.get("delta_7d", {})
        for dm in ["views_pct", "calls_pct", "ctr_pct"]:
            v = delta.get(dm)
            if v is not None:
                sign = "+" if v >= 0 else ""
                ledger.add(EvidenceItem(
                    id=f"merchant.perf.delta_7d.{dm}", value=v,
                    display_value=f"{sign}{v*100:.0f}%",
                    source=f"merchant.performance.delta_7d.{dm}",
                    sensitivity="derived"
                ))

        # Derived: CTR vs peer
        merchant_ctr = perf.get("ctr")
        peer_ctr = peer.get("avg_ctr")
        if merchant_ctr and peer_ctr:
            diff = merchant_ctr - peer_ctr
            pct_diff = (merchant_ctr - peer_ctr) / peer_ctr * 100
            sign = "above" if diff >= 0 else "below"
            ledger.add(EvidenceItem(
                id="merchant.perf.ctr_vs_peer",
                value=diff,
                display_value=f"{abs(pct_diff):.0f}% {sign} category average",
                source="derived:merchant.performance.ctr/category.peer_stats.avg_ctr",
                sensitivity="derived"
            ))

    # Offers
    for offer in merchant.get("offers", []):
        oid = offer.get("id", "")
        if offer.get("status") == "active":
            ledger.add(EvidenceItem(
                id=f"merchant.offer.{oid}",
                value=offer.get("title", ""),
                display_value=offer.get("title", ""),
                source=f"merchant.offers.{oid}"
            ))

    # Customer aggregate
    for fld, dv_fmt in [
        ("total_unique_ytd", "{}"),
        ("lapsed_180d_plus", "{} lapsed >180d"),
        ("lapsed_90d_plus", "{} lapsed >90d"),
        ("retention_6mo_pct", "{:.0%} 6mo retention"),
        ("retention_3mo_pct", "{:.0%} 3mo retention"),
        ("chronic_rx_count", "{} chronic-Rx customers"),
        ("total_active_members", "{} active members"),
        ("repeat_customer_pct", "{:.0%} repeat customers"),
        ("high_risk_adult_count", "{} high-risk adult patients"),
        ("delivery_orders_30d", "{} delivery orders/30d"),
        ("dine_in_orders_30d", "{} dine-in orders/30d"),
        ("monthly_churn_pct", "{:.0%} monthly churn"),
        ("trial_to_paid_pct", "{:.0%} trial-to-paid"),
        ("repeat_customer_pct", "{:.0%} repeat customer rate"),
    ]:
        v = cust_agg.get(fld)
        if v is not None:
            try:
                display = dv_fmt.format(v)
            except Exception:
                display = str(v)
            ledger.add(EvidenceItem(
                id=f"merchant.cust_agg.{fld}", value=v,
                display_value=display,
                source=f"merchant.customer_aggregate.{fld}"
            ))

    # Signals
    for sig in merchant.get("signals", []):
        ledger.add(EvidenceItem(
            id=f"merchant.signal.{sig[:40]}",
            value=sig, display_value=sig,
            source="merchant.signals"
        ))

    # Review themes
    for rt in merchant.get("review_themes", []):
        theme = rt.get("theme", "")
        sentiment = rt.get("sentiment", "")
        n = rt.get("occurrences_30d", 0)
        if n > 0:
            ledger.add(EvidenceItem(
                id=f"merchant.review.{theme}",
                value=n,
                display_value=f"{n} {sentiment} reviews mentioning '{theme}'",
                source=f"merchant.review_themes.{theme}"
            ))

    # Conversation history
    history = merchant.get("conversation_history", [])
    for i, turn in enumerate(history[-3:]):  # last 3 turns
        body = turn.get("body", "")
        from_who = turn.get("from", "")
        engagement = turn.get("engagement", "")
        ledger.add(EvidenceItem(
            id=f"merchant.conv_history.{i}",
            value=body, display_value=f"[{from_who}] {body}",
            source=f"merchant.conversation_history.{i}",
            sensitivity="sensitive"
        ))
        if engagement:
            ledger.add(EvidenceItem(
                id=f"merchant.conv_history.{i}.engagement",
                value=engagement, display_value=engagement,
                source=f"merchant.conversation_history.{i}.engagement"
            ))

    # ── TRIGGER evidence ────────────────────────────────────────────────────
    trg_id = trigger.get("id", "")
    trg_kind = trigger.get("kind", "")
    trg_urgency = trigger.get("urgency", 1)
    trg_payload = trigger.get("payload", {})
    trg_expires = trigger.get("expires_at", "")
    trg_suppression = trigger.get("suppression_key", "")

    ledger.add(EvidenceItem(
        id="trigger.id", value=trg_id,
        display_value=trg_id, source="trigger.id"
    ))
    ledger.add(EvidenceItem(
        id="trigger.kind", value=trg_kind,
        display_value=trg_kind, source="trigger.kind"
    ))
    ledger.add(EvidenceItem(
        id="trigger.urgency", value=trg_urgency,
        display_value=str(trg_urgency), source="trigger.urgency"
    ))
    ledger.add(EvidenceItem(
        id="trigger.suppression_key", value=trg_suppression,
        display_value=trg_suppression, source="trigger.suppression_key"
    ))
    if trg_expires:
        ledger.add(EvidenceItem(
            id="trigger.expires_at", value=trg_expires,
            display_value=trg_expires, source="trigger.expires_at"
        ))

    # Payload facts — traverse and add all leaf values
    _add_payload_evidence(ledger, trg_payload, "trigger.payload")

    # Link to category digest item if referenced
    top_item_id = trg_payload.get("top_item_id")
    if top_item_id:
        for d_item in digest:
            if d_item.get("id") == top_item_id:
                for k, v in d_item.items():
                    if k != "id" and v:
                        ledger.add(EvidenceItem(
                            id=f"trigger.digest_item.{k}",
                            value=v, display_value=str(v),
                            source=f"trigger.payload.top_item_id -> category.digest.{top_item_id}.{k}"
                        ))
                break

    # ── CUSTOMER evidence ────────────────────────────────────────────────────
    if customer:
        c_id = customer.get("customer_id", "")
        c_identity = customer.get("identity", {})
        c_rel = customer.get("relationship", {})
        c_prefs = customer.get("preferences", {})
        c_consent = customer.get("consent", {})
        c_state = customer.get("state", "")

        ledger.add(EvidenceItem(
            id="customer.id", value=c_id,
            display_value=c_id, source="customer.customer_id",
            sensitivity="sensitive"
        ))
        ledger.add(EvidenceItem(
            id="customer.name", value=c_identity.get("name", ""),
            display_value=c_identity.get("name", ""),
            source="customer.identity.name"
        ))
        ledger.add(EvidenceItem(
            id="customer.language_pref", value=c_identity.get("language_pref", "en"),
            display_value=c_identity.get("language_pref", "en"),
            source="customer.identity.language_pref"
        ))
        ledger.add(EvidenceItem(
            id="customer.age_band", value=c_identity.get("age_band", ""),
            display_value=c_identity.get("age_band", ""),
            source="customer.identity.age_band"
        ))
        ledger.add(EvidenceItem(
            id="customer.state", value=c_state,
            display_value=c_state, source="customer.state"
        ))
        ledger.add(EvidenceItem(
            id="customer.preferred_slots",
            value=c_prefs.get("preferred_slots", ""),
            display_value=c_prefs.get("preferred_slots", ""),
            source="customer.preferences.preferred_slots"
        ))

        # Consent
        consent_scope = c_consent.get("scope", [])
        ledger.add(EvidenceItem(
            id="customer.consent_scope",
            value=consent_scope,
            display_value=", ".join(consent_scope),
            source="customer.consent.scope"
        ))

        # Relationship
        for fld in ["first_visit", "last_visit", "visits_total", "lifetime_value"]:
            v = c_rel.get(fld)
            if v is not None:
                ledger.add(EvidenceItem(
                    id=f"customer.rel.{fld}", value=v,
                    display_value=str(v),
                    source=f"customer.relationship.{fld}"
                ))

        # Services received
        services = c_rel.get("services_received", [])
        if services:
            ledger.add(EvidenceItem(
                id="customer.rel.services",
                value=services,
                display_value=", ".join(str(s) for s in services),
                source="customer.relationship.services_received"
            ))

        # Senior citizen flag
        is_senior = c_identity.get("senior_citizen", False)
        if is_senior:
            ledger.add(EvidenceItem(
                id="customer.is_senior", value=True,
                display_value="senior citizen",
                source="customer.identity.senior_citizen"
            ))

        # Compute lapsed duration if we have last_visit
        last_visit_str = c_rel.get("last_visit")
        if last_visit_str and c_state in ("lapsed_soft", "lapsed_hard"):
            try:
                lv = datetime.fromisoformat(last_visit_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                days = (now - lv).days
                ledger.add(EvidenceItem(
                    id="customer.days_since_last_visit",
                    value=days,
                    display_value=f"{days} days since last visit",
                    source="derived:customer.relationship.last_visit",
                    sensitivity="derived"
                ))
            except Exception:
                pass

    return ledger


def _add_payload_evidence(ledger: EvidenceLedger, payload: dict, prefix: str):
    """Recursively extract leaf values from trigger payload."""
    if not isinstance(payload, dict):
        return
    for k, v in payload.items():
        eid = f"{prefix}.{k}"
        if isinstance(v, dict):
            _add_payload_evidence(ledger, v, eid)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _add_payload_evidence(ledger, item, f"{eid}.{i}")
                elif item is not None:
                    ledger.add(EvidenceItem(
                        id=f"{eid}.{i}", value=item,
                        display_value=str(item), source=eid
                    ))
        elif v is not None:
            ledger.add(EvidenceItem(
                id=eid, value=v, display_value=str(v), source=eid
            ))
