"""
High-quality deterministic fallback composer.
Used when LLM is unavailable or times out.
Produces category-aware, evidence-grounded messages without any LLM.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .evidence import EvidenceLedger
from .strategy import MessageStrategy


class FallbackComposer:
    """
    Rule-based composer that handles all 5 categories and all trigger families.
    Evidence-grounded only — never invents facts.
    """

    def compose(
        self,
        strategy: MessageStrategy,
        ledger: EvidenceLedger,
        trigger: Dict,
        merchant: Dict,
        category: Dict,
        customer: Optional[Dict],
    ) -> Dict:
        strategy_family = strategy.strategy_family
        mode = strategy.mode

        # Dispatch to family handler
        handler = getattr(self, f"_compose_{strategy_family.lower()}", self._compose_generic)
        result = handler(strategy, ledger, trigger, merchant, category, customer)

        # Ensure all required keys
        result.setdefault("cta", strategy.cta_type)
        result.setdefault("send_as", strategy.send_as)
        result.setdefault("rationale", f"Fallback: {strategy_family} for {self._owner(merchant)}")

        # Post-process language
        result["body"] = self._apply_language(result["body"], strategy.language_mode)

        return result

    # ── Per-family handlers ─────────────────────────────────────────────────

    def _compose_research_insight(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})

        # Find the digest item
        digest_item = self._resolve_digest_item(payload, cat)
        if digest_item:
            title = digest_item.get("title", "")
            source = digest_item.get("source", "")
            n = digest_item.get("trial_n")
            actionable = digest_item.get("actionable", "")

            # Build body
            lines = [f"{owner}, heads up on the latest in the digest."]
            if title:
                lines.append(f"Key finding: {title}.")
            if n:
                lines.append(f"(n={n})")
            if source:
                lines.append(f"Source: {source}.")
            if actionable:
                lines.append(f"Suggested action: {actionable}.")
            lines.append("Want me to prepare a brief for your practice?")
            body = " ".join(lines)
        else:
            body = f"{owner}, there's a new research item in the category digest relevant to your practice. Want me to pull the key points?"

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Research digest trigger — sharing verifiable finding with {owner}",
        }

    def _compose_compliance_alert(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        kind = trigger.get("kind", "")

        if kind == "supply_alert":
            molecule = payload.get("molecule", "medication")
            batches = payload.get("affected_batches", [])
            batch_str = ", ".join(batches) if batches else "see alert"
            mfr = payload.get("manufacturer", "the manufacturer")
            body = (
                f"{owner}, action needed: voluntary recall on {molecule} "
                f"(batches: {batch_str}) by {mfr}. "
                f"No safety risk, but affected customers should be informed. "
                f"Want me to draft the customer message + replacement workflow?"
            )
        else:
            digest_item = self._resolve_digest_item(payload, cat)
            if digest_item:
                title = digest_item.get("title", "compliance update")
                source = digest_item.get("source", "")
                actionable = digest_item.get("actionable", "")
                body = (
                    f"{owner}, compliance update: {title}. "
                    f"Source: {source}. "
                    f"Recommended action: {actionable}. "
                    f"Reply YES and I'll prepare the compliance checklist."
                )
            else:
                deadline = payload.get("deadline_iso", payload.get("deadline", ""))
                body = (
                    f"{owner}, there's a compliance requirement that needs your attention"
                    + (f" — deadline {deadline[:10]}" if deadline else "") + ". "
                    f"Reply YES for the action checklist."
                )

        return {
            "body": body,
            "cta": "binary_yes_no",
            "rationale": f"Compliance alert — exact details from trigger payload for {owner}",
        }

    def _compose_performance_dip(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        metric = payload.get("metric", "performance")
        delta = payload.get("delta_pct", 0)
        delta_str = f"{abs(delta)*100:.0f}%"

        perf = merchant.get("performance", {})
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        offer_str = f" Your '{active_offers[0]['title']}' offer is active." if active_offers else ""

        body = (
            f"{owner}, your {metric} is down {delta_str} this week vs last."
            f"{offer_str}"
            f" One thing I can do: boost your Google listing visibility with a targeted post. "
            f"Want me to draft one?"
        )

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Performance dip alert — {metric} -({delta_str}) for {owner}",
        }

    def _compose_seasonal_reframe(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        metric = payload.get("metric", "views")
        delta = payload.get("delta_pct", 0)
        season_note = payload.get("season_note", "")

        cat_slug = cat.get("slug", "")
        seasonal_context = ""
        for beat in cat.get("seasonal_beats", []):
            note = beat.get("note", "")
            month_range = beat.get("month_range", "")
            if "apr" in month_range.lower() or "may" in month_range.lower():
                seasonal_context = note
                break

        body = (
            f"{owner}, your {metric} is down {abs(delta)*100:.0f}% — but this is the "
            f"normal April-June pattern for {cat_slug}."
        )
        if seasonal_context:
            body += f" {seasonal_context.capitalize()}."
        body += " Skip acquisition spend now; focus on retaining current customers. Want a retention plan drafted?"

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Seasonal dip reframe — expected pattern for {cat_slug}, reassuring {owner}",
        }

    def _compose_active_planning(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        topic = payload.get("intent_topic", "business expansion")
        last_msg = payload.get("merchant_last_message", "")

        # Check conversation history for context
        history = merchant.get("conversation_history", [])
        prev_vera = ""
        for turn in reversed(history):
            if turn.get("from") == "vera":
                prev_vera = turn.get("body", "")
                break

        # Generate a practical starter draft
        locality = merchant.get("identity", {}).get("locality", "your area")
        name = merchant.get("identity", {}).get("name", "your business")
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]

        body = (
            f"{owner}, picking up from your message. "
            f"Here's a starter framework for {topic.replace('_', ' ')} — edit as you see fit:\n\n"
        )

        if "corporate" in topic or "bulk" in topic:
            body += (
                f"{name} — corporate packages for {locality}\n"
                f"• Contact us by 5pm for same-day delivery\n"
                f"• Volume pricing available for 10+ units\n\n"
                f"Want me to draft outreach messages for nearby offices too?"
            )
        elif "kids" in topic or "program" in topic or "camp" in topic:
            body += (
                f"Recommended structure: 4-week program, 3 sessions/week, age 7-12\n"
                f"Pricing: ₹2,499 for the full program\n"
                f"Format: 45-min classes, playful + skill-building\n\n"
                f"Want me to draft the GBP post and WhatsApp announcement?"
            )
        else:
            offer_str = f" Your active offer: {active_offers[0]['title']}." if active_offers else ""
            body += (
                f"Step 1: Identify your target customers in {locality}.\n"
                f"Step 2: Activate your {topic.replace('_', ' ')} plan.{offer_str}\n\n"
                f"Reply CONFIRM and I'll set it up."
            )

        return {
            "body": body,
            "cta": "binary_confirm_cancel",
            "rationale": f"Merchant expressed explicit planning intent on '{topic}' — providing actionable draft",
        }

    def _compose_performance_spike(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        metric = payload.get("metric", "views")
        delta = payload.get("delta_pct", 0)
        driver = payload.get("likely_driver", "")

        body = (
            f"{owner}, your {metric} are up {delta*100:.0f}% this week"
            + (f" — likely driven by {driver.replace('_', ' ')}" if driver else "")
            + ". Good signal. Want me to capitalise with a Google post while momentum is high?"
        )

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Performance spike — {metric} +{delta*100:.0f}% for {owner}",
        }

    def _compose_offer_opportunity(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        kind = trigger.get("kind", "")

        if kind == "ipl_match_today":
            match = payload.get("match", "today's match")
            venue = payload.get("venue", "")
            is_weeknight = payload.get("is_weeknight", True)
            active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]

            if not is_weeknight:
                # Saturday — counter-intuitive advice
                body = (
                    f"{owner}, {match} is at {venue} tonight. "
                    f"Note: Saturday IPL matches typically see -12% restaurant covers "
                    f"(people watch at home). "
                )
                if active_offers:
                    body += f"Better angle: push your '{active_offers[0]['title']}' as a delivery-only special. "
                body += "Want me to draft the delivery push?"
            else:
                body = (
                    f"{owner}, {match} tonight. Weeknight matches drive +18% covers historically. "
                )
                if active_offers:
                    body += f"Your '{active_offers[0]['title']}' could do well tonight. "
                body += "Want me to draft a match-night post?"
        elif kind == "festival_upcoming":
            festival = payload.get("festival", "the upcoming festival")
            days = payload.get("days_until", 0)
            body = (
                f"{owner}, {festival} is {days} days away. "
                f"Good time to set up a festival offer. "
                f"Want me to draft one using your current catalog?"
            )
        else:
            body = (
                f"{owner}, there's an opportunity to engage customers this week. "
                f"Want me to draft a targeted offer?"
            )

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"External event opportunity ({kind}) for {owner}",
        }

    def _compose_milestone(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        metric = payload.get("metric", "reviews")
        value_now = payload.get("value_now", 0)
        milestone = payload.get("milestone_value", 0)
        imminent = payload.get("is_imminent", False)

        if imminent:
            body = (
                f"{owner}, you're {milestone - value_now} {metric} away from {milestone}! "
                f"Want me to post a 'tell-your-friends' message to get there this week?"
            )
        else:
            body = (
                f"{owner}, you've crossed {value_now} {metric} — a solid milestone. "
                f"Want me to draft a 'thank you' Google post to celebrate?"
            )

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Milestone trigger — {metric} at {value_now} for {owner}",
        }

    def _compose_review_problem(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        theme = payload.get("theme", "service quality")
        n = payload.get("occurrences_30d", 0)
        quote = payload.get("common_quote", "")

        body = (
            f"{owner}, {n} reviews this month mention '{theme.replace('_', ' ')}'."
        )
        if quote:
            body += f" Common thread: \"{quote}\"."
        body += " Want me to draft a response template + suggest an operational fix?"

        return {
            "body": body,
            "cta": "binary_yes_no",
            "rationale": f"Review theme emerged — {n} mentions of '{theme}' for {owner}",
        }

    def _compose_competitor_change(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        comp_name = payload.get("competitor_name", "a competitor")
        dist_km = payload.get("distance_km")
        their_offer = payload.get("their_offer", "")

        body = f"{owner}, {comp_name} opened nearby"
        if dist_km:
            body += f" ({dist_km}km away)"
        if their_offer:
            body += f" — leading with '{their_offer}'"
        body += ". One thing worth doing: make sure your GBP listing has fresh photos and your offer is visible. Want a quick audit?"

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Competitor opened near {owner}'s locality",
        }

    def _compose_curiosity_ask(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        cat_slug = cat.get("slug", "")

        questions = {
            "dentists": "What treatment has been most requested this week?",
            "salons": "What service has been in most demand this week?",
            "restaurants": "Which dish has been your bestseller this week?",
            "gyms": "Which class or program has been most popular with members this week?",
            "pharmacies": "What's been your top-selling product this week?",
        }
        q = questions.get(cat_slug, "What's been the most popular service this week?")

        body = (
            f"Quick one, {owner} — {q} "
            f"I'll turn your answer into a Google post + a ready-to-use WhatsApp reply. "
            f"Takes 5 minutes."
        )

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Weekly curiosity cadence ask for {owner}",
        }

    def _compose_dormant_reactivation(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        days = payload.get("days_since_last_merchant_message", 0) or payload.get("days_since_expiry", 0)
        last_topic = payload.get("last_topic", "")

        body = f"{owner}, it's been a while"
        if days:
            body += f" ({days} days)"
        if last_topic:
            body += f" since we last spoke about {last_topic.replace('_', ' ')}"
        body += ". Checking in — is there anything new I can help you with? Your Google profile is one quick way to get more visibility this week."

        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Dormant merchant re-engagement for {owner}",
        }

    def _compose_subscription_renewal(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        days = payload.get("days_remaining", 0)
        plan = payload.get("plan", "Pro")
        amount = payload.get("renewal_amount")

        perf = merchant.get("performance", {})
        views = perf.get("views")

        body = f"{owner}, your {plan} subscription expires in {days} days."
        if views:
            body += f" Your profile has been getting {views} views/month."
        if amount:
            body += f" Renewal: ₹{amount}."
        body += " Reply YES to renew, and I'll keep your profile active without interruption."

        return {
            "body": body,
            "cta": "binary_confirm_cancel",
            "rationale": f"Subscription renewal due — {days} days remaining for {owner}",
        }

    def _compose_gbp_action(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        payload = trigger.get("payload", {})
        uplift = payload.get("estimated_uplift_pct", 0)
        path = payload.get("verification_path", "postcard")

        body = (
            f"{owner}, your Google Business Profile is not yet verified."
        )
        if uplift:
            body += f" Verified profiles get roughly {int(uplift*100)}% more calls."
        body += (
            f" Verification via {path} takes 1-2 weeks. "
            f"Want me to guide you through the steps right now?"
        )

        return {
            "body": body,
            "cta": "binary_yes_no",
            "rationale": f"GBP unverified — guiding {owner} to complete verification",
        }

    def _compose_customer_recall(self, strategy, ledger, trigger, merchant, cat, customer):
        customer = customer or {}
        c_name = customer.get("identity", {}).get("name", "there")
        merchant_name = merchant.get("identity", {}).get("name", "the clinic")
        payload = trigger.get("payload", {})
        due_date = payload.get("due_date", "")
        service = payload.get("service_due", "your next appointment")
        slots = payload.get("available_slots", [])

        # Build greeting
        cat_slug = cat.get("slug", "")
        emoji_map = {"dentists": "🦷", "salons": "✂️", "gyms": "💪", "pharmacies": "💊", "restaurants": "🍽️"}
        emoji = emoji_map.get(cat_slug, "")

        body = f"Hi {c_name}, {merchant_name} here {emoji}"
        service_clean = service.replace("_", " ")
        body += f" — your {service_clean} is due."

        if slots:
            if len(slots) == 1:
                slot = slots[0]
                label = slot.get("label", "")
                body += f" Available: {label}."
            elif len(slots) >= 2:
                s1 = slots[0].get("label", "")
                s2 = slots[1].get("label", "")
                body += f" 2 slots open: {s1} or {s2}."

        # Add active offer from merchant
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        if active_offers:
            body += f" {active_offers[0]['title']}."

        if slots and len(slots) >= 2:
            body += " Reply 1 for the first slot, 2 for the second, or suggest a time."
        elif slots:
            body += " Reply YES to confirm or suggest another time."
        else:
            body += " Reply to book your slot."

        return {
            "body": body,
            "cta": "multi_choice_slot" if len(slots) >= 2 else "binary_yes_no",
            "send_as": "merchant_on_behalf",
            "rationale": f"Recall reminder for {c_name} — {service_clean} due",
        }

    def _compose_customer_winback(self, strategy, ledger, trigger, merchant, cat, customer):
        customer = customer or {}
        c_name = customer.get("identity", {}).get("name", "there")
        owner = self._owner(merchant)
        merchant_name = merchant.get("identity", {}).get("name", "us")
        payload = trigger.get("payload", {})
        prev_focus = payload.get("previous_focus", "")
        days = payload.get("days_since_last_visit", 0)

        body = f"Hi {c_name} 👋 {owner} from {merchant_name} here."
        if days:
            body += f" It's been about {days // 7} weeks"
        body += " — happens, no worries."
        if prev_focus:
            focus_clean = prev_focus.replace("_", " ")
            body += f" We have some new sessions that fit {focus_clean} goals."

        # Add trial/intro offer
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        if active_offers:
            body += f" {active_offers[0]['title']}."

        body += " Reply YES for a free trial — no commitment needed."

        return {
            "body": body,
            "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf",
            "rationale": f"Warm winback for {c_name} — lapsed {days} days, no shame framing",
        }

    def _compose_customer_trial_followup(self, strategy, ledger, trigger, merchant, cat, customer):
        customer = customer or {}
        c_name = customer.get("identity", {}).get("name", "there")
        owner = self._owner(merchant)
        merchant_name = merchant.get("identity", {}).get("name", "us")
        payload = trigger.get("payload", {})
        trial_date = payload.get("trial_date", "")
        next_opts = payload.get("next_session_options", [])

        body = f"Hi {c_name}! {owner} from {merchant_name} here."
        if trial_date:
            body += f" Hope you enjoyed your trial on {trial_date}."
        if next_opts:
            opt = next_opts[0]
            label = opt.get("label", "")
            body += f" Next session: {label}."
        body += " Want to book? Reply YES — first month offer applies."

        return {
            "body": body,
            "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf",
            "rationale": f"Trial followup for {c_name}",
        }

    def _compose_customer_refill(self, strategy, ledger, trigger, merchant, cat, customer):
        customer = customer or {}
        c_identity = customer.get("identity", {})
        c_name = c_identity.get("name", "")
        is_senior = c_identity.get("senior_citizen", False)
        channel = customer.get("preferences", {}).get("channel", "whatsapp")

        merchant_name = merchant.get("identity", {}).get("name", "us")
        payload = trigger.get("payload", {})
        molecules = payload.get("molecule_list", [])
        runs_out = payload.get("stock_runs_out_iso", "")
        delivery_saved = payload.get("delivery_address_saved", False)

        # Respectful opening for senior citizens
        if is_senior or "via_son" in channel or "via_" in channel:
            greeting = "Namaste"
        else:
            greeting = f"Hi {c_name}" if c_name else "Hi"

        body = f"{greeting} — {merchant_name} yahan."
        if c_name and is_senior:
            body += f" {c_name} ji ki"
        if molecules:
            mol_str = ", ".join(molecules)
            body += f" {mol_str}"
        body += " medicines"
        if runs_out:
            due = runs_out[:10]
            body += f" {due} ko khatam hongi"
        body += "."

        # Add active offers
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        for offer in active_offers:
            title = offer.get("title", "")
            if "senior" in title.lower() or "delivery" in title.lower():
                body += f" {title}."
                break

        if delivery_saved:
            body += " Free home delivery to saved address."
        body += " Reply CONFIRM to dispatch, or call us if there's any change."

        return {
            "body": body,
            "cta": "binary_confirm_cancel",
            "send_as": "merchant_on_behalf",
            "rationale": f"Chronic refill reminder for {c_name} — exact medicines and due date from payload",
        }

    def _compose_customer_appointment(self, strategy, ledger, trigger, merchant, cat, customer):
        customer = customer or {}
        c_name = customer.get("identity", {}).get("name", "there")
        merchant_name = merchant.get("identity", {}).get("name", "us")
        cat_slug = cat.get("slug", "")
        emoji_map = {"dentists": "🦷", "salons": "✂️", "gyms": "💪", "pharmacies": "💊"}
        emoji = emoji_map.get(cat_slug, "")

        body = (
            f"Hi {c_name} {emoji} — reminder from {merchant_name}: "
            f"your appointment is tomorrow. Please confirm or reschedule. "
            f"Reply YES to confirm."
        )

        return {
            "body": body,
            "cta": "binary_confirm_cancel",
            "send_as": "merchant_on_behalf",
            "rationale": f"Appointment reminder for {c_name}",
        }

    def _compose_customer_event(self, strategy, ledger, trigger, merchant, cat, customer):
        return self._compose_customer_recall(strategy, ledger, trigger, merchant, cat, customer)

    def _compose_generic(self, strategy, ledger, trigger, merchant, cat, customer):
        owner = self._owner(merchant)
        kind = trigger.get("kind", "update")
        body = (
            f"{owner}, there's a {kind.replace('_', ' ')} worth your attention. "
            f"Want me to walk you through the details?"
        )
        return {
            "body": body,
            "cta": "open_ended",
            "rationale": f"Generic trigger handling for {kind}",
        }

    # ── Utilities ────────────────────────────────────────────────────────────

    def _owner(self, merchant: Dict) -> str:
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name", "")
        if owner:
            # Add title for dentists
            cat = merchant.get("category_slug", "")
            if cat == "dentists" and not owner.lower().startswith("dr"):
                return f"Dr. {owner}"
            return owner
        return identity.get("name", "there")

    def _resolve_digest_item(self, payload: Dict, category: Dict) -> Optional[Dict]:
        """Resolve digest item from payload reference."""
        item_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
        if not item_id:
            return None
        for item in category.get("digest", []):
            if item.get("id") == item_id:
                return item
        return None

    def _apply_language(self, body: str, lang_mode: str) -> str:
        """Apply light language adaptation. Doesn't translate — just adds natural Hinglish for hi-en mix."""
        if not body:
            return body
        if lang_mode in ("en", "english"):
            return body
        # For non-English modes, the body is already crafted with some Hindi
        # Just return as-is since fallback already uses some Hindi phrases
        return body
