"""
Vera Signal Engine — Main FastAPI Application
magicpin AI Challenge submission.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import config
from .conversation import ConversationGuard
from .composer import compose_message
from .evidence import build_ledger
from .llm import create_provider_from_config
from .ranker import SignalRanker, _stable_conv_id
from .state import context_store, conversation_store, suppression_store
from .strategy import build_strategy

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vera")

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vera Signal Engine",
    description="magicpin AI Challenge — merchant engagement bot",
    version=config.VERSION,
)

START_TIME = time.time()

# Singletons
_ranker = SignalRanker()
_guard = ConversationGuard()
_llm = create_provider_from_config(config)

logger.info(f"LLM provider: {_llm.name}")


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str = ""


class TickRequest(BaseModel):
    now: str
    available_triggers: List[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: str = ""
    turn_number: int = 1


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = context_store.counts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": {
            "category": counts.get("category", 0),
            "merchant": counts.get("merchant", 0),
            "customer": counts.get("customer", 0),
            "trigger": counts.get("trigger", 0),
        },
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": config.TEAM_NAME,
        "team_members": config.TEAM_MEMBERS.split(","),
        "model": _llm.name,
        "approach": (
            "Signal->Proof->Action engine: deterministic SignalRanker + Evidence Ledger "
            "grounding + MessageStrategy layer + LLM copy composer + Evidence validator + "
            "deterministic fallback. No hardcoded examples."
        ),
        "contact_email": config.CONTACT_EMAIL,
        "version": config.VERSION,
        "submitted_at": "2026-09-27T00:00:00Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextRequest):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope",
                     "details": f"scope must be one of {valid_scopes}"}
        )

    accepted, reason, cur_version = context_store.push(
        body.scope, body.context_id, body.version, body.payload
    )

    if not accepted:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": reason, "current_version": cur_version}
        )

    logger.info(f"Context stored: scope={body.scope} id={body.context_id} v={body.version}")
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/v1/tick")
async def tick(body: TickRequest):
    """
    Main proactive send endpoint.
    1. Rank available triggers
    2. Build evidence → strategy → compose
    3. Return actions (max 20)
    """
    start = time.time()
    actions = []
    seen_conv_ids = set()
    merchant_action_counts: Dict[str, int] = {}  # cap at 2 distinct conversations per merchant per tick

    try:
        ranked = _ranker.rank(body.available_triggers, context_store, body.now)

        for ranked_trigger in ranked:
            if len(actions) >= config.MAX_ACTIONS_PER_TICK:
                break
            if time.time() - start > 25:  # internal timeout
                logger.warning("Tick timeout — returning partial results")
                break

            if not ranked_trigger.should_send:
                continue

            trg = ranked_trigger.trigger
            merchant = ranked_trigger.merchant
            category = ranked_trigger.category
            customer = ranked_trigger.customer
            strategy_family = ranked_trigger.strategy
            mode = ranked_trigger.mode
            merchant_id = merchant.get("merchant_id", "")

            # Cap at 2 distinct conversations per merchant per tick to avoid spamming
            if merchant_action_counts.get(merchant_id, 0) >= 2:
                continue

            # Generate stable conversation ID
            conv_id = _stable_conv_id(
                merchant_id, ranked_trigger.trigger_id,
                strategy_family, trg.get("scope", "merchant")
            )

            # Add customer to conv_id if customer-scoped
            if customer:
                cid = customer.get("customer_id", "")
                conv_id = _stable_conv_id(
                    merchant_id + "_" + cid[:10],
                    ranked_trigger.trigger_id,
                    strategy_family, "customer"
                )

            # Exactly one action per (merchant_id, conversation_id) pair per tick (§FAQ)
            if conv_id in seen_conv_ids:
                continue

            # Check existing conversation (opt-out, sent body, etc.)
            conv_state = conversation_store.get(conv_id) or {}
            if conv_state.get("opted_out"):
                continue

            # Build evidence ledger
            ledger = build_ledger(category, merchant, trg, customer, body.now)

            # Build strategy
            strategy = build_strategy(
                strategy_family, mode, merchant, category, trg, customer,
                ranked_trigger.evidence_ids
            )

            # Get conversation history for composer
            hist_turns = conv_state.get("turns", [])

            # Compose message
            try:
                result = compose_message(
                    strategy, ledger, trg, merchant, category, customer,
                    hist_turns, _llm, body.now
                )
            except Exception as e:
                logger.error(f"Compose error for {ranked_trigger.trigger_id}: {e}")
                continue

            body_text = result.get("body", "").strip()
            if not body_text:
                continue

            # Anti-repetition: check against sent bodies
            sent_bodies = conversation_store.get_sent_bodies(conv_id)
            if body_text in sent_bodies:
                logger.warning(f"Duplicate body detected for {conv_id} — skipping")
                continue

            # Determine first-touch template
            is_new_conv = not conv_state.get("turns")
            template_name = strategy.template_name if is_new_conv else None
            template_params = _build_template_params(result, merchant, customer) if is_new_conv else None

            # Build action
            action = {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer.get("customer_id") if customer else None,
                "send_as": result.get("send_as", strategy.send_as),
                "trigger_id": ranked_trigger.trigger_id,
                "body": body_text,
                "cta": result.get("cta", strategy.cta_type),
                "suppression_key": trg.get("suppression_key", ""),
                "rationale": result.get("rationale", ""),
            }
            if template_name:
                action["template_name"] = template_name
            if template_params:
                action["template_params"] = template_params

            actions.append(action)
            seen_conv_ids.add(conv_id)
            merchant_action_counts[merchant_id] = merchant_action_counts.get(merchant_id, 0) + 1

            # Update state
            conversation_store.get_or_create(conv_id)
            conversation_store.update(conv_id, {
                "merchant_id": merchant_id,
                "customer_id": customer.get("customer_id") if customer else None,
                "trigger_id": ranked_trigger.trigger_id,
                "strategy": ranked_trigger.to_plan(),
                "category_slug": category.get("slug", ""),
                "current_topic": strategy_family.lower(),
            })
            conversation_store.add_turn(conv_id, "vera", body_text)
            conversation_store.add_sent_body(conv_id, body_text)

            # Register suppression
            sup_key = trg.get("suppression_key", "")
            if sup_key:
                suppression_store.suppress(sup_key, conv_id)

            logger.info(
                f"Action created: conv={conv_id} strategy={strategy_family} "
                f"priority={ranked_trigger.priority:.2f} llm={_llm.name}"
            )

    except Exception as e:
        logger.error(f"Tick error: {e}", exc_info=True)
        # Return empty actions rather than 500
        return {"actions": []}

    elapsed = time.time() - start
    logger.info(f"Tick done: {len(actions)} actions in {elapsed:.2f}s")
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    """
    Handle merchant/customer reply.
    Classify → update state → decide send/wait/end.
    """
    conv_id = body.conversation_id
    merchant_id = body.merchant_id
    customer_id = body.customer_id
    message = body.message

    # Get existing conversation state
    conv_state = conversation_store.get_or_create(conv_id)

    # Use merchant/customer from stored state if not provided
    if not merchant_id:
        merchant_id = conv_state.get("merchant_id", "")

    # Check opted-out
    if conv_state.get("opted_out"):
        return {
            "action": "end",
            "rationale": "Conversation already ended (opted-out or hostile)."
        }

    # Fetch contexts
    merchant = context_store.get("merchant", merchant_id) if merchant_id else None
    cat_slug = merchant.get("category_slug", "") if merchant else ""
    category = context_store.get("category", cat_slug) if cat_slug else None
    customer = context_store.get("customer", customer_id) if customer_id else None

    # Log incoming turn
    conversation_store.add_turn(conv_id, body.from_role, message)

    # Guard handles everything
    response = _guard.handle_reply(
        conv_id=conv_id,
        message=message,
        turn_number=body.turn_number,
        conv_state=conv_state,
        merchant=merchant,
        category=category,
        customer=customer,
        llm=_llm,
        context_store=context_store,
    )

    # If sending, update conversation state
    if response.get("action") == "send":
        reply_body = response.get("body", "")
        if reply_body:
            # Anti-repetition
            sent_bodies = conversation_store.get_sent_bodies(conv_id)
            if reply_body in sent_bodies:
                # Slightly vary the message
                reply_body = reply_body + " Let me know how you'd like to proceed."
                response["body"] = reply_body

            conversation_store.add_turn(conv_id, "vera", reply_body)
            conversation_store.add_sent_body(conv_id, reply_body)

    elif response.get("action") == "end":
        conversation_store.update(conv_id, {"opted_out": True})

    logger.info(
        f"Reply: conv={conv_id} turn={body.turn_number} "
        f"msg_preview={message[:50]!r} action={response.get('action')}"
    )
    return response


@app.post("/v1/teardown")
async def teardown():
    """Wipe all runtime state (called by judge at end of test)."""
    context_store.clear()
    conversation_store.clear()
    suppression_store.clear()
    _guard.clear()
    logger.info("State wiped via /v1/teardown")
    return {"status": "ok", "message": "State cleared"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_template_params(result: Dict, merchant: Dict, customer: Optional[Dict]) -> List[str]:
    """Build template_params from composed message for first-touch WA template."""
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", identity.get("name", ""))
    body = result.get("body", "")

    # Truncate body into 3 segments for template params
    words = body.split()
    if len(words) <= 10:
        return [owner, body, ""]
    mid = len(words) // 2
    part1 = " ".join(words[:mid])
    part2 = " ".join(words[mid:])
    return [owner, part1, part2]
