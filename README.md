# Vera Signal Engine — magicpin AI Challenge

## What I Built

A production-quality merchant engagement bot that implements the **Signal→Proof→Action** pipeline to replicate and improve upon magicpin's Vera assistant.

## Architecture

```
Trigger → SignalRanker → Evidence Ledger → MessageStrategy → Composer → Validator
```

| Module | Purpose |
|---|---|
| `app/ranker.py` | Scores and prioritizes triggers by urgency × merchant engagement state |
| `app/evidence.py` | Evidence Ledger — grounds every claim to context data, prevents hallucination |
| `app/strategy.py` | Selects the right MessageStrategy (RESEARCH_DIGEST, PERF_DIP, RECALL, etc.) |
| `app/composer.py` | Composes WhatsApp copy using LLM + deterministic fallback |
| `app/conversation.py` | Multi-turn guard: auto-reply detection, intent transitions, graceful exit |
| `app/state.py` | Ephemeral context store with versioned idempotent pushes |

## Key Improvements Over Production Vera

1. **Auto-reply detection in 1 turn** — Ends the conversation on first canned auto-reply match, not after 2-3 wasted turns
2. **Intent-handoff**: Detects "let's do it / go ahead / yes" and immediately switches to action mode (no re-qualifying)
3. **Category-correct voice**: Dentist messages use clinical peer tone; salons are warm/practical; restaurants are operator-to-operator
4. **Specificity enforcement**: Every message anchors on a verifiable number from the context (CTR %, views, peer benchmark, offer price)
5. **Anti-hallucination**: Only cites data present in the provided context — no fabricated stats or offers
6. **Hindi-English code-mix**: Detects merchant language preference and matches it per-message

## LLM Setup

The bot runs in **deterministic fallback mode** when no API key is provided, producing rule-based but fully-grounded messages. To enable LLM:

```bash
# Create .env in project root:
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

Free keys available at: https://console.groq.com

## Running Locally

```bash
pip install -r requirements.txt
python bot.py          # starts server on port 8080
python judge_simulator.py  # runs all validation tests
```

## What Additional Context Would Help Most

- **Real merchant reply history** — knowing what language/tone merchants actually respond to (not just preference tags) would significantly improve message personalization
- **Offer conversion rates** — knowing which offer types historically drive replies (e.g., free consultation vs. fixed price) would sharpen the strategy selection
- **Time-of-day patterns** — when merchants are most likely to engage would help cadence planning

## Team
- **Name**: Arpit
- **Approach**: Rule-based Signal ranker + Evidence Ledger + LLM composer with deterministic fallback
