"""
Thread-safe, async-safe in-memory state store for contexts and conversations.
"""
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


class ContextStore:
    """
    Stores context payloads keyed by (scope, context_id).
    Version-aware: same version → no-op, lower → reject (409), higher → replace.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # (scope, context_id) -> {"version": int, "payload": dict, "stored_at": str}
        self._store: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def push(
        self, scope: str, context_id: str, version: int, payload: dict
    ) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Returns (accepted, reason, current_version).
        reason is None when accepted.
        """
        key = (scope, context_id)
        with self._lock:
            existing = self._store.get(key)
            if existing is not None:
                if existing["version"] > version:
                    return False, "stale_version", existing["version"]
                elif existing["version"] == version:
                    # Idempotent re-post: no-op per challenge testing brief §2.1
                    return True, None, version
            self._store[key] = {
                "version": version,
                "payload": payload,
                "stored_at": datetime.utcnow().isoformat() + "Z",
            }
            return True, None, version

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        key = (scope, context_id)
        with self._lock:
            entry = self._store.get(key)
            return entry["payload"] if entry else None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        key = (scope, context_id)
        with self._lock:
            entry = self._store.get(key)
            return entry["version"] if entry else None

    def counts(self) -> Dict[str, int]:
        with self._lock:
            c: Dict[str, int] = {}
            for (scope, _) in self._store:
                c[scope] = c.get(scope, 0) + 1
            return c

    def all_of_scope(self, scope: str) -> Dict[str, dict]:
        with self._lock:
            return {
                cid: entry["payload"]
                for (s, cid), entry in self._store.items()
                if s == scope
            }

    def clear(self):
        with self._lock:
            self._store.clear()


class ConversationStore:
    """
    Stores conversation state per conversation_id.
    State includes: turns, suppression flags, last_body_sent,
    auto_reply_count, opted_out, etc.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # conversation_id -> ConversationState dict
        self._store: Dict[str, Dict[str, Any]] = {}

    def get_or_create(self, conv_id: str) -> Dict[str, Any]:
        with self._lock:
            if conv_id not in self._store:
                self._store[conv_id] = {
                    "conv_id": conv_id,
                    "turns": [],
                    "opted_out": False,
                    "suppressed": False,
                    "auto_reply_count": 0,
                    "last_auto_reply_body": None,
                    "last_bot_body": None,
                    "sent_bodies": [],
                    "merchant_intent": None,  # "qualified", "accepted", "rejected", "planning"
                    "current_topic": None,
                    "turn_count": 0,
                    "merchant_id": None,
                    "customer_id": None,
                    "trigger_id": None,
                    "strategy": None,
                    "category_slug": None,
                    "wait_until": None,
                }
            return self._store[conv_id]

    def update(self, conv_id: str, updates: Dict[str, Any]):
        with self._lock:
            if conv_id in self._store:
                self._store[conv_id].update(updates)

    def add_turn(self, conv_id: str, role: str, body: str, metadata: Dict = None):
        with self._lock:
            state = self.get_or_create(conv_id)
            turn = {
                "role": role,
                "body": body,
                "ts": datetime.utcnow().isoformat() + "Z",
                **(metadata or {}),
            }
            state["turns"].append(turn)
            state["turn_count"] += 1

    def get(self, conv_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._store.get(conv_id)

    def is_opted_out(self, conv_id: str) -> bool:
        with self._lock:
            state = self._store.get(conv_id)
            return state.get("opted_out", False) if state else False

    def get_sent_bodies(self, conv_id: str) -> list:
        with self._lock:
            state = self._store.get(conv_id)
            return state.get("sent_bodies", []) if state else []

    def add_sent_body(self, conv_id: str, body: str):
        with self._lock:
            state = self.get_or_create(conv_id)
            state["sent_bodies"].append(body)
            state["last_bot_body"] = body

    def clear(self):
        with self._lock:
            self._store.clear()


class SuppressionStore:
    """
    Tracks suppression keys that have already been acted on.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._keys: Dict[str, str] = {}  # suppression_key -> conv_id

    def is_suppressed(self, key: str) -> bool:
        with self._lock:
            return key in self._keys

    def suppress(self, key: str, conv_id: str):
        with self._lock:
            self._keys[key] = conv_id

    def clear(self):
        with self._lock:
            self._keys.clear()


# Singletons
context_store = ContextStore()
conversation_store = ConversationStore()
suppression_store = SuppressionStore()
