"""
magicpin AI Challenge — Context Store
====================================
Thread-safe, in-memory state store for the 4 context layers:
- CategoryContext
- MerchantContext
- TriggerContext
- CustomerContext

Handles idempotent updates, version conflicts, and fast retrieval.
"""

import threading
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


class ContextStore:
    def __init__(self):
        self._lock = threading.RLock()
        self.categories: Dict[str, Dict[str, Any]] = {}
        self.merchants: Dict[str, Dict[str, Any]] = {}
        self.customers: Dict[str, Dict[str, Any]] = {}
        self.triggers: Dict[str, Dict[str, Any]] = {}
        self.versions: Dict[Tuple[str, str], int] = {}
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.suppressed_keys: set = set()
        self.start_time = datetime.utcnow()

    def store_context(self, scope: str, context_id: str, version: int, payload: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[int]]:
        """
        Store context idempotently.
        Returns: (accepted, error_reason, current_version)
        """
        with self._lock:
            key = (scope, context_id)
            current_version = self.versions.get(key, -1)

            if version == current_version:
                # Replaying a delivered version is an idempotent no-op.
                return True, None, current_version
            if version < current_version:
                # Older versions cannot replace newer context.
                return False, "stale_version", current_version

            # Higher version: replace atomically
            if scope == "category":
                self.categories[context_id] = payload
            elif scope == "merchant":
                self.merchants[context_id] = payload
            elif scope == "customer":
                self.customers[context_id] = payload
            elif scope == "trigger":
                self.triggers[context_id] = payload
                # When trigger is updated or re-pushed, allow re-evaluation
                supp_key = payload.get("suppression_key")
                if supp_key and supp_key in self.suppressed_keys:
                    self.suppressed_keys.discard(supp_key)
            else:
                return False, f"invalid_scope: {scope}", None

            self.versions[key] = version
            return True, None, version

    def get_category(self, slug: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.categories.get(slug)

    def get_merchant(self, merchant_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.merchants.get(merchant_id)

    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.customers.get(customer_id)

    def get_trigger(self, trigger_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.triggers.get(trigger_id)

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "category": len(self.categories),
                "merchant": len(self.merchants),
                "customer": len(self.customers),
                "trigger": len(self.triggers)
            }

    def get_uptime_seconds(self) -> int:
        return int((datetime.utcnow() - self.start_time).total_seconds())

    def record_suppression(self, key: str):
        with self._lock:
            if key:
                self.suppressed_keys.add(key)

    def is_suppressed(self, key: str) -> bool:
        with self._lock:
            return key in self.suppressed_keys

    def record_conversation(self, conv_id: str, data: Dict[str, Any]):
        with self._lock:
            if conv_id not in self.active_conversations:
                self.active_conversations[conv_id] = {
                    "turns": [],
                    "merchant_id": data.get("merchant_id"),
                    "customer_id": data.get("customer_id"),
                    "trigger_id": data.get("trigger_id"),
                    "state": "initiated",
                    "auto_reply_count": 0,
                    "created_at": datetime.utcnow().isoformat() + "Z"
                }
            self.active_conversations[conv_id].update(data)

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.active_conversations.get(conv_id)

    def update_conversation(self, conv_id: str, updates: Dict[str, Any]):
        with self._lock:
            conv = self.active_conversations.get(conv_id, {})
            conv.update(updates)
            self.active_conversations[conv_id] = conv

    def clear_all(self):
        """Wipe all state for teardown (spec §11)."""
        with self._lock:
            self.categories.clear()
            self.merchants.clear()
            self.customers.clear()
            self.triggers.clear()
            self.versions.clear()
            self.active_conversations.clear()
            self.suppressed_keys.clear()
