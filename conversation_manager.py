"""
magicpin AI Challenge — Conversation Manager
===========================================
Handles multi-turn dialogue dynamics:
1. Auto-Reply Detection: Detects merchant WhatsApp Business automated responders
   and ends or backs off rather than getting stuck in loops.
2. Intent Transition: Switches immediately from qualification to action mode
   when the merchant commits ("Ok lets do it", "Yes", "Go ahead").
3. Hostile / Opt-out Handling: Gracefully exits or apologizes when the merchant
   requests no more messages.
4. General Dialogue: Context-aware responses respecting category tone.
"""

import re
from typing import Dict, Any, Tuple


# Common phrases in automated WhatsApp Business auto-replies
AUTO_REPLY_PATTERNS = [
    r"thank\s+you\s+for\s+contacting",
    r"thanks\s+for\s+reaching\s+out",
    r"our\s+team\s+will\s+respond",
    r"we\s+will\s+get\s+back\s+to\s+you",
    r"currently\s+(unavailable|away|closed)",
    r"auto(mated)?\s+reply",
    r"this\s+is\s+an\s+automated\s+message",
    r"please\s+leave\s+your\s+message",
    r"how\s+can\s+we\s+help\s+you\s+today",
    r"welcome\s+to\s+[a-zA-Z0-9\s]+!\s+how\s+can\s+we",
    r"outside\s+our\s+business\s+hours"
]

# Phrases indicating anger, spam complaint, or opt-out
HOSTILE_PATTERNS = [
    r"\bstop\b",
    r"\bunsubscribe\b",
    r"\bspam\b",
    r"\bdon'?t\s+message\b",
    r"\bstop\s+messaging\b",
    r"\bnot\s+interested\b",
    r"\bleave\s+me\s+alone\b",
    r"\bremove\s+me\b",
    r"\bblock(ed)?\b",
    r"\buseless\b",
    r"\bwaste\s+of\s+time\b",
    r"\bfuck\s*off\b",
    r"\bshut\s*up\b"
]

# Phrases indicating positive commitment / intent handoff
COMMITMENT_PATTERNS = [
    r"\blet'?s\s+do\s+it\b",
    r"\bwhat'?s\s+next\b",
    r"\bgo\s+ahead\b",
    r"\bi\s+want\s+to\s+join\b",
    r"\bi'?m\s+ready\s+to\s+(start|proceed)\b",
    r"\bhow\s+do\s+i\s+start\b",
    r"\bplease\s+(send|share)\b",
    r"\bsend\s+it\b",
    r"\bshare\s+it\b"
]


class ConversationManager:
    def __init__(self, context_store, llm_client=None):
        self.context_store = context_store
        self.llm_client = llm_client

    def is_auto_reply(self, message: str) -> bool:
        """Check if message matches typical canned auto-reply signatures."""
        clean = message.lower().strip()
        for pat in AUTO_REPLY_PATTERNS:
            if re.search(pat, clean):
                return True
        return False

    def is_hostile(self, message: str) -> bool:
        """Check if merchant wants to stop or is hostile."""
        clean = message.lower().strip()
        for pat in HOSTILE_PATTERNS:
            if re.search(pat, clean):
                return True
        return False

    def is_commitment(self, message: str) -> bool:
        """Check if merchant agreed to take action or asked what's next."""
        clean = message.lower().strip()
        for pat in COMMITMENT_PATTERNS:
            if re.search(pat, clean):
                return True
        return False

    def handle_reply(
        self,
        conv_id: str,
        merchant_id: str,
        customer_id: str | None,
        from_role: str,
        message: str,
        turn_number: int
    ) -> Dict[str, Any]:
        """
        Produce next action: 'send', 'wait', or 'end'.
        Returns compliant response dict.

        Auto-reply escalation (per spec api-call-examples.md §4.1):
          - 1st auto-reply → send one acknowledgment nudge aimed at human owner
          - 2nd auto-reply (same canned text again) → wait 24h
          - 3rd+ auto-reply → end conversation
        """
        conv = self.context_store.get_conversation(conv_id) or {}
        if not conv:
            self.context_store.record_conversation(conv_id, {
                "merchant_id": merchant_id,
                "customer_id": customer_id
            })
            conv = self.context_store.get_conversation(conv_id) or {}
        merchant = self.context_store.get_merchant(merchant_id) or {}
        owner_name = merchant.get("identity", {}).get("owner_first_name") or "there"
        merchant_biz = merchant.get("identity", {}).get("name", "your business")

        if conv.get("state") in {
            "ended_hostile", "ended_customer_reply", "ended_auto_reply", "ended_model"
        }:
            return {
                "action": "end",
                "cta": "none",
                "rationale": "Conversation is already closed."
            }

        # Track message history in conversation for repetition & auto-reply counting
        msg_history = conv.get("msg_history", [])

        # 1. Check for Auto-reply pattern
        if self.is_auto_reply(message):
            # Count how many times we've already seen an auto-reply in this conv
            ar_count = conv.get("auto_reply_count", 0) + 1
            self.context_store.update_conversation(conv_id, {
                "auto_reply_count": ar_count,
                "msg_history": msg_history + [message]
            })

            if ar_count == 1:
                # First auto-reply: send ONE nudge for the human owner to see
                return {
                    "action": "send",
                    "body": (
                        f"Looks like an auto-reply. When {owner_name} sees this, "
                        f"just reply 'Yes' to continue — I have a ready update for {merchant_biz}."
                    ),
                    "cta": "binary_yes_no",
                    "rationale": "Detected merchant auto-reply on turn 1; sending one human-readable nudge for owner, not the bot."
                }
            elif ar_count == 2:
                # Second consecutive auto-reply: back off 24h
                return {
                    "action": "wait",
                    "wait_seconds": 86400,
                    "rationale": "Same auto-reply 2× in a row — owner not at phone. Waiting 24h before retry."
                }
            else:
                # Third or more: end gracefully
                self.context_store.update_conversation(conv_id, {"state": "ended_auto_reply"})
                return {
                    "action": "end",
                    "rationale": f"Auto-reply {ar_count}× in a row with no real engagement signal. Closing conversation."
                }

        # 2. Check for Hostile / Opt-out. Do not send another promotional
        # message after an explicit stop request.
        if self.is_hostile(message):
            self.context_store.update_conversation(conv_id, {
                "msg_history": msg_history + [message],
                "state": "ended_hostile"
            })
            return {
                "action": "end",
                "cta": "none",
                "rationale": "Merchant expressed frustration or opted out. Conversation closed without another promotional message."
            }

        if self.llm_client and self.llm_client.enabled:
            trigger_id = conv.get("trigger_id")
            trigger_record = (
                self.context_store.get_trigger(trigger_id)
                if isinstance(trigger_id, str) else None
            )
            trigger = trigger_record.get("payload") if trigger_record else None
            customer = (
                self.context_store.get_customer(customer_id)
                if isinstance(customer_id, str) else None
            )
            result = self.llm_client.generate_reply(
                merchant=merchant,
                trigger=trigger,
                customer=customer,
                conversation=conv,
                message=message,
                from_role=from_role,
                turn_number=turn_number,
            )
            turns = conv.get("turns", []) + [{
                "role": from_role,
                "body": message,
                "turn_number": turn_number,
            }]
            updates = {
                "turns": turns,
                "auto_reply_count": 0,
                "msg_history": msg_history + [message],
            }
            if result["action"] == "send":
                assistant_turn = {
                    "role": "vera",
                    "body": result["body"],
                    "turn_number": turn_number,
                }
                turns.append(assistant_turn)
                updates.update({
                    "turns": turns,
                    "last_sent_body": result["body"],
                    "last_action": {
                        "action": "send",
                        "body": result["body"],
                        "cta": result["cta"],
                    },
                })
            elif result["action"] == "end":
                updates["state"] = "ended_model"
            self.context_store.update_conversation(conv_id, updates)
            return result

        if from_role == "customer":
            self.context_store.update_conversation(conv_id, {
                "msg_history": msg_history + [message],
                "state": "ended_customer_reply"
            })
            return {
                "action": "end",
                "cta": "none",
                "rationale": "Customer reply recorded; no booking or order execution is integrated, so no unsupported confirmation was sent."
            }

        # 3. Check for Commitment / "Let's do it" / Intent Transition
        short_confirmation = bool(re.fullmatch(
            r"(yes|yeah|yep|sure|ok|okay|go ahead|please do)[.! ]*",
            message.strip().lower()
        ))
        previous_action = conv.get("last_action", {})
        confirms_previous_action = (
            short_confirmation
            and previous_action.get("cta") in {"binary", "binary_yes_no"}
        )
        if self.is_commitment(message) or confirms_previous_action:
            # Switch to ACTION mode — NO qualifying questions!
            active_offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
            if active_offers and active_offers[0]:
                top_offer = active_offers[0]
                body = (
                    f"Done — switching to action mode. Here is a draft using your active offer:\n\n"
                    f"\"{merchant_biz}: {top_offer}. Contact us for details.\"\n\n"
                    f"Please review the copy before it is sent."
                )
            else:
                body = (
                    f"Done — switching to action mode for {merchant_biz}. "
                    "I don't see an active offer in the supplied catalog, so I won't invent one. "
                    "Add the offer details and I can prepare the customer copy."
                )
            self.context_store.update_conversation(conv_id, {
                "msg_history": msg_history + [message],
                "state": "action_mode"
            })
            return {
                "action": "send",
                "body": body,
                "cta": "binary",
                "rationale": "Merchant committed. Switched to action mode without another qualifying question or unverified offer terms."
            }

        # 4. Check if merchant asked to wait / busy
        wait_patterns = [r"\blater\b", r"\bbusy\b", r"\bcall\s+me\s+later\b", r"\btomorrow\b", r"\bnot\s+now\b"]
        if any(re.search(p, message.lower()) for p in wait_patterns):
            self.context_store.update_conversation(conv_id, {"msg_history": msg_history + [message]})
            return {
                "action": "wait",
                "wait_seconds": 3600,
                "rationale": "Merchant indicated they are busy. Backing off for 1 hour."
            }

        # 5. Anti-repetition: check if we'd send the same body as last time
        last_body = conv.get("last_sent_body", "")

        # 6. Default progressive response
        self.context_store.update_conversation(conv_id, {
            "msg_history": msg_history + [message],
            "auto_reply_count": 0  # reset counter on real engagement
        })
        body = (
            f"Understood, {owner_name}. What would you like to adjust in the proposed update for {merchant_biz}?"
        )
        # Avoid repetition penalty
        if body == last_body:
            body = (
                f"Great {owner_name}! Confirming the campaign is queued for {merchant_biz}. "
                f"Tell me what you'd like to revise in the proposed copy."
            )
        self.context_store.update_conversation(conv_id, {"last_sent_body": body})
        return {
            "action": "send",
            "body": body,
            "cta": "binary",
            "rationale": "Progressing conversation toward low-friction execution."
        }
