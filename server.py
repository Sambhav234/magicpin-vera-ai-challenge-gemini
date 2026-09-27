"""
magicpin AI Challenge — Candidate Bot HTTP API Server
====================================================
Exposes the 5 mandatory endpoints required by the challenge contract:
1. GET  /v1/healthz     — Liveness probe (uptime and context counts)
2. GET  /v1/metadata    — Bot identity, model, approach, and version
3. POST /v1/context     — Idempotent context ingestion across 4 layers
4. POST /v1/tick        — Periodic wake-up (returns up to 20 composed actions)
5. POST /v1/reply       — Multi-turn response (send / wait / end within 30s)

Default local port: 8081; Render supplies its own PORT.
"""

import os
import sys
import json
import logging
import uuid
import time
from datetime import datetime, timezone
from flask import Flask, request, jsonify

from context_store import ContextStore
from conversation_manager import ConversationManager
from bot import compose
from llm_client import GeminiClient, ModelError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("VeraServer")

app = Flask(__name__)

# Singletons
store = ContextStore()
llm_client = GeminiClient()
conv_manager = ConversationManager(store, llm_client)

# Metadata configuration
METADATA = {
    "team_name": os.environ.get("VERA_TEAM_NAME", "Team VeraCraft"),
    "team_members": [os.environ.get("VERA_CANDIDATE_NAME", "Sambhav Mishra")],
    "model": llm_client.model if llm_client.enabled else "hybrid-template-vera",
    "approach": (
        "Gemini structured generation grounded in four contexts with deterministic consent, "
        "routing, and conversation safety checks"
        if llm_client.enabled else
        "Deterministic four-context composition with adaptive multi-turn conversation routing"
    ),
    "contact_email": os.environ.get("VERA_CONTACT_EMAIL", "sambhavmishra234@gmail.com"),
    "version": "1.0.0",
    "submitted_at": os.environ.get(
        "VERA_SUBMITTED_AT",
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
}


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.route("/v1/healthz", methods=["GET"])
def healthz():
    """Liveness probe: reports uptime and counts of stored context."""
    return jsonify({
        "status": "ok",
        "uptime_seconds": store.get_uptime_seconds(),
        "contexts_loaded": store.get_stats()
    }), 200


@app.route("/v1/metadata", methods=["GET"])
def metadata():
    """Bot identity for the evaluation harness."""
    return jsonify(METADATA), 200


@app.route("/v1/context", methods=["POST"])
def push_context():
    """
    Idempotent context push endpoint.
    Payload: { scope, context_id, version, payload, delivered_at }
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"accepted": False, "reason": "invalid_json"}), 400

    scope = data.get("scope")
    context_id = data.get("context_id")
    version = data.get("version", 1)
    payload = data.get("payload")

    if not scope or not context_id or payload is None:
        return jsonify({"accepted": False, "reason": "missing_required_fields"}), 400
    if scope not in {"category", "merchant", "customer", "trigger"}:
        return jsonify({"accepted": False, "reason": "invalid_scope"}), 400
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        return jsonify({"accepted": False, "reason": "invalid_version"}), 400
    if not isinstance(payload, dict):
        return jsonify({"accepted": False, "reason": "invalid_payload"}), 400

    accepted, error, curr_ver = store.store_context(scope, context_id, version, payload)

    if not accepted:
        if error == "stale_version":
            return jsonify({
                "accepted": False,
                "reason": "stale_version",
                "current_version": curr_ver
            }), 409
        return jsonify({
            "accepted": False,
            "reason": error or "storage_failed"
        }), 400

    ack_id = f"ack_{context_id}_v{curr_ver}"
    return jsonify({
        "accepted": True,
        "ack_id": ack_id,
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }), 200


@app.route("/v1/tick", methods=["POST"])
def tick():
    """
    Periodic wake-up endpoint.
    Payload: { now, available_triggers }
    Returns: { actions: [...] } (max 20)
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid_json"}), 400
    available_triggers = data.get("available_triggers", [])
    if not isinstance(available_triggers, list) or not all(isinstance(tid, str) for tid in available_triggers):
        return jsonify({"error": "available_triggers_must_be_a_list_of_ids"}), 400

    actions = []
    model_deadline = time.monotonic() + 24

    for tid in available_triggers:
        if len(actions) >= 20:
            break

        trigger = store.get_trigger(tid)
        if not trigger:
            logger.debug(f"Trigger {tid} not yet pushed to context store, skipping")
            continue

        supp_key = trigger.get("suppression_key")
        if supp_key and store.is_suppressed(supp_key):
            logger.debug(f"Trigger {tid} suppressed by key {supp_key}")
            continue

        # Resolve target merchant
        trigger_payload = trigger.get("payload", {})
        if not isinstance(trigger_payload, dict):
            logger.warning(f"Trigger {tid} has invalid payload, skipping")
            continue

        mid = trigger.get("merchant_id") or trigger_payload.get("merchant_id")
        merchant = store.get_merchant(mid) if mid else None
        if not merchant or merchant.get("merchant_id") != mid:
            logger.debug(f"Merchant context missing or mismatched for trigger {tid}, skipping")
            continue

        # Resolve target customer
        cid = trigger.get("customer_id") or trigger_payload.get("customer_id")
        customer = store.get_customer(cid) if cid else None
        if trigger.get("scope") == "customer" and (not cid or not customer):
            logger.debug(f"Customer context missing for customer trigger {tid}, skipping")
            continue
        if customer and (customer.get("customer_id") != cid or customer.get("merchant_id") != mid):
            logger.warning(f"Customer {cid} does not belong to merchant {mid}, skipping trigger {tid}")
            continue

        # Resolve category
        cat_slug = merchant.get("category_slug") or trigger.get("payload", {}).get("category")
        category = store.get_category(cat_slug) if cat_slug else None
        if not category or category.get("slug") != cat_slug:
            logger.debug(f"Category context missing for trigger {tid}, skipping")
            continue

        # Compose message
        composed = compose(category, merchant, trigger, customer)
        if not composed.get("body"):
            logger.debug(f"No safe message available for trigger {tid}, skipping")
            continue
        if llm_client.enabled:
            remaining = model_deadline - time.monotonic()
            if remaining < 1:
                logger.warning("Tick model time budget exhausted; remaining triggers were skipped.")
                break
            try:
                composed = llm_client.compose_message(
                    category,
                    merchant,
                    trigger,
                    customer,
                    composed,
                    timeout_seconds=min(llm_client.timeout_seconds, remaining),
                )
            except ModelError as error:
                logger.warning("Gemini composition failed for trigger %s: %s", tid, error)
                continue
        conv_id = f"conv_{uuid.uuid4().hex}"

        # Keep the first outbound inside a declared message-template contract.
        send_as = composed.get("send_as", "vera")
        t_kind = trigger.get("kind", "generic")
        template_name = f"vera_{t_kind}_v1" if send_as == "vera" else f"merchant_{t_kind}_v1"
        body_text = composed.get("body", "")
        # Extract first 120 chars of body as second param, rest as third
        template_params = [body_text]

        action = {
            "conversation_id": conv_id,
            "merchant_id": mid,
            "customer_id": cid,
            "send_as": send_as,
            "trigger_id": tid,
            "template_name": template_name,
            "template_body": "{{1}}",
            "template_params": template_params,
            "body": body_text,
            "cta": composed.get("cta", "binary"),
            "suppression_key": composed.get("suppression_key", supp_key or f"supp_{tid}"),
            "rationale": composed.get("rationale", "")
        }

        # Record suppression and conversation
        if composed.get("suppression_key"):
            store.record_suppression(composed["suppression_key"])

        store.record_conversation(conv_id, {
            "merchant_id": mid,
            "customer_id": cid,
            "trigger_id": tid,
            "last_action": action
        })

        actions.append(action)

    return jsonify({"actions": actions}), 200


@app.route("/v1/reply", methods=["POST"])
def reply():
    """
    Multi-turn conversation reply endpoint.
    Payload: { conversation_id, merchant_id, customer_id, from_role, message, turn_number, received_at }
    Returns: { action: 'send' | 'wait' | 'end', body?, cta?, rationale }
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid_json"}), 400
    conv_id = data.get("conversation_id")
    merchant_id = data.get("merchant_id")
    customer_id = data.get("customer_id")
    from_role = data.get("from_role")
    message = data.get("message")
    turn_number = data.get("turn_number")
    conversation = store.get_conversation(conv_id) if isinstance(conv_id, str) else None
    if conversation and conversation.get("merchant_id") != merchant_id:
        return jsonify({"error": "conversation_merchant_mismatch"}), 400
    if conversation and conversation.get("customer_id") != customer_id:
        return jsonify({"error": "conversation_customer_mismatch"}), 400
    if (not isinstance(conv_id, str) or not conv_id
            or not isinstance(merchant_id, str) or not merchant_id
            or from_role not in {"merchant", "customer"}
            or not isinstance(message, str) or not message.strip()
            or isinstance(turn_number, bool) or not isinstance(turn_number, int) or turn_number < 1):
        return jsonify({"error": "missing_or_invalid_reply_fields"}), 400

    try:
        result = conv_manager.handle_reply(
            conv_id=conv_id,
            merchant_id=merchant_id,
            customer_id=customer_id,
            from_role=from_role,
            message=message,
            turn_number=turn_number
        )
    except ModelError as error:
        logger.warning("Gemini reply generation failed for conversation %s: %s", conv_id, error)
        return jsonify({"error": "model_unavailable", "detail": str(error)}), 503

    return jsonify(result), 200


@app.route("/v1/teardown", methods=["POST"])
def teardown():
    """
    Optional end-of-test teardown: wipe all in-memory state.
    Spec: challenge-testing-brief.md §11
    """
    store.clear_all()
    logger.info("Teardown called — all state wiped.")
    return jsonify({"status": "cleared"}), 200


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_server(port: int = 8081):
    logger.info(f"Starting Vera Bot Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8081))
    run_server(port)
