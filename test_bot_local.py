"""
magicpin AI Challenge — Local Test Verification Suite
=====================================================
Validates bot endpoints and logic locally before running judge_simulator.py:
1. /v1/healthz
2. /v1/metadata
3. /v1/context (Category, Merchant, Customer, Trigger)
4. /v1/tick (Action generation, suppression, format)
5. /v1/reply (Auto-reply backoff, Hostile end, Intent action transition)
"""

import time
import json
import threading
from urllib import request as urlrequest, error as urlerror
from pathlib import Path

import server
from server import app, store


def run_test_server():
    app.run(host="127.0.0.1", port=8081, debug=False, use_reloader=False)


def http_req(method: str, path: str, body_dict: dict = None):
    url = f"http://127.0.0.1:8081{path}"
    data = json.dumps(body_dict).encode("utf-8") if body_dict else None
    req = urlrequest.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        resp = urlrequest.urlopen(req, timeout=10)
        return json.loads(resp.read().decode("utf-8")), None, resp.status
    except urlerror.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8")), str(e), e.code
        except Exception:
            return None, str(e), e.code
    except Exception as e:
        return None, str(e), 500


def run_tests():
    # Keep this suite offline and deterministic even when a developer has a local key.
    server.llm_client.api_key = None

    print("\n" + "=" * 60)
    print("RUNNING VERA BOT LOCAL TEST SUITE")
    print("=" * 60 + "\n")

    # Start server in daemon thread
    server_thread = threading.Thread(target=run_test_server, daemon=True)
    server_thread.start()
    time.sleep(1.5)

    # 1. Healthz
    data, err, status = http_req("GET", "/v1/healthz")
    assert status == 200 and data.get("status") == "ok", f"healthz failed: {err}"
    print("[PASS] 1. GET /v1/healthz passed")

    # 2. Metadata
    data, err, status = http_req("GET", "/v1/metadata")
    assert status == 200 and data.get("team_name"), f"metadata failed: {err}"
    print(f"[PASS] 2. GET /v1/metadata passed (Team: {data.get('team_name')})")

    # 3. Context Push (Category, Merchant, Trigger)
    cat_file = Path("expanded/categories/dentists.json")
    if cat_file.exists():
        cat_data = json.load(open(cat_file))
        data, err, status = http_req("POST", "/v1/context", {
            "scope": "category",
            "context_id": "dentists",
            "version": 1,
            "payload": cat_data,
            "delivered_at": "2026-04-26T10:00:00Z"
        })
        assert status == 200 and data.get("accepted"), f"category push failed: {err}"
        print("[PASS] 3a. POST /v1/context (category) passed")

    m_file = Path("expanded/merchants/m_001_drmeera_dentist_delhi.json")
    if m_file.exists():
        m_data = json.load(open(m_file))
        data, err, status = http_req("POST", "/v1/context", {
            "scope": "merchant",
            "context_id": "m_001_drmeera_dentist_delhi",
            "version": 1,
            "payload": m_data,
            "delivered_at": "2026-04-26T10:00:00Z"
        })
        assert status == 200 and data.get("accepted"), f"merchant push failed: {err}"
        print("[PASS] 3b. POST /v1/context (merchant) passed")

    trg_file = Path("expanded/triggers/trg_001_research_digest_dentists.json")
    if trg_file.exists():
        trg_data = json.load(open(trg_file))
        data, err, status = http_req("POST", "/v1/context", {
            "scope": "trigger",
            "context_id": "trg_001_research_digest_dentists",
            "version": 1,
            "payload": trg_data,
            "delivered_at": "2026-04-26T10:00:00Z"
        })
        assert status == 200 and data.get("accepted"), f"trigger push failed: {err}"
        print("[PASS] 3c. POST /v1/context (trigger) passed")

    # 4. Tick
    data, err, status = http_req("POST", "/v1/tick", {
        "now": "2026-04-26T10:30:00Z",
        "available_triggers": ["trg_001_research_digest_dentists"]
    })
    assert status == 200 and "actions" in data, f"tick failed: {err}"
    actions = data.get("actions", [])
    assert len(actions) == 1, f"Expected 1 action, got {len(actions)}"
    act = actions[0]
    print(f"[PASS] 4. POST /v1/tick passed (Generated {len(actions)} action)")
    print(f"       Action Body: \"{act.get('body')[:70]}...\"")
    print(f"       CTA: {act.get('cta')}, Suppression Key: {act.get('suppression_key')}")
    assert act.get("template_body") == "{{1}}" and act.get("template_params") == [act.get("body")]

    # Same version replay is accepted without duplicating the context entry.
    duplicate_data, _, duplicate_status = http_req("POST", "/v1/context", {
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": cat_data
    })
    assert duplicate_status == 200 and duplicate_data.get("accepted"), "same-version context replay should be idempotent"
    higher_data, _, higher_status = http_req("POST", "/v1/context", {
        "scope": "category", "context_id": "dentists",
        "version": 2, "payload": cat_data
    })
    assert higher_status == 200 and higher_data.get("accepted"), "higher context version should replace existing context"
    stale_data, _, stale_status = http_req("POST", "/v1/context", {
        "scope": "category", "context_id": "dentists",
        "version": 1, "payload": cat_data
    })
    assert stale_status == 409 and stale_data.get("reason") == "stale_version", "older context version should conflict"

    # Never route an unknown merchant trigger to an arbitrary loaded merchant.
    unknown_trigger = {
        "id": "trg_unknown_merchant",
        "scope": "merchant",
        "kind": "curious_ask_due",
        "merchant_id": "m_not_loaded",
        "payload": {},
        "suppression_key": "test:unknown-merchant"
    }
    data, err, status = http_req("POST", "/v1/context", {
        "scope": "trigger", "context_id": unknown_trigger["id"],
        "version": 1, "payload": unknown_trigger
    })
    assert status == 200 and data.get("accepted"), f"unknown trigger push failed: {err}"
    data, err, status = http_req("POST", "/v1/tick", {
        "available_triggers": [unknown_trigger["id"]]
    })
    assert status == 200 and data.get("actions") == [], "unknown merchant trigger must not be sent to another merchant"

    # A customer trigger is withheld without purpose-specific opt-in, then sent
    # after a higher customer-context version supplies the required consent.
    customer_file = Path("expanded/customers/c_001_priya_for_m001.json")
    customer_data = json.loads(customer_file.read_text(encoding="utf-8"))
    customer_data["consent"]["scope"] = ["appointment_reminders"]
    customer_context = {
        "scope": "customer", "context_id": "c_001_priya_for_m001",
        "version": 1, "payload": customer_data
    }
    data, err, status = http_req("POST", "/v1/context", customer_context)
    assert status == 200 and data.get("accepted"), f"customer push failed: {err}"

    recall = json.loads(Path("expanded/triggers/trg_003_recall_due_priya.json").read_text(encoding="utf-8"))
    recall_context = {
        "scope": "trigger", "context_id": recall["id"],
        "version": 1, "payload": recall
    }
    data, err, status = http_req("POST", "/v1/context", recall_context)
    assert status == 200 and data.get("accepted"), f"recall trigger push failed: {err}"
    data, err, status = http_req("POST", "/v1/tick", {"available_triggers": [recall["id"]]})
    assert status == 200 and data.get("actions") == [], "customer outreach without recall consent must be withheld"

    customer_data["consent"]["scope"] = ["appointment_reminders", "recall_reminders"]
    customer_context["version"] = 2
    customer_context["payload"] = customer_data
    data, err, status = http_req("POST", "/v1/context", customer_context)
    assert status == 200 and data.get("accepted"), f"updated customer consent push failed: {err}"
    data, err, status = http_req("POST", "/v1/tick", {"available_triggers": [recall["id"]]})
    assert status == 200 and len(data.get("actions", [])) == 1, "valid customer consent should permit the recall message"
    assert data["actions"][0].get("send_as") == "merchant_on_behalf"
    customer_action = data["actions"][0]
    data, err, status = http_req("POST", "/v1/reply", {
        "conversation_id": customer_action["conversation_id"],
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": "c_001_priya_for_m001",
        "from_role": "customer",
        "message": "YES",
        "turn_number": 2
    })
    assert status == 200 and data.get("action") == "end", "customer confirmation must not be treated as merchant campaign approval"

    # 5. Auto-Reply Detection: nudge once, wait on repeat, then end.
    auto_reply_msg = "Thank you for contacting us! Our team will respond shortly."
    auto_payload = {
        "conversation_id": "conv_test_auto",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": auto_reply_msg,
        "turn_number": 2
    }
    data, err, status = http_req("POST", "/v1/reply", auto_payload)
    assert status == 200, f"auto-reply request failed: {err}"
    assert data.get("action") == "send", f"Expected first auto-reply action='send', got {data.get('action')}"
    data, err, status = http_req("POST", "/v1/reply", auto_payload)
    assert data.get("action") == "wait", f"Expected repeated auto-reply action='wait', got {data.get('action')}"
    data, err, status = http_req("POST", "/v1/reply", auto_payload)
    assert data.get("action") == "end", f"Expected third auto-reply action='end', got {data.get('action')}"
    print("[PASS] 5. POST /v1/reply (Auto-Reply Detection) backed off and ended correctly")

    # 6. Hostile Detection
    hostile_msg = "Stop messaging me. This is useless spam."
    data, err, status = http_req("POST", "/v1/reply", {
        "conversation_id": "conv_test_hostile",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": hostile_msg,
        "turn_number": 2
    })
    assert status == 200, f"hostile request failed: {err}"
    assert data.get("action") == "end", f"Expected action='end' for hostile, got {data.get('action')}"
    print("[PASS] 6. POST /v1/reply (Hostile Handling) correctly returned action='end'")

    # 7. Intent Transition (Switch to Action Mode)
    intent_msg = "Ok lets do it. Whats next?"
    data, err, status = http_req("POST", "/v1/reply", {
        "conversation_id": "conv_test_intent",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": intent_msg,
        "turn_number": 2
    })
    assert status == 200, f"intent request failed: {err}"
    assert data.get("action") == "send", f"Expected action='send' for intent, got {data.get('action')}"
    body_lower = data.get("body", "").lower()
    actioning = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]
    qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
    assert any(w in body_lower for w in actioning), f"Expected actioning words in body, got: {data.get('body')}"
    assert not any(w in body_lower for w in qualifying), f"Should not qualify after commitment, got: {data.get('body')}"
    print("[PASS] 7. POST /v1/reply (Intent Transition) correctly switched to Action Mode without re-qualifying")

    # A bare "yes" without a preceding action is not enough to infer intent.
    data, err, status = http_req("POST", "/v1/reply", {
        "conversation_id": "conv_ambiguous_yes",
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Yes",
        "turn_number": 2
    })
    assert status == 200 and "Done" not in data.get("body", ""), "bare yes without conversation context must not trigger action mode"

    # Teardown clears every context and conversation.
    data, err, status = http_req("POST", "/v1/teardown")
    assert status == 200 and data.get("status") == "cleared", f"teardown failed: {err}"
    data, err, status = http_req("GET", "/v1/healthz")
    assert status == 200 and sum(data.get("contexts_loaded", {}).values()) == 0, "teardown must clear stored contexts"

    print("\n" + "=" * 60)
    print("ALL LOCAL ENDPOINT AND SAFETY TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_tests()
