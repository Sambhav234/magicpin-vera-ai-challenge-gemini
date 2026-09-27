"""Offline tests for structured Groq requests and reply integration."""

import json
from io import BytesIO
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from context_store import ContextStore
from conversation_manager import ConversationManager
from llm_client import GroqClient, ModelError
import server


class _Response:
    def __init__(self, result):
        self.result = result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "choices": [{
                "message": {"content": json.dumps(self.result)}
            }],
        }).encode("utf-8")


class _ReplyModel:
    enabled = True

    def __init__(self, result):
        self.result = result
        self.calls = []

    def generate_reply(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class GroqClientTests(unittest.TestCase):
    def test_provider_http_error_includes_groq_diagnostic(self):
        client = GroqClient(api_key="test-key")
        provider_error = HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            503,
            "Service Unavailable",
            {},
            BytesIO(b'{"error":{"status":"UNAVAILABLE","message":"The model is overloaded."}}'),
        )
        with patch("llm_client.urlopen", side_effect=provider_error):
            with self.assertRaisesRegex(
                ModelError,
                r"Groq request failed with HTTP 503 \(UNAVAILABLE\): The model is overloaded",
            ):
                client._generate_json({"task": "test"}, {})

    def test_tick_uses_model_composer_when_configured(self):
        server.store.clear_all()
        server.store.store_context("category", "dentists", 1, {"slug": "dentists"})
        server.store.store_context("merchant", "m1", 1, {
            "merchant_id": "m1",
            "category_slug": "dentists",
            "identity": {"name": "Dental Clinic"},
        })
        server.store.store_context("trigger", "t1", 1, {
            "id": "t1",
            "scope": "merchant",
            "kind": "curious_ask_due",
            "merchant_id": "m1",
            "payload": {},
        })
        generated = {
            "body": "What would your customers find most useful this week?",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": "test-suppression",
            "rationale": "A relevant conversational prompt.",
        }
        with patch.object(server.llm_client, "api_key", "test-key"), \
                patch.object(server.llm_client, "compose_message", return_value=generated) as compose:
            response = server.app.test_client().post(
                "/v1/tick",
                json={"available_triggers": ["t1"]},
            )
        server.store.clear_all()
        self.assertEqual(response.status_code, 200)
        action = response.get_json()["actions"][0]
        self.assertEqual(action["body"], generated["body"])
        self.assertEqual(action["send_as"], "vera")
        self.assertEqual(action["suppression_key"], "test-suppression")
        compose.assert_called_once()

    def test_structured_request_hides_key_and_redacts_customer_context(self):
        client = GroqClient(api_key="not-a-real-key", model="openai/gpt-oss-20b")
        result = {
            "body": "Hello, your Dental Cleaning @ ₹299 is due. Reply YES to arrange a visit.",
            "cta": "binary",
            "rationale": "Uses the supplied service and price.",
        }
        category = {"slug": "dentists", "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}]}
        merchant = {"merchant_id": "merchant-private", "identity": {"name": "Dental Clinic"}}
        trigger = {"id": "trigger-private", "kind": "recall_due", "payload": {"service_due": "Dental Cleaning"}}
        customer = {
            "customer_id": "customer-private",
            "identity": {"first_name": "Priya", "phone": "+91 98765 43210"},
            "consent": {"opted_in_at": "2026-01-01"},
        }
        draft = {
            "body": "Dental Cleaning @ ₹299 is due.",
            "cta": "binary",
            "send_as": "merchant_on_behalf",
            "suppression_key": "recall:customer-private:trigger-private",
            "rationale": "Grounded draft.",
        }

        with patch("llm_client.urlopen", return_value=_Response(result)) as mocked_urlopen:
            composed = client.compose_message(category, merchant, trigger, customer, draft)

        request = mocked_urlopen.call_args.args[0]
        self.assertNotIn("not-a-real-key", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer not-a-real-key")
        sent_body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(sent_body["model"], "openai/gpt-oss-20b")
        self.assertEqual(sent_body["response_format"]["type"], "json_schema")
        self.assertTrue(sent_body["response_format"]["json_schema"]["strict"])
        sent_text = sent_body["messages"][1]["content"]
        self.assertNotIn("customer-private", sent_text)
        self.assertNotIn("98765", sent_text)
        self.assertEqual(composed["body"], result["body"])
        self.assertEqual(composed["send_as"], "merchant_on_behalf")
        self.assertEqual(composed["suppression_key"], draft["suppression_key"])

    def test_composition_rejects_unreferenced_numeric_claim(self):
        client = GroqClient(api_key="test-key")
        result = {
            "body": "Your profile received 999 new calls.",
            "cta": "binary",
            "rationale": "Performance message.",
        }
        with patch("llm_client.urlopen", return_value=_Response(result)):
            with self.assertRaises(ModelError):
                client.compose_message(
                    {"slug": "dentists"},
                    {"identity": {"name": "Dental Clinic"}},
                    {"kind": "perf_spike", "payload": {"metric": "calls", "delta_pct": 0.2}},
                    None,
                    {"body": "Calls rose 20%.", "cta": "binary", "send_as": "vera"},
                )

    def test_general_reply_uses_model_and_opt_out_bypasses_it(self):
        store = ContextStore()
        store.store_context("merchant", "m1", 1, {
            "merchant_id": "m1",
            "identity": {"name": "Cafe"},
            "category_slug": "restaurants",
        })
        store.record_conversation("c1", {"merchant_id": "m1", "last_action": {"cta": "open_ended"}})
        model = _ReplyModel({
            "action": "send",
            "body": "I can help revise the post. What would you like to change?",
            "cta": "open_ended",
            "rationale": "Responds to the merchant's request.",
        })
        manager = ConversationManager(store, model)

        response = manager.handle_reply("c1", "m1", None, "merchant", "Can you revise the post?", 2)
        self.assertEqual(response["action"], "send")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(store.get_conversation("c1")["last_action"]["body"], response["body"])

        stopped = manager.handle_reply("c2", "m1", None, "merchant", "Stop messaging me", 2)
        self.assertEqual(stopped["action"], "end")
        self.assertEqual(len(model.calls), 1)

    def test_wait_requires_a_bounded_delay(self):
        client = GroqClient(api_key="test-key")
        result = {
            "action": "wait",
            "body": "",
            "cta": "none",
            "wait_seconds": 5,
            "rationale": "Try later.",
        }
        with patch("llm_client.urlopen", return_value=_Response(result)):
            with self.assertRaises(ModelError):
                client.generate_reply({}, None, None, {}, "Later", "merchant", 2)


if __name__ == "__main__":
    unittest.main()
