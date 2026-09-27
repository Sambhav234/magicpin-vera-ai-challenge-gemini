# VeraCraft — magicpin AI Challenge

## Approach

This submission implements a deterministic, context-grounded Vera assistant. It composes
messages from the four pushed context layers:

- **Category:** vertical voice, peer benchmarks, digest items, and catalog patterns.
- **Merchant:** identity, locality, performance, verified offers, subscription, and signals.
- **Trigger:** the event that explains why Vera is contacting the merchant now.
- **Customer:** relationship, language preference, consent-related outreach context, and
  appointment/recall details when the message is sent on the merchant's behalf.

The composer uses trigger-specific templates with one primary CTA. It only states facts
present in the supplied contexts. Customer outreach requires a recorded opt-in for the
specific message purpose and a WhatsApp channel preference. Missing consent, mismatched
merchant/customer data, placeholder data, or unsupported trigger types produce no
outbound action. The composer does not invent prices, dates, citations, statistics,
schedules, competitor claims, medical claims, or completed campaign work.

## API and state

`server.py` exposes the required `GET /v1/healthz`, `GET /v1/metadata`,
`POST /v1/context`, `POST /v1/tick`, and `POST /v1/reply` endpoints, plus the optional
`POST /v1/teardown` endpoint. `ContextStore` is thread-safe and supports atomic
higher-version replacement and same-version idempotent replay.

The reply state machine:

1. Sends one human-owner nudge for the first canned auto-reply.
2. Waits 24 hours after a repeated auto-reply.
3. Ends after the third repeated auto-reply.
4. Ends immediately on an explicit opt-out/hostile message.
5. Switches directly to action mode when the merchant commits.
6. Treats a bare affirmative as commitment only when it follows a recorded binary action.

## Run locally

```powershell
python server.py
python test_bot_local.py
python generate_submission.py
```

The server listens on port `8081` by default. `submission.jsonl` contains the generated
30-line canonical submission. No external model or API key is required.

## Deploy on Render

The repository includes `render.yaml`, `requirements.txt`, and `.python-version`.

1. Push the project to a GitHub repository.
2. In Render, choose **New + → Blueprint** and connect that repository.
3. Review the `veracraft-bot` web service. The Blueprint defaults to Render's free plan.
   Free instances can spin down when idle; upgrade to a paid plan before evaluation if
   the challenge requires uninterrupted availability.
4. Enter `VERA_TEAM_NAME`, `VERA_CANDIDATE_NAME`, and `VERA_CONTACT_EMAIL` when prompted.
   Optionally set `VERA_SUBMITTED_AT` to the submission timestamp in ISO-8601 UTC format;
   otherwise metadata reports the service process start time.
5. Deploy, then verify `https://<your-service>.onrender.com/v1/healthz` and
   `/v1/metadata`.
6. Submit the base HTTPS URL to the challenge portal and keep the service running.

The outbound action includes a one-parameter template (`{{1}}`) containing the composed
message body, plus its `template_params` value.

Gunicorn is configured with one worker because context and conversation state are held
in memory; multiple workers would maintain separate, inconsistent copies of that state.
The service must remain available throughout evaluation. Render's free instances can
spin down and restart, which temporarily affects availability and clears in-memory state.

## Tradeoff

The implementation favors verifiability and safe restraint over persuasive copy when
context is incomplete. Additional structured context for placeholder triggers would allow
more specific customer and merchant messages without violating the no-fabrication rule.
