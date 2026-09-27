# Vera merchant growth assistant — V2

Vera is a deterministic FastAPI service for the Magicpin challenge. It proposes grounded merchant/customer messages from the most recently pushed category, merchant, trigger, and optional customer contexts. It makes no external data or LLM calls.

## Decision and message pipeline

1. Validate request shape, scope, version, and payload size. Context is keyed by `(scope, context_id)`; an identical version is idempotent, a conflicting or older version is rejected, and a higher version replaces the stored payload.
2. For each active trigger ID, load the latest merchant/category/customer contexts. Reject missing context, expired triggers, merchant/customer ownership mismatches, unknown customer-trigger kinds, and customer sends without matching opt-in scope and WhatsApp channel preference.
3. Generate a grounded candidate. Category-specific templates cover dentists, salons, restaurants, gyms, and pharmacies. Use only supplied metrics, dates, offers, digest items, relationship facts, and preferences. Unknown merchant-trigger kinds are considered only when a small set of safe facts is present; otherwise Vera stays silent.
4. Rank eligible candidates deterministically using urgency together with concrete evidence, timing, category fit, merchant state, active offers, and customer relationship/preferences. Stable trigger IDs break ties. A tick sends at most one message per merchant and no more than 20 total.
5. Suppress reused suppression keys and identical content per recipient. Replies recognize commitment, questions, later requests, opt-outs, auto-replies, and out-of-scope requests. An opt-out blocks future sends to that contact for the current process session.

## API

- `GET /v1/healthz` — liveness and context counts.
- `GET /v1/metadata` — bot identity and implementation approach.
- `POST /v1/context` — versioned context ingestion; 500 KB maximum.
- `POST /v1/tick` — select and return up to 20 proactive actions.
- `POST /v1/reply` — return `send`, `wait`, or `end` for a conversation turn.
- `POST /v1/teardown` — clear in-memory test state.

All state is process-local and disappears on restart or scale-to-zero. This follows the challenge's in-memory allowance while the process remains alive; it is not durable production storage. For the Cloud Run challenge deployment, a single instance reduces cross-instance state divergence, but does not protect state from instance restarts. Use shared storage for durable production use.

## Run and test

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
uvicorn bot:app --host 0.0.0.0 --port 8080
```

To use the supplied LLM-scored judge simulator, set the endpoint and scoring credential outside the repository. `TEST_SCENARIO` accepts `all`, `phase2_short`, `full_evaluation`, and the replay scenarios declared in `judge_simulator.py`.

```powershell
$env:BOT_URL = "http://localhost:8080"
$env:LLM_API_KEY = "your-judge-provider-key"
$env:TEST_SCENARIO = "phase2_short"
python judge_simulator.py
$env:TEST_SCENARIO = "full_evaluation"
python judge_simulator.py
```

The API itself needs no LLM key. The simulator's scores require its separate scoring-provider credential; do not commit that key.

## Docker

```powershell
docker build -t vera-bot .
docker run --rm -p 8080:8080 -e PORT=8080 vera-bot
```

The container binds Uvicorn to `0.0.0.0` and reads Cloud Run's `PORT` (default `8080`). Configure `/v1/healthz` as the platform health check. No bot-side secrets or external services are required.

## Limits

Language handling is deliberately restrained: the templates use a small Hindi-English CTA substitution when the pushed profile indicates Hindi/`hi-en`; they do not translate arbitrary text. A reply can use only the current in-memory trigger and context facts. This implementation does not send WhatsApp messages; returned actions are proposals for the judge. For production use, add durable shared state, a reviewed consent/channel policy, localization review, and deployment monitoring.
