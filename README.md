# AI Code Bank Sandbox API

A fully asynchronous FastAPI application that emulates a Stripe-style sandbox for integration tests. The service persists data with SQLModel and offers deterministic identifiers, ledger tracking, idempotency support, and background webhook retries. **All endpoints are for test use only and never move real money.**

## Features

- Token, customer, balance, charge, refund, event, and account summary endpoints with strict sandbox headers.
- Persistent ledger that records every transaction event, including refunds.
- Idempotency key storage to guarantee repeatable responses.
- Optional webhook delivery with exponential backoff and HMAC signatures.
- **Webhook Security**: Webhook signatures now include a timestamp to prevent replay attacks (`X-Signature-Timestamp`).
- **Pagination**: Charge, refund, and ledger event listings support `limit`, `offset`, totals, and `has_more` metadata.
- Refund workflow with partial and full refunds, refund listing, charge detail lookup, and account summaries.
- SQLite-compatible JSON columns, making local testing fast and dependency-free.
- Automated regression test that exercises the full charge/refund lifecycle.

## Requirements

The application targets Python 3.11. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the API locally

1. Export environment overrides if needed (optional):
   ```bash
   export SANDBOX_API_KEY="your_test_key"
   export SANDBOX_HMAC_SECRET="super_secret"
   export DATABASE_URL="sqlite+aiosqlite:///./sandbox.db"
   ```
2. Start the server:
   ```bash
   uvicorn app.main:app --reload --port 8080
   ```
3. Send requests with the required headers:
   - `X-Test-Mode: 1`
   - `X-Api-Key: <your SANDBOX_API_KEY>`
   - `X-Signature: <HMAC signature>` (only when `REQUIRE_HMAC=true`)

## Key endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/tokens` | Create a disposable token. |
| `POST` | `/v1/customers` | Register a customer and provision a seeded ledger. |
| `GET` | `/v1/accounts/{account_id}/balance` | Retrieve ledger balance details. |
| `POST` | `/v1/charges` | Create a charge and reduce the customer's balance. |
| `GET` | `/v1/customers/{customer_id}/charges` | List charges with refund totals, status filtering, `limit`/`offset`, totals, and `has_more`. |
| `GET` | `/v1/charges/{charge_id}` | Retrieve a single charge with metadata and refund totals. |
| `POST` | `/v1/charges/{charge_id}/refunds` | Issue partial or full refunds with idempotency support. |
| `GET` | `/v1/charges/{charge_id}/refunds` | List refunds for a charge with pagination metadata. |
| `GET` | `/v1/accounts/{account_id}/summary` | Retrieve charge count, refunded cents, net spend, balance, and event count. |
| `GET` | `/v1/accounts/{account_id}/events` | Inspect chronological ledger events with optional `event_type`, `limit`, and `offset`. |
| `GET` | `/health` | Lightweight liveness probe. |
| `GET` | `/matrix/api/config/lisp` | Return DAN Omni Common Lisp infrastructure configuration. |
| `GET` | `/matrix/api/config/sentient-bank-lisp` | Return the Sentient Mega-Bank & Crypto AGI Simulator Common Lisp script. |

Each response contains `"test_only": true` to emphasise that no real funds move.

## Testing

Run the asynchronous pytest suite:

```bash
pytest
```

## Deployment notes

- The included `Dockerfile` installs dependencies via `requirements.txt` and exposes the app on port `8080`.
- For PostgreSQL deployments, set `DATABASE_URL` accordingly.
- Use `docker-compose up` to start the API and a PostgreSQL database.

## Next steps

- Introduce API key management endpoints to rotate sandbox credentials programmatically.
- Add a webhook event delivery audit table so tests can inspect retry attempts.


## Sentient Mega-Bank simulator payload

This app now includes the full **Sentient Mega-Bank & Crypto AGI Simulator** Common Lisp payload and serves it via:

```
GET /matrix/api/config/sentient-bank-lisp
```

The response contains:
- `name`: `sentient-mega-bank-crypto-agi-simulator`
- `language`: `common-lisp`
- `config`: the complete Lisp source provided
- `test_only`: `true`
