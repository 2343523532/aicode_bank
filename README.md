# AI Code Bank Sandbox API

A fully asynchronous FastAPI application that emulates a Stripe-style sandbox for integration tests. The service persists data with SQLModel and offers deterministic identifiers, ledger tracking, idempotency support, and background webhook retries. **All endpoints are for test use only and never move real money.**

## Features

- Token, customer, balance, charge, and event endpoints with strict sandbox headers.
- Persistent ledger that records every transaction event, including refunds.
- Idempotency key storage to guarantee repeatable responses.
- Optional webhook delivery with exponential backoff and HMAC signatures.
- New refund workflow with partial and full refunds plus charge summaries.
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
| `GET` | `/v1/customers/{customer_id}/charges` | List recent charges with refund totals. |
| `POST` | `/v1/charges/{charge_id}/refunds` | Issue partial or full refunds with idempotency support. |
| `GET` | `/v1/accounts/{account_id}/events` | Inspect chronological ledger events. |
| `GET` | `/health` | Lightweight liveness probe. |

Each response contains `"test_only": true` to emphasise that no real funds move.

## Testing

Run the asynchronous pytest suite, which provisions an isolated SQLite database and validates the end-to-end payment flow:

```bash
pytest
```

The test covers customer creation, charging, partial refunds, full refunds, ledger balance recovery, and audit events.

## Deployment notes

- The included Dockerfile (named `code`) installs dependencies via `requirements.txt` and exposes the app on port `8080`.
- For PostgreSQL deployments, set `DATABASE_URL` accordingly; JSON columns automatically adapt to portable SQLAlchemy types.
- Configure webhook endpoints by storing a `webhook_url` value in customer metadata to receive `charge.succeeded` and `charge.refunded` notifications.

## Next steps

- Extend the webhook payloads with signature timestamps for replay protection.
- Add pagination and filtering to the charge listing endpoint.
- Introduce API key management endpoints to rotate sandbox credentials programmatically.
