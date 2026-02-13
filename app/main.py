"""Sandbox Stripe-like API (REALISTIC SANDBOX).

This module implements a FastAPI application that mimics a subset of the
Stripe API for test purposes only.  The API enforces explicit test mode
headers, an API key, and optional HMAC signatures.  All state is persisted in
PostgreSQL using SQLModel's async support and is therefore safe for repeated
integration tests that require idempotent behaviour.

WARNING: TEST-SANDBOX ONLY. DO NOT USE THIS TO DEFRAUD OR IMITATE REAL
PROVIDERS.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, select

from app import database
from app.models import Customer, Ledger
from app.utils import make_id, money_to_cents, cents_to_str
from app.config import DEFAULT_TRILLION_BALANCE_USD, SANDBOX_API_KEY as CONFIG_SANDBOX_API_KEY
from app.routers import v1

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sandbox")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sandbox Stripe-like API (TEST_ONLY)",
    version="1.0.0",
    description=(
        "Realistic sandbox for testing. ALL RESPONSES ARE TEST_ONLY and NOT REAL "
        "MONEY."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1.router)

# Keep compatibility for test modules that access main.async_engine directly.
async_engine = database.get_async_engine()
# Compatibility export for existing tests and integrations.
SANDBOX_API_KEY = CONFIG_SANDBOX_API_KEY


# ---------------------------------------------------------------------------
# Startup: create tables and seed demo customer
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Creating DB tables (if not exist)...")
    async with database.get_async_engine().begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    database.refresh_engine_if_needed()
    async with database.AsyncSessionLocal() as session:
        first = await session.exec(select(Customer))
        first = first.first()
        if not first:
            demo_id = make_id("cus_test")
            demo = Customer(id=demo_id, name="Demo Customer", email="dev@example.com")
            session.add(demo)

            balance_cents = money_to_cents(DEFAULT_TRILLION_BALANCE_USD)
            ledger = Ledger(
                account_id=demo_id,
                currency="USD",
                balance_cents=balance_cents,
                events=[
                    {
                        "type": "initial_seed",
                        "amount_cents": balance_cents,
                        "note": "startup seed (TEST_ONLY)",
                    }
                ],
            )
            session.add(ledger)
            await session.commit()
            logger.info(
                "Seeded demo customer %s with balance %s",
                demo_id,
                cents_to_str(balance_cents),
            )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "test_only": True}
