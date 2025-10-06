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

import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from decimal import Decimal
from typing import Any, Dict, List, Optional

import httpx
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import BigInteger, Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession as SAAsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Field, SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sandbox")


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@db:5432/sandboxdb",
)
SANDBOX_API_KEY = os.environ.get("SANDBOX_API_KEY", "change_this_api_key")
SANDBOX_HMAC_SECRET = os.environ.get(
    "SANDBOX_HMAC_SECRET", "change_this_hmac_secret"
)
REQUIRE_HMAC = os.environ.get("REQUIRE_HMAC", "false").lower() == "true"
DEFAULT_TRILLION_BALANCE_USD = Decimal(
    os.environ.get("SANDBOX_DEFAULT_TRILLION_BALANCE_USD", "100000000000000")
)  # 100T USD
WEBHOOK_RETRY_MAX = int(os.environ.get("WEBHOOK_RETRY_MAX", "6"))


# ---------------------------------------------------------------------------
# Database setup (async)
# ---------------------------------------------------------------------------
async_engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    async_engine, expire_on_commit=False, class_=SAAsyncSession
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Customer(SQLModel, table=True):
    id: Optional[str] = Field(default=None, primary_key=True)
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    created_at: int = Field(default_factory=lambda: int(__import__("time").time()))


class Ledger(SQLModel, table=True):
    account_id: str = Field(primary_key=True)
    currency: str = Field(default="USD")
    balance_cents: int = Field(default=0, sa_column=Column(BigInteger))
    events: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSONB))


class Token(SQLModel, table=True):
    id: str = Field(primary_key=True)
    card_last4: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(__import__("time").time()))
    metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))


class Charge(SQLModel, table=True):
    id: str = Field(primary_key=True)
    customer_id: str = Field(index=True)
    amount_cents: int = Field(sa_column=Column(BigInteger))
    currency: str = Field(default="USD")
    status: str = Field(default="succeeded")
    idempotency_key: Optional[str] = Field(default=None, index=True)
    metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB))
    created_at: int = Field(default_factory=lambda: int(__import__("time").time()))


class IdempotencyKey(SQLModel, table=True):
    key: str = Field(primary_key=True)
    response_payload: Dict[str, Any] = Field(sa_column=Column(JSONB))
    created_at: int = Field(default_factory=lambda: int(__import__("time").time()))


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------
class TokenCreate(BaseModel):
    card_number: Optional[str] = None
    exp_month: Optional[int] = None
    exp_year: Optional[int] = None
    cvc: Optional[str] = None


class TokenResponse(BaseModel):
    id: str
    object: str = "token"
    card_last4: Optional[str]
    test_only: bool = True


class CustomerCreate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    initial_balance_usd: Optional[Decimal] = None


class CustomerResponse(BaseModel):
    id: str
    object: str = "customer"
    name: Optional[str]
    email: Optional[str]
    test_only: bool = True


class ChargeCreate(BaseModel):
    customer_id: str
    amount_cents: int
    currency: Optional[str] = "USD"
    description: Optional[str] = None
    source_token: Optional[str] = None


class ChargeResponse(BaseModel):
    id: str
    object: str = "charge"
    amount_cents: int
    currency: str
    customer_id: str
    status: str
    test_only: bool = True


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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def compute_hmac_hex(body_bytes: bytes) -> str:
    """Compute a SHA256 HMAC for the provided body using the sandbox secret."""
    return hmac.new(SANDBOX_HMAC_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()


def money_to_cents(amount: Decimal) -> int:
    """Convert Decimal USD -> integer cents."""
    quantized = (amount * Decimal(100)).to_integral_value()
    return int(quantized)


def cents_to_str(cents: int, currency: str = "USD") -> str:
    """Format cents for human readability."""
    dec = Decimal(cents) / Decimal(100)
    return f"{dec:,.2f} {currency}"


def make_id(prefix: str) -> str:
    """Create unique identifiers with predictable prefixes."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session  # pragma: no cover - thin wrapper


# ---------------------------------------------------------------------------
# Request verification
# ---------------------------------------------------------------------------
async def verify_request_safety(
    request: Request,
    x_test_mode: Optional[str],
    x_api_key: Optional[str],
    x_signature: Optional[str],
) -> None:
    """Enforce sandbox authentication headers and optional HMAC validation."""

    if x_api_key != SANDBOX_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    if x_test_mode != "1":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Test-Mode header required and must be '1'",
        )

    if REQUIRE_HMAC:
        raw = await request.body()
        if not x_signature:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-Signature (HMAC)",
            )
        expected = compute_hmac_hex(raw)
        if not hmac.compare_digest(expected, x_signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid HMAC signature",
            )


# ---------------------------------------------------------------------------
# Startup: create tables and seed demo customer
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Creating DB tables (if not exist)...")
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Customer))
        first = result.first()
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
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/v1/tokens", response_model=TokenResponse)
async def create_token(
    payload: TokenCreate,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> TokenResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    token_id = make_id("tok_test")
    last4: Optional[str] = None
    if payload.card_number:
        digits = "".join(ch for ch in payload.card_number if ch.isdigit())
        last4 = digits[-4:] if digits else None

    token = Token(id=token_id, card_last4=last4)
    session.add(token)
    await session.commit()
    await session.refresh(token)

    return TokenResponse(id=token.id, card_last4=token.card_last4)


@app.post("/v1/customers", response_model=CustomerResponse)
async def create_customer(
    payload: CustomerCreate,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> CustomerResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    customer_id = make_id("cus_test")
    customer = Customer(id=customer_id, name=payload.name, email=payload.email)
    session.add(customer)

    usd_amount = (
        payload.initial_balance_usd
        if payload.initial_balance_usd is not None
        else DEFAULT_TRILLION_BALANCE_USD
    )
    balance_cents = money_to_cents(usd_amount)
    ledger = Ledger(
        account_id=customer_id,
        currency="USD",
        balance_cents=balance_cents,
        events=[
            {
                "type": "initial_seed",
                "amount_cents": balance_cents,
                "note": "TEST_ONLY initial seed",
            }
        ],
    )
    session.add(ledger)
    await session.commit()
    await session.refresh(customer)

    return CustomerResponse(
        id=customer.id,
        name=customer.name,
        email=customer.email,
    )


@app.get("/v1/accounts/{account_id}/balance")
async def get_balance(
    account_id: str,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    ledger = await session.get(Ledger, account_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="account not found (sandbox)")

    return {
        "test_only": True,
        "account_id": ledger.account_id,
        "currency": ledger.currency,
        "balance_cents": str(ledger.balance_cents),
        "balance": cents_to_str(ledger.balance_cents, ledger.currency),
        "note": "TEST_ONLY — not real money; do not use for real payments or representation",
    }


@app.post("/v1/charges", response_model=ChargeResponse)
async def create_charge(
    payload: ChargeCreate,
    request: Request,
    background: BackgroundTasks,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> ChargeResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    ledger = await session.get(Ledger, payload.customer_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="customer not found (sandbox)")

    if idempotency_key:
        stored = await session.get(IdempotencyKey, idempotency_key)
        if stored:
            return JSONResponse(status_code=200, content=stored.response_payload)

    charge_id = make_id("ch_test")
    charge = Charge(
        id=charge_id,
        customer_id=payload.customer_id,
        amount_cents=payload.amount_cents,
        currency=payload.currency or "USD",
        status="succeeded",
        idempotency_key=idempotency_key,
        metadata={"description": payload.description} if payload.description else {},
    )
    session.add(charge)

    ledger.balance_cents -= payload.amount_cents
    ledger.events.append(
        {
            "type": "charge",
            "id": charge_id,
            "amount_cents": -payload.amount_cents,
            "note": payload.description,
        }
    )

    response_payload: Dict[str, Any] = {
        "id": charge.id,
        "object": "charge",
        "amount_cents": charge.amount_cents,
        "currency": charge.currency,
        "customer_id": charge.customer_id,
        "status": charge.status,
        "test_only": True,
    }

    if idempotency_key:
        idempotency = IdempotencyKey(key=idempotency_key, response_payload=response_payload)
        session.add(idempotency)

    await session.commit()

    if idempotency_key:
        stored = await session.get(IdempotencyKey, idempotency_key)
        if stored:
            await session.refresh(charge)

    customer = await session.get(Customer, payload.customer_id)
    webhook_url = customer.metadata.get("webhook_url") if customer and customer.metadata else None
    if webhook_url:
        background.add_task(
            deliver_webhook_with_retries,
            webhook_url,
            {"type": "charge.succeeded", "data": response_payload},
            secret="webhook_secret_dummy",
        )

    return ChargeResponse(**response_payload)


@app.get("/v1/accounts/{account_id}/events")
async def account_events(
    account_id: str,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    ledger = await session.get(Ledger, account_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="account not found (sandbox)")

    return {"test_only": True, "account_id": account_id, "events": ledger.events}


# ---------------------------------------------------------------------------
# Webhook delivery with retries
# ---------------------------------------------------------------------------
async def deliver_webhook_with_retries(url: str, payload: Dict[str, Any], secret: str) -> bool:
    body = json.dumps(payload)
    signature = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {"Content-Type": "application/json", "X-Signature": signature}

    backoff = 1
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, WEBHOOK_RETRY_MAX + 1):
            try:
                response = await client.post(url, content=body, headers=headers)
                if 200 <= response.status_code < 300:
                    logger.info("Webhook delivered to %s", url)
                    return True
                logger.warning("Webhook %s returned %s", url, response.status_code)
            except Exception as exc:  # pragma: no cover - network interaction
                logger.warning("Webhook delivery error: %s", exc)

            await asyncio.sleep(backoff)
            backoff *= 2

    logger.error("Failed to deliver webhook to %s after retries", url)
    return False


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "test_only": True}
