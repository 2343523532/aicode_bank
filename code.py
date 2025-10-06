# app/main.py
"""
Sandbox Stripe-like API (REALISTIC SANDBOX)
- Persistent Postgres (SQLModel)
- API Key auth + X-Test-Mode required
- HMAC signature optional verification
- Idempotency stored in DB
- Ledger with BIGINT cents (can hold trillions)
- Background webhook delivery with retry
- All responses include "test_only": True

WARNING: TEST-SANDBOX ONLY. DO NOT USE THIS TO DEFRAUD OR IMITATE REAL PROVIDERS.
"""

import os
import hmac
import hashlib
import logging
import asyncio
import uuid
from decimal import Decimal
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, Request, Header, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlmodel import Field as SQLField, SQLModel, create_engine, Session, select, Column, BigInteger, JSON
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession as SAAsyncSession
from sqlalchemy.orm import sessionmaker

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sandbox")

# --- Config from env ---
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://postgres:password@db:5432/sandboxdb")
SANDBOX_API_KEY = os.environ.get("SANDBOX_API_KEY", "change_this_api_key")
SANDBOX_HMAC_SECRET = os.environ.get("SANDBOX_HMAC_SECRET", "change_this_hmac_secret")
REQUIRE_HMAC = os.environ.get("REQUIRE_HMAC", "false").lower() == "true"
DEFAULT_TRILLION_BALANCE_USD = Decimal(os.environ.get("SANDBOX_DEFAULT_TRILLION_BALANCE_USD", "100000000000000"))  # 100T USD
WEBHOOK_RETRY_MAX = int(os.environ.get("WEBHOOK_RETRY_MAX", "6"))

# --- DB setup (async) ---
async_engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(async_engine, expire_on_commit=False, class_=SAAsyncSession)

# --- Models ---
class Customer(SQLModel, table=True):
    id: Optional[str] = SQLField(default=None, primary_key=True)
    name: Optional[str]
    email: Optional[str]
    metadata: Optional[Dict[str, Any]] = SQLField(sa_column=Column(JSON), default={})
    created_at: Optional[int] = SQLField(default_factory=lambda: int(__import__("time").time()))

class Ledger(SQLModel, table=True):
    account_id: str = SQLField(primary_key=True)
    currency: str = SQLField(default="USD")
    balance_cents: int = SQLField(sa_column=Column(BigInteger), default=0)  # BIGINT for trillions
    events: List[Dict[str, Any]] = SQLField(sa_column=Column(JSON), default=[])

class Token(SQLModel, table=True):
    id: str = SQLField(primary_key=True)
    card_last4: Optional[str]
    created_at: Optional[int] = SQLField(default_factory=lambda: int(__import__("time").time()))
    metadata: Optional[Dict[str, Any]] = SQLField(sa_column=Column(JSON), default={})

class Charge(SQLModel, table=True):
    id: str = SQLField(primary_key=True)
    customer_id: str = SQLField(index=True)
    amount_cents: int = SQLField(sa_column=Column(BigInteger))
    currency: str = SQLField(default="USD")
    status: str = SQLField(default="succeeded")
    idempotency_key: Optional[str] = SQLField(index=True, default=None)
    metadata: Optional[Dict[str, Any]] = SQLField(sa_column=Column(JSON), default={})
    created_at: Optional[int] = SQLField(default_factory=lambda: int(__import__("time").time()))

class IdempotencyKey(SQLModel, table=True):
    key: str = SQLField(primary_key=True)
    response_payload: Dict[str, Any] = SQLField(sa_column=Column(JSON))
    created_at: Optional[int] = SQLField(default_factory=lambda: int(__import__("time").time()))

# --- Pydantic request/response models ---
class TokenCreate(BaseModel):
    card_number: Optional[str]
    exp_month: Optional[int]
    exp_year: Optional[int]
    cvc: Optional[str]

class TokenResponse(BaseModel):
    id: str
    object: str = "token"
    card_last4: Optional[str]
    test_only: bool = True

class CustomerCreate(BaseModel):
    name: Optional[str]
    email: Optional[str]
    initial_balance_usd: Optional[Decimal]

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

# --- FastAPI app ---
app = FastAPI(title="Sandbox Stripe-like API (TEST_ONLY)", version="1.0.0",
              description="Realistic sandbox for testing. ALL RESPONSES ARE TEST_ONLY and NOT REAL MONEY.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utilities ---
def compute_hmac_hex(body_bytes: bytes) -> str:
    return hmac.new(SANDBOX_HMAC_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()

async def verify_request_safety(request: Request,
                                x_test_mode: Optional[str],
                                x_api_key: Optional[str],
                                x_signature: Optional[str]):
    # API key required
    if x_api_key != SANDBOX_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    # test mode header required
    if x_test_mode != "1":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="X-Test-Mode header required and must be '1'")
    # optional HMAC verification (recommended)
    if REQUIRE_HMAC:
        raw = await request.body()
        if not x_signature:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Signature (HMAC)")
        expected = compute_hmac_hex(raw)
        if not hmac.compare_digest(expected, x_signature):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid HMAC signature")

def money_to_cents(amount: Decimal) -> int:
    # convert Decimal USD -> cents (int)
    return int((amount * Decimal(100)).to_integral_value())

def cents_to_str(cents: int, currency="USD") -> str:
    dec = Decimal(cents) / Decimal(100)
    return f"{dec:,.2f} {currency}"

def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"

# --- DB dependency ---
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# --- Startup: create tables and seed demo customer ---
@app.on_event("startup")
async def on_startup():
    # Create tables (for demo; for production use migrations)
    logger.info("Creating DB tables (if not exist)...")
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    # seed demo if none
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Customer))
        first = result.first()
        if not first:
            demo_id = make_id("cus_test")
            demo = Customer(id=demo_id, name="Demo Customer", email="dev@example.com")
            session.add(demo)
            # seed ledger with huge default balance (100T)
            balance_cents = money_to_cents(DEFAULT_TRILLION_BALANCE_USD)
            ledger = Ledger(account_id=demo_id, currency="USD", balance_cents=balance_cents, events=[
                {"type": "initial_seed", "amount_cents": balance_cents, "note": "startup seed (TEST_ONLY)"}
            ])
            session.add(ledger)
            await session.commit()
            logger.info(f"Seeded demo customer {demo_id} with balance {cents_to_str(balance_cents)}")

# --- Endpoints ---

@app.post("/v1/tokens", response_model=TokenResponse)
async def create_token(payload: TokenCreate, request: Request,
                       x_test_mode: Optional[str] = Header(None),
                       x_api_key: Optional[str] = Header(None),
                       x_signature: Optional[str] = Header(None),
                       session: AsyncSession = Depends(get_session)):
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    tid = make_id("tok_test")
    last4 = None
    if payload.card_number:
        digits = "".join(ch for ch in payload.card_number if ch.isdigit())
        last4 = digits[-4:] if digits else None

    token = Token(id=tid, card_last4=last4)
    session.add(token)
    await session.commit()
    await session.refresh(token)
    return TokenResponse(id=token.id, card_last4=token.card_last4)

@app.post("/v1/customers", response_model=CustomerResponse)
async def create_customer(payload: CustomerCreate, request: Request,
                          x_test_mode: Optional[str] = Header(None),
                          x_api_key: Optional[str] = Header(None),
                          x_signature: Optional[str] = Header(None),
                          session: AsyncSession = Depends(get_session)):
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    cid = make_id("cus_test")
    customer = Customer(id=cid, name=payload.name, email=payload.email, metadata={})
    session.add(customer)

    # seed ledger
    usd_amount = payload.initial_balance_usd if payload.initial_balance_usd is not None else DEFAULT_TRILLION_BALANCE_USD
    balance_cents = money_to_cents(usd_amount)
    ledger = Ledger(account_id=cid, currency="USD", balance_cents=balance_cents, events=[
        {"type": "initial_seed", "amount_cents": balance_cents, "note": "TEST_ONLY initial seed"}
    ])
    session.add(ledger)
    await session.commit()
    await session.refresh(customer)
    return CustomerResponse(id=customer.id, name=customer.name, email=customer.email)

@app.get("/v1/accounts/{account_id}/balance")
async def get_balance(account_id: str, request: Request,
                      x_test_mode: Optional[str] = Header(None),
                      x_api_key: Optional[str] = Header(None),
                      x_signature: Optional[str] = Header(None),
                      session: AsyncSession = Depends(get_session)):
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    q = await session.get(Ledger, account_id)
    if not q:
        raise HTTPException(status_code=404, detail="account not found (sandbox)")
    # Return numeric cents as string (prevents accidental numeric pipelines) AND a formatted string
    return {
        "test_only": True,
        "account_id": q.account_id,
        "currency": q.currency,
        "balance_cents": str(q.balance_cents),
        "balance": cents_to_str(q.balance_cents, q.currency),
        "note": "TEST_ONLY — not real money; do not use for real payments or representation"
    }

@app.post("/v1/charges", response_model=ChargeResponse)
async def create_charge(payload: ChargeCreate, request: Request,
                        background: BackgroundTasks,
                        idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
                        x_test_mode: Optional[str] = Header(None),
                        x_api_key: Optional[str] = Header(None),
                        x_signature: Optional[str] = Header(None),
                        session: AsyncSession = Depends(get_session)):
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    # Validate customer / ledger
    ledger = await session.get(Ledger, payload.customer_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="customer not found (sandbox)")

    # Idempotency handling: return stored response if key exists
    if idempotency_key:
        stored = await session.get(IdempotencyKey, idempotency_key)
        if stored:
            return JSONResponse(status_code=200, content=stored.response_payload)

    # Create charge record, always "succeed" in sandbox
    cid = make_id("ch_test")
    charge = Charge(id=cid, customer_id=payload.customer_id, amount_cents=payload.amount_cents,
                    currency=payload.currency or "USD", status="succeeded", idempotency_key=idempotency_key,
                    metadata={"description": payload.description})
    session.add(charge)

    # Update ledger (we'll subtract amount for realistic behavior)
    ledger.balance_cents = ledger.balance_cents - payload.amount_cents
    ledger.events.append({"type": "charge", "id": cid, "amount_cents": -payload.amount_cents, "note": payload.description})

    # Build response payload and store idempotency record
    response_payload = {
        "id": charge.id,
        "object": "charge",
        "amount_cents": charge.amount_cents,
        "currency": charge.currency,
        "customer_id": charge.customer_id,
        "status": charge.status,
        "test_only": True
    }
    if idempotency_key:
        ik = IdempotencyKey(key=idempotency_key, response_payload=response_payload)
        session.add(ik)
    await session.commit()

    # Optional: trigger webhooks for this customer (if metadata has 'webhook_url')
    # For demo: if customer.metadata contains webhook_url, enqueue a delivery
    customer = await session.get(Customer, payload.customer_id)
    webhook_url = None
    if customer and customer.metadata:
        webhook_url = customer.metadata.get("webhook_url")
    if webhook_url:
        background.add_task(deliver_webhook_with_retries, webhook_url, {"type": "charge.succeeded", "data": response_payload}, secret="webhook_secret_dummy")

    return ChargeResponse(**response_payload)

# events endpoint for demo (view account events)
@app.get("/v1/accounts/{account_id}/events")
async def account_events(account_id: str, request: Request,
                         x_test_mode: Optional[str] = Header(None),
                         x_api_key: Optional[str] = Header(None),
                         x_signature: Optional[str] = Header(None),
                         session: AsyncSession = Depends(get_session)):
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)
    ledger = await session.get(Ledger, account_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="account not found (sandbox)")
    return {"test_only": True, "account_id": account_id, "events": ledger.events}

# --- Webhook delivery with retries ---
async def deliver_webhook_with_retries(url: str, payload: dict, secret: str):
    body = httpx.dumps(payload) if hasattr(httpx, "dumps") else __import__("json").dumps(payload)
    sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {"Content-Type": "application/json", "X-Signature": sig}
    backoff = 1
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, WEBHOOK_RETRY_MAX + 1):
            try:
                r = await client.post(url, content=body, headers=headers)
                if 200 <= r.status_code < 300:
                    logger.info(f"Webhook delivered to {url}")
                    return True
                else:
                    logger.warning(f"Webhook {url} returned {r.status_code}")
            except Exception as e:
                logger.warning(f"Webhook delivery error: {e}")
            await asyncio.sleep(backoff)
            backoff *= 2
    logger.error(f"Failed to deliver webhook to {url} after retries")
    return False

# Health
@app.get("/health")
async def health():
    return {"ok": True, "test_only": True}
