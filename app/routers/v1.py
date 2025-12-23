from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Body, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.database import get_session
from app.dependencies import verify_request_safety
from app.models import Customer, Ledger, Token, Charge, Refund, IdempotencyKey
from app.schemas import (
    TokenCreate, TokenResponse,
    CustomerCreate, CustomerResponse,
    ChargeCreate, ChargeResponse, ChargesListResponse,
    RefundCreate, RefundResponse
)
from app.utils import (
    make_id, money_to_cents, cents_to_str, total_refunded_cents,
    deliver_webhook_with_retries
)
from app.config import DEFAULT_TRILLION_BALANCE_USD

router = APIRouter(prefix="/v1")

@router.post("/tokens", response_model=TokenResponse)
async def create_token(
    request: Request,
    payload: TokenCreate,
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


@router.post("/customers", response_model=CustomerResponse)
async def create_customer(
    request: Request,
    payload: CustomerCreate,
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


@router.get("/accounts/{account_id}/balance")
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


@router.post("/charges", response_model=ChargeResponse)
async def create_charge(
    request: Request,
    background: BackgroundTasks,
    payload: ChargeCreate,
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
        metadata_json={"description": payload.description} if payload.description else {},
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
        "refunded_cents": 0,
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
    webhook_url = (
        customer.metadata_json.get("webhook_url")
        if customer and customer.metadata_json
        else None
    )
    if webhook_url:
        background.add_task(
            deliver_webhook_with_retries,
            webhook_url,
            {"type": "charge.succeeded", "data": response_payload},
            secret="webhook_secret_dummy",
        )

    return ChargeResponse(**response_payload)


@router.get("/customers/{customer_id}/charges", response_model=ChargesListResponse)
async def list_charges(
    customer_id: str,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    limit: int = 10,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> ChargesListResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    # Added pagination logic here
    result = await session.exec(
        select(Charge)
        .where(Charge.customer_id == customer_id)
        .order_by(Charge.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    charges = result.all()

    # To get total count (optional, but good for pagination response)
    # For now, just returning the count of retrieved items or maybe I should do a separate count query.
    # The original code returned len(responses), I'll stick to that but for the full list.
    # Actually, efficient pagination usually requires a separate count query or returning has_more.
    # But for now, let's just paginate the results.

    responses: List[ChargeResponse] = []
    for charge in charges:
        refunded = await total_refunded_cents(session, charge.id)
        responses.append(
            ChargeResponse(
                id=charge.id,
                amount_cents=charge.amount_cents,
                currency=charge.currency,
                customer_id=charge.customer_id,
                status=charge.status,
                refunded_cents=refunded,
            )
        )

    return ChargesListResponse(customer_id=customer_id, charges=responses, count=len(responses))


@router.get("/accounts/{account_id}/events")
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


@router.post("/charges/{charge_id}/refunds", response_model=RefundResponse)
async def create_refund(
    charge_id: str,
    request: Request,
    background: BackgroundTasks,
    payload: RefundCreate = Body(default_factory=RefundCreate),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> RefundResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    charge = await session.get(Charge, charge_id)
    if not charge:
        raise HTTPException(status_code=404, detail="charge not found (sandbox)")

    ledger = await session.get(Ledger, charge.customer_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="customer ledger not found (sandbox)")

    if idempotency_key:
        stored = await session.get(IdempotencyKey, idempotency_key)
        if stored:
            return RefundResponse(**stored.response_payload)

    already_refunded = await total_refunded_cents(session, charge_id)
    remaining = charge.amount_cents - already_refunded
    if remaining <= 0:
        raise HTTPException(status_code=400, detail="charge already fully refunded")

    refund_amount = payload.amount_cents or remaining
    if refund_amount <= 0 or refund_amount > remaining:
        raise HTTPException(status_code=400, detail="invalid refund amount")

    refund_id = make_id("re_test")
    metadata_payload = {"reason": payload.reason} if payload.reason else {}
    refund = Refund(
        id=refund_id,
        charge_id=charge_id,
        amount_cents=refund_amount,
        currency=charge.currency,
        metadata_json=metadata_payload,
    )
    session.add(refund)

    ledger.balance_cents += refund_amount
    ledger.events.append(
        {
            "type": "refund",
            "id": refund_id,
            "charge_id": charge_id,
            "amount_cents": refund_amount,
            "note": payload.reason,
        }
    )

    total_after_refund = already_refunded + refund_amount
    if total_after_refund >= charge.amount_cents:
        charge.status = "refunded"
    elif total_after_refund > 0:
        charge.status = "partially_refunded"

    response_payload = RefundResponse(
        id=refund.id,
        charge_id=refund.charge_id,
        amount_cents=refund.amount_cents,
        currency=refund.currency,
        status=refund.status,
        reason=payload.reason,
    )

    if idempotency_key:
        idempotency = IdempotencyKey(key=idempotency_key, response_payload=response_payload.dict())
        session.add(idempotency)

    await session.commit()
    await session.refresh(refund)

    customer = await session.get(Customer, charge.customer_id)
    webhook_url = (
        customer.metadata_json.get("webhook_url")
        if customer and customer.metadata_json
        else None
    )
    if webhook_url:
        charge_payload = ChargeResponse(
            id=charge.id,
            amount_cents=charge.amount_cents,
            currency=charge.currency,
            customer_id=charge.customer_id,
            status=charge.status,
            refunded_cents=total_after_refund,
        ).dict()
        background.add_task(
            deliver_webhook_with_retries,
            webhook_url,
            {
                "type": "charge.refunded",
                "data": {"refund": response_payload.dict(), "charge": charge_payload},
            },
            secret="webhook_secret_dummy",
        )

    return response_payload
