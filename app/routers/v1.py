from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import JSONResponse
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config import DEFAULT_TRILLION_BALANCE_USD
from app.database import get_session
from app.dependencies import verify_request_safety
from app.models import Charge, Customer, IdempotencyKey, Ledger, Refund, Token
from app.schemas import (
    AccountSummaryResponse,
    ChargeCreate,
    ChargeResponse,
    ChargesListResponse,
    CustomerCreate,
    CustomerResponse,
    RefundCreate,
    RefundResponse,
    RefundsListResponse,
    TokenCreate,
    TokenResponse,
)
from app.utils import (
    cents_to_str,
    deliver_webhook_with_retries,
    make_id,
    money_to_cents,
    total_refunded_cents,
)

router = APIRouter(prefix="/v1")


def _idempotency_storage_key(operation: str, key: str) -> str:
    """Scope idempotency keys per operation to avoid cross-endpoint collisions."""
    return f"{operation}:{key}"


def _model_dump(model: Any) -> Dict[str, Any]:
    """Return a Pydantic model dictionary across supported Pydantic versions."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _charge_metadata(payload: ChargeCreate) -> Dict[str, Any]:
    metadata = dict(payload.metadata)
    if payload.description:
        metadata["description"] = payload.description
    if payload.source_token:
        metadata["source_token"] = payload.source_token
    return metadata


def _refund_metadata(payload: RefundCreate) -> Dict[str, Any]:
    metadata = dict(payload.metadata)
    if payload.reason:
        metadata["reason"] = payload.reason
    return metadata


async def _count_charges(session: AsyncSession, customer_id: str) -> int:
    result = await session.exec(
        select(func.count()).select_from(Charge).where(Charge.customer_id == customer_id)
    )
    return int(result.one())


async def _count_refunds(session: AsyncSession, charge_id: str) -> int:
    result = await session.exec(
        select(func.count()).select_from(Refund).where(Refund.charge_id == charge_id)
    )
    return int(result.one())


async def _charge_response(session: AsyncSession, charge: Charge) -> ChargeResponse:
    return ChargeResponse(
        id=charge.id,
        amount_cents=charge.amount_cents,
        currency=charge.currency,
        customer_id=charge.customer_id,
        status=charge.status,
        refunded_cents=await total_refunded_cents(session, charge.id),
        created_at=charge.created_at,
        metadata=charge.metadata_json or {},
    )


def _refund_response(refund: Refund) -> RefundResponse:
    metadata = refund.metadata_json or {}
    return RefundResponse(
        id=refund.id,
        charge_id=refund.charge_id,
        amount_cents=refund.amount_cents,
        currency=refund.currency,
        status=refund.status,
        created_at=refund.created_at,
        metadata=metadata,
        reason=metadata.get("reason"),
    )


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
    metadata = dict(payload.metadata)
    if payload.webhook_url:
        metadata["webhook_url"] = payload.webhook_url
    customer = Customer(
        id=customer_id,
        name=payload.name,
        email=payload.email,
        metadata_json=metadata,
    )
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
        created_at=customer.created_at,
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


@router.get("/accounts/{account_id}/summary", response_model=AccountSummaryResponse)
async def account_summary(
    account_id: str,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> AccountSummaryResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    ledger = await session.get(Ledger, account_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="account not found (sandbox)")

    charges_count = await _count_charges(session, account_id)
    refunded_result = await session.exec(
        select(func.coalesce(func.sum(Refund.amount_cents), 0))
        .join(Charge, Refund.charge_id == Charge.id)
        .where(Charge.customer_id == account_id)
    )
    refunded_cents = int(refunded_result.one() or 0)
    charged_result = await session.exec(
        select(func.coalesce(func.sum(Charge.amount_cents), 0)).where(
            Charge.customer_id == account_id
        )
    )
    charged_cents = int(charged_result.one() or 0)

    return AccountSummaryResponse(
        account_id=ledger.account_id,
        currency=ledger.currency,
        balance_cents=str(ledger.balance_cents),
        balance=cents_to_str(ledger.balance_cents, ledger.currency),
        charges_count=charges_count,
        refunded_cents=refunded_cents,
        net_spend_cents=charged_cents - refunded_cents,
        event_count=len(ledger.events or []),
        note="TEST_ONLY analytics summary — no real funds move.",
    )


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

    storage_key = (
        _idempotency_storage_key("charge.create", idempotency_key)
        if idempotency_key
        else None
    )
    if storage_key:
        stored = await session.get(IdempotencyKey, storage_key)
        if stored:
            return JSONResponse(status_code=200, content=stored.response_payload)

    charge_id = make_id("ch_test")
    charge = Charge(
        id=charge_id,
        customer_id=payload.customer_id,
        amount_cents=payload.amount_cents,
        currency=payload.currency or "USD",
        status="succeeded",
        idempotency_key=storage_key,
        metadata_json=_charge_metadata(payload),
    )
    session.add(charge)

    ledger.balance_cents -= payload.amount_cents
    ledger.events.append(
        {
            "type": "charge",
            "id": charge_id,
            "amount_cents": -payload.amount_cents,
            "currency": charge.currency,
            "note": payload.description,
        }
    )

    response = await _charge_response(session, charge)
    response_payload = _model_dump(response)

    if storage_key:
        idempotency = IdempotencyKey(key=storage_key, response_payload=response_payload)
        session.add(idempotency)

    await session.commit()
    await session.refresh(charge)
    response = await _charge_response(session, charge)
    response_payload = _model_dump(response)

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

    return response


@router.get("/charges/{charge_id}", response_model=ChargeResponse)
async def get_charge(
    charge_id: str,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    session: AsyncSession = Depends(get_session),
) -> ChargeResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    charge = await session.get(Charge, charge_id)
    if not charge:
        raise HTTPException(status_code=404, detail="charge not found (sandbox)")
    return await _charge_response(session, charge)


@router.get("/customers/{customer_id}/charges", response_model=ChargesListResponse)
async def list_charges(
    customer_id: str,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
) -> ChargesListResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    customer = await session.get(Customer, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="customer not found (sandbox)")

    base_query = select(Charge).where(Charge.customer_id == customer_id)
    count_query = (
        select(func.count()).select_from(Charge).where(Charge.customer_id == customer_id)
    )
    if status:
        base_query = base_query.where(Charge.status == status)
        count_query = count_query.where(Charge.status == status)

    total_result = await session.exec(count_query)
    total_count = int(total_result.one())

    result = await session.exec(
        base_query.order_by(Charge.created_at.desc(), Charge.id.desc())
        .offset(offset)
        .limit(limit)
    )
    charges = result.all()

    responses: List[ChargeResponse] = []
    for charge in charges:
        responses.append(await _charge_response(session, charge))

    return ChargesListResponse(
        customer_id=customer_id,
        charges=responses,
        count=len(responses),
        total_count=total_count,
        limit=limit,
        offset=offset,
        has_more=offset + len(responses) < total_count,
    )


@router.get("/accounts/{account_id}/events")
async def account_events(
    account_id: str,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    event_type: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
) -> Dict[str, Any]:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    ledger = await session.get(Ledger, account_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="account not found (sandbox)")

    events = list(ledger.events or [])
    if event_type:
        events = [event for event in events if event.get("type") == event_type]
    total_count = len(events)
    page = events[offset : offset + limit]
    return {
        "test_only": True,
        "account_id": account_id,
        "events": page,
        "count": len(page),
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(page) < total_count,
    }


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

    storage_key = (
        _idempotency_storage_key("refund.create", idempotency_key)
        if idempotency_key
        else None
    )
    if storage_key:
        stored = await session.get(IdempotencyKey, storage_key)
        if stored:
            return RefundResponse(**stored.response_payload)

    already_refunded = await total_refunded_cents(session, charge_id)
    remaining = charge.amount_cents - already_refunded
    if remaining <= 0:
        raise HTTPException(status_code=400, detail="charge already fully refunded")

    refund_amount = payload.amount_cents or remaining
    if refund_amount > remaining:
        raise HTTPException(status_code=400, detail="invalid refund amount")

    refund_id = make_id("re_test")
    refund = Refund(
        id=refund_id,
        charge_id=charge_id,
        amount_cents=refund_amount,
        currency=charge.currency,
        metadata_json=_refund_metadata(payload),
    )
    session.add(refund)

    ledger.balance_cents += refund_amount
    ledger.events.append(
        {
            "type": "refund",
            "id": refund_id,
            "charge_id": charge_id,
            "amount_cents": refund_amount,
            "currency": charge.currency,
            "note": payload.reason,
        }
    )

    total_after_refund = already_refunded + refund_amount
    if total_after_refund >= charge.amount_cents:
        charge.status = "refunded"
    elif total_after_refund > 0:
        charge.status = "partially_refunded"

    response_payload = _model_dump(_refund_response(refund))

    if storage_key:
        idempotency = IdempotencyKey(key=storage_key, response_payload=response_payload)
        session.add(idempotency)

    await session.commit()
    await session.refresh(refund)
    response = _refund_response(refund)

    customer = await session.get(Customer, charge.customer_id)
    webhook_url = (
        customer.metadata_json.get("webhook_url")
        if customer and customer.metadata_json
        else None
    )
    if webhook_url:
        charge_payload = _model_dump(await _charge_response(session, charge))
        background.add_task(
            deliver_webhook_with_retries,
            webhook_url,
            {
                "type": "charge.refunded",
                "data": {"refund": _model_dump(response), "charge": charge_payload},
            },
            secret="webhook_secret_dummy",
        )

    return response


@router.get("/charges/{charge_id}/refunds", response_model=RefundsListResponse)
async def list_refunds(
    charge_id: str,
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> RefundsListResponse:
    await verify_request_safety(request, x_test_mode, x_api_key, x_signature)

    charge = await session.get(Charge, charge_id)
    if not charge:
        raise HTTPException(status_code=404, detail="charge not found (sandbox)")

    total_count = await _count_refunds(session, charge_id)
    result = await session.exec(
        select(Refund)
        .where(Refund.charge_id == charge_id)
        .order_by(Refund.created_at.desc(), Refund.id.desc())
        .offset(offset)
        .limit(limit)
    )
    refunds = result.all()
    responses = [_refund_response(refund) for refund in refunds]

    return RefundsListResponse(
        charge_id=charge_id,
        refunds=responses,
        count=len(responses),
        total_count=total_count,
        limit=limit,
        offset=offset,
        has_more=offset + len(responses) < total_count,
    )
