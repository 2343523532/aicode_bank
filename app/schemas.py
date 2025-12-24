from typing import List, Optional
from decimal import Decimal
from pydantic import BaseModel

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
    refunded_cents: int = 0
    test_only: bool = True


class ChargesListResponse(BaseModel):
    customer_id: str
    charges: List[ChargeResponse]
    count: int
    test_only: bool = True


class RefundCreate(BaseModel):
    amount_cents: Optional[int] = None
    reason: Optional[str] = None


class RefundResponse(BaseModel):
    id: str
    object: str = "refund"
    charge_id: str
    amount_cents: int
    currency: str
    status: str
    test_only: bool = True
    reason: Optional[str] = None
