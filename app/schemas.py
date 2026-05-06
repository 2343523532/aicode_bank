from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TokenCreate(BaseModel):
    card_number: Optional[str] = None
    exp_month: Optional[int] = Field(default=None, ge=1, le=12)
    exp_year: Optional[int] = Field(default=None, ge=2000, le=9999)
    cvc: Optional[str] = Field(default=None, min_length=3, max_length=4)


class TokenResponse(BaseModel):
    id: str
    object: str = "token"
    card_last4: Optional[str]
    test_only: bool = True


class CustomerCreate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    initial_balance_usd: Optional[Decimal] = Field(default=None, ge=Decimal("0"))
    webhook_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CustomerResponse(BaseModel):
    id: str
    object: str = "customer"
    name: Optional[str]
    email: Optional[str]
    created_at: Optional[int] = None
    test_only: bool = True


class ChargeCreate(BaseModel):
    customer_id: str
    amount_cents: int = Field(gt=0)
    currency: Optional[str] = "USD"
    description: Optional[str] = Field(default=None, max_length=500)
    source_token: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> str:
        normalized = (value or "USD").upper()
        if len(normalized) != 3 or not normalized.isalpha():
            raise ValueError("currency must be a three-letter ISO-style code")
        return normalized


class ChargeResponse(BaseModel):
    id: str
    object: str = "charge"
    amount_cents: int
    currency: str
    customer_id: str
    status: str
    refunded_cents: int = 0
    created_at: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    test_only: bool = True


class ChargesListResponse(BaseModel):
    customer_id: str
    charges: List[ChargeResponse]
    count: int
    total_count: int = 0
    limit: int = 10
    offset: int = 0
    has_more: bool = False
    test_only: bool = True


class RefundCreate(BaseModel):
    amount_cents: Optional[int] = Field(default=None, gt=0)
    reason: Optional[str] = Field(default=None, max_length=250)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RefundResponse(BaseModel):
    id: str
    object: str = "refund"
    charge_id: str
    amount_cents: int
    currency: str
    status: str
    created_at: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    test_only: bool = True
    reason: Optional[str] = None


class RefundsListResponse(BaseModel):
    charge_id: str
    refunds: List[RefundResponse]
    count: int
    total_count: int = 0
    limit: int = 10
    offset: int = 0
    has_more: bool = False
    test_only: bool = True


class AccountSummaryResponse(BaseModel):
    test_only: bool = True
    account_id: str
    currency: str
    balance_cents: str
    balance: str
    charges_count: int
    refunded_cents: int
    net_spend_cents: int
    event_count: int
    note: str
