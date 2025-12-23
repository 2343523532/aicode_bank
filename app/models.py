from typing import Any, Dict, List, Optional
from sqlalchemy import BigInteger, Column
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel
import time

class Customer(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[str] = Field(default=None, primary_key=True)
    name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    metadata_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", MutableDict.as_mutable(JSON)),
    )
    created_at: int = Field(default_factory=lambda: int(time.time()))


class Ledger(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    account_id: str = Field(primary_key=True)
    currency: str = Field(default="USD")
    balance_cents: int = Field(default=0, sa_column=Column(BigInteger))
    events: List[Dict[str, Any]] = Field(
        default_factory=list, sa_column=Column(MutableList.as_mutable(JSON))
    )


class Token(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: str = Field(primary_key=True)
    card_last4: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    metadata_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", MutableDict.as_mutable(JSON)),
    )


class Charge(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: str = Field(primary_key=True)
    customer_id: str = Field(index=True)
    amount_cents: int = Field(sa_column=Column(BigInteger))
    currency: str = Field(default="USD")
    status: str = Field(default="succeeded")
    idempotency_key: Optional[str] = Field(default=None, index=True)
    metadata_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", MutableDict.as_mutable(JSON)),
    )
    created_at: int = Field(default_factory=lambda: int(time.time()))


class Refund(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: str = Field(primary_key=True)
    charge_id: str = Field(index=True)
    amount_cents: int = Field(sa_column=Column(BigInteger))
    currency: str = Field(default="USD")
    status: str = Field(default="succeeded")
    metadata_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column("metadata", MutableDict.as_mutable(JSON)),
    )
    created_at: int = Field(default_factory=lambda: int(time.time()))


class IdempotencyKey(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    key: str = Field(primary_key=True)
    response_payload: Dict[str, Any] = Field(sa_column=Column(JSON))
    created_at: int = Field(default_factory=lambda: int(time.time()))
