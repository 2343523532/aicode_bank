import hashlib
import hmac
import json
import logging
import uuid
import asyncio
import httpx
import time
from decimal import Decimal
from typing import Any, Dict
from sqlmodel import select, func
from sqlmodel.ext.asyncio.session import AsyncSession
from app.config import SANDBOX_HMAC_SECRET, WEBHOOK_RETRY_MAX
from app.models import Refund

logger = logging.getLogger("sandbox")

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

async def total_refunded_cents(session: AsyncSession, charge_id: str) -> int:
    """Return the total amount refunded for a charge in cents."""
    query = select(func.coalesce(func.sum(Refund.amount_cents), 0)).where(
        Refund.charge_id == charge_id
    )
    result = await session.exec(query)
    value = result.one()
    return int(value[0] if isinstance(value, (tuple, list)) else value)

async def deliver_webhook_with_retries(url: str, payload: Dict[str, Any], secret: str) -> bool:
    body = json.dumps(payload)

    # Implement timestamp signature for security (Next Step requirement)
    timestamp = str(int(time.time()))
    payload_to_sign = f"{timestamp}.{body}"
    signature = hmac.new(secret.encode("utf-8"), payload_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Signature": signature,
        "X-Signature-Timestamp": timestamp
    }

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
