import hmac
from typing import Optional
from fastapi import Header, HTTPException, Request, status
from app.config import SANDBOX_API_KEY, REQUIRE_HMAC
from app.utils import compute_hmac_hex

async def verify_request_safety(
    request: Request,
    x_test_mode: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    x_signature: Optional[str] = Header(None),
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
