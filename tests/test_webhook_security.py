import asyncio
import pytest
from httpx import AsyncClient
from app.main import app
from app.config import SANDBOX_HMAC_SECRET
from app.utils import deliver_webhook_with_retries
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_webhook_timestamp():
    # We want to intercept httpx.AsyncClient.post and check headers
    with patch("httpx.AsyncClient.post") as mock_post:
        # Mock response to be successful so retry doesn't loop
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        url = "http://example.com/webhook"
        payload = {"foo": "bar"}
        secret = SANDBOX_HMAC_SECRET

        success = await deliver_webhook_with_retries(url, payload, secret)

        assert success is True
        assert mock_post.called

        # Check args
        args, kwargs = mock_post.call_args
        headers = kwargs["headers"]

        assert "X-Signature-Timestamp" in headers
        assert "X-Signature" in headers

        timestamp = headers["X-Signature-Timestamp"]
        signature = headers["X-Signature"]

        # Verify signature manually
        import hmac
        import hashlib
        import json

        body = json.dumps(payload)
        expected_payload = f"{timestamp}.{body}"
        expected_signature = hmac.new(secret.encode("utf-8"), expected_payload.encode("utf-8"), hashlib.sha256).hexdigest()

        assert signature == expected_signature
