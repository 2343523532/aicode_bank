import pytest
from httpx import AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_matrix_spa_page_and_search() -> None:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        page = await ac.get("/matrix")
        assert page.status_code == 200
        assert "ACTIVE TRACE & WEB-MATRIX v5.0" in page.text

        search = await ac.get("/matrix/api/search", params={"q": "covert"})
        assert search.status_code == 200
        payload = search.json()
        assert any(row["group"] == "stealth" for row in payload)


@pytest.mark.asyncio
async def test_trace_lockout_flow() -> None:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        for _ in range(3):
            bad_attempt = await ac.post(
                "/matrix/api/bank/attempt",
                json={
                    "user": "bad",
                    "account_id": "fed_reserve_001",
                    "amount": 10,
                    "signature": "invalid",
                    "passphrase": "nope",
                    "required_concept": "stealth",
                },
            )
            assert bad_attempt.status_code == 200

        status = await ac.get("/matrix/api/status")
        assert status.status_code == 200
        assert status.json()["locked"] is True
        assert status.json()["trace_level"] == 100


@pytest.mark.asyncio
async def test_lisp_config_endpoint() -> None:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        res = await ac.get("/matrix/api/config/lisp")
        assert res.status_code == 200
        payload = res.json()
        assert payload["language"] == "common-lisp"
        assert "*dan-omniscient-infrastructure*" in payload["config"]
        assert "1000000000000.00" in payload["config"]
        assert payload["test_only"] is True
