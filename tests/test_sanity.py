import asyncio
import os
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_sanity():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "test_only": True}
