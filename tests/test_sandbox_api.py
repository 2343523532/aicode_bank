import asyncio
import importlib
import os
from typing import AsyncIterator, Dict

import httpx
import pytest
from asgi_lifespan import LifespanManager
from sqlmodel import SQLModel


@pytest.fixture(scope="module")
def event_loop() -> AsyncIterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def sandbox_app(tmp_path_factory) -> AsyncIterator[object]:
    db_dir = tmp_path_factory.mktemp("db")
    db_path = db_dir / "sandbox.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"

    import app.main as main

    main = importlib.reload(main)
    async with main.async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)

    yield main


@pytest.fixture
async def client(sandbox_app) -> AsyncIterator[httpx.AsyncClient]:
    async with LifespanManager(sandbox_app.app):
        async with httpx.AsyncClient(app=sandbox_app.app, base_url="http://test") as test_client:
            yield test_client


@pytest.mark.asyncio
async def test_charge_and_refund_flow(client: httpx.AsyncClient, sandbox_app) -> None:
    headers: Dict[str, str] = {
        "X-Test-Mode": "1",
        "X-Api-Key": sandbox_app.SANDBOX_API_KEY,
    }

    customer_payload = {
        "name": "Test Customer",
        "email": "customer@example.com",
        "initial_balance_usd": "1000.00",
    }
    create_customer = await client.post("/v1/customers", json=customer_payload, headers=headers)
    assert create_customer.status_code == 200
    customer_id = create_customer.json()["id"]

    balance_response = await client.get(f"/v1/accounts/{customer_id}/balance", headers=headers)
    assert balance_response.status_code == 200
    assert balance_response.json()["balance"] == "1,000.00 USD"

    charge_payload = {
        "customer_id": customer_id,
        "amount_cents": 5000,
        "currency": "USD",
        "description": "Integration test charge",
    }
    create_charge = await client.post("/v1/charges", json=charge_payload, headers=headers)
    assert create_charge.status_code == 200
    charge_data = create_charge.json()
    assert charge_data["status"] == "succeeded"
    assert charge_data["amount_cents"] == 5000
    charge_id = charge_data["id"]

    list_charges = await client.get(f"/v1/customers/{customer_id}/charges", headers=headers)
    assert list_charges.status_code == 200
    charges = list_charges.json()
    assert charges["count"] == 1
    assert charges["charges"][0]["id"] == charge_id

    refund_payload = {"amount_cents": 2000, "reason": "customer_request"}
    partial_refund = await client.post(
        f"/v1/charges/{charge_id}/refunds", json=refund_payload, headers=headers
    )
    assert partial_refund.status_code == 200
    refund_data = partial_refund.json()
    assert refund_data["amount_cents"] == 2000
    assert refund_data["reason"] == "customer_request"

    charges_after_partial = await client.get(f"/v1/customers/{customer_id}/charges", headers=headers)
    assert charges_after_partial.status_code == 200
    charge_summary = charges_after_partial.json()["charges"][0]
    assert charge_summary["status"] == "partially_refunded"
    assert charge_summary["refunded_cents"] == 2000

    final_refund = await client.post(f"/v1/charges/{charge_id}/refunds", headers=headers)
    assert final_refund.status_code == 200
    final_refund_data = final_refund.json()
    assert final_refund_data["amount_cents"] == 3000

    charges_after_full = await client.get(f"/v1/customers/{customer_id}/charges", headers=headers)
    assert charges_after_full.status_code == 200
    charge_summary = charges_after_full.json()["charges"][0]
    assert charge_summary["status"] == "refunded"
    assert charge_summary["refunded_cents"] == 5000

    balance_after = await client.get(f"/v1/accounts/{customer_id}/balance", headers=headers)
    assert balance_after.status_code == 200
    assert balance_after.json()["balance"] == "1,000.00 USD"

    events = await client.get(f"/v1/accounts/{customer_id}/events", headers=headers)
    assert events.status_code == 200
    event_types = [event["type"] for event in events.json()["events"]]
    assert event_types.count("charge") == 1
    assert event_types.count("refund") == 2
