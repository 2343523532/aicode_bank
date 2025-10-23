from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# httpx compatibility shim
# ---------------------------------------------------------------------------
_OriginalAsyncClient = httpx.AsyncClient


class _SandboxAsyncClient(_OriginalAsyncClient):
    """Accept the legacy ``app`` parameter removed in httpx>=0.27."""

    def __init__(self, *args, app=None, transport=None, **kwargs):
        if app is not None and transport is None:
            transport = httpx.ASGITransport(app=app)
        super().__init__(*args, transport=transport, **kwargs)


httpx.AsyncClient = _SandboxAsyncClient
