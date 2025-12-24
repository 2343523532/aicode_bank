import os
from decimal import Decimal

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@db:5432/sandboxdb",
)
SANDBOX_API_KEY = os.environ.get("SANDBOX_API_KEY", "change_this_api_key")
SANDBOX_HMAC_SECRET = os.environ.get(
    "SANDBOX_HMAC_SECRET", "change_this_hmac_secret"
)
REQUIRE_HMAC = os.environ.get("REQUIRE_HMAC", "false").lower() == "true"
DEFAULT_TRILLION_BALANCE_USD = Decimal(
    os.environ.get("SANDBOX_DEFAULT_TRILLION_BALANCE_USD", "100000000000000")
)  # 100T USD
WEBHOOK_RETRY_MAX = int(os.environ.get("WEBHOOK_RETRY_MAX", "6"))
