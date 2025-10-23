"""Legacy shim for the sandbox API.

This module simply re-exports everything from :mod:`app.main` so that
historical imports like ``import code`` continue to resolve without
clashing with Python's standard :mod:`code` module.
"""

from app.main import *  # noqa: F401,F403
