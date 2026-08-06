"""Canonical six-agent training entry.

The implementation remains in the legacy module so existing long-running jobs
and saved launch commands continue to work.
"""

from __future__ import annotations

from .six_agent_private_train import *  # noqa: F401,F403
from .six_agent_private_train import main


if __name__ == "__main__":
    main()
