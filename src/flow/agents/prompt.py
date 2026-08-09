from __future__ import annotations

from datetime import UTC, datetime


def current_date() -> str:
    """Today's date, for stamping into agent instructions.
    """
    return datetime.now(UTC).strftime("%Y-%m-%d (%A), UTC")
