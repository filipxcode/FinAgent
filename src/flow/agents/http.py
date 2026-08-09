from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


async def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> tuple[Any | None, str | None]:
    """GET a JSON endpoint and return ``(payload, error)``.

    Shared by every agent tool that talks to an external HTTP API. Transport and
    HTTP failures never propagate — they come back as a readable ``error``
    string so a tool can degrade instead of failing the whole flow run. Exactly
    one of the two slots is ever populated.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json(), None
    except Exception as exc:
        logger.exception("HTTP GET failed url=%s params=%s", url, params)
        return None, f"{httpx.URL(url).host} request error: {exc}"
