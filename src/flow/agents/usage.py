from __future__ import annotations

import logging
from typing import Any

from genai_prices import calc_price
from pydantic_ai.agent import AgentRunResult

logger = logging.getLogger(__name__)


def run_cost(*runs: AgentRunResult[Any]) -> float:
    """USD price of one or more finished agent runs.

    pydantic-ai reports token counts, not money, so the price table comes from
    ``genai_prices`` (a dependency it already ships). A run nobody can price —
    an unknown model, a test model — counts as 0.0 rather than failing the whole
    node over an analytics figure.
    """
    total = 0.0
    for run in runs:
        response = run.response
        try:
            price = calc_price(
                run.usage,
                response.model_name or "",
                provider_id=response.provider_name,
            )
        except Exception:
            logger.debug(
                "Could not price run model=%s provider=%s",
                response.model_name,
                response.provider_name,
                exc_info=True,
            )
            continue
        total += float(price.total_price)
    return total
