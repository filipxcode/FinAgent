from __future__ import annotations

from collections.abc import Iterable

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from src.flow.types import BasicMessage


def to_model_messages(msgs: Iterable[BasicMessage]) -> list[ModelMessage]:
    """Convert stored conversation messages into pydantic-ai history.

    Passed to ``agent.run(..., message_history=...)`` rather than rendered into
    the instructions, so the system prompt stays a stable, cacheable prefix and
    the transport-level fields (ids, timestamps, metadata) never reach the model.
    """
    out: list[ModelMessage] = []
    for m in msgs:
        if m.role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=m.content)]))
        elif m.role == "assistant":
            out.append(ModelResponse(parts=[TextPart(content=m.content)]))
    return out
