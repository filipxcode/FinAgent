from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from pydantic import BaseModel

from src.db.postgres import ConversationService
from src.flow.flow import Flow
from src.flow.types import (
    BasicMessage,
    ConversationState,
    ConversationStatusT,
    FlowEvent,
    FlowInput,
    FlowRunResult,
)

logger = logging.getLogger(__name__)

_HISTORY_LIMIT = 20
_FALLBACK_REPLY = "Sorry, something went wrong processing your request."
_EMPTY_REPLY = "No response"


class ConversationResult(BaseModel):
    """Terminal event of one turn - the single shape both endpoints map from."""

    conversation_id: str
    content: str
    status: ConversationStatusT
    run: FlowRunResult | None = None


async def run_conversation(
    *,
    flow: Flow,
    service: ConversationService,
    conversation_id: str,
    text: str,
) -> AsyncIterator[FlowEvent | ConversationResult]:
    """Run one conversation turn: persist the user message, stream the flow,
    persist the reply.
    """
    logger.info("Conversation request for conversation_id=%s", conversation_id)
    await service.save_message(
        conversation_id=conversation_id, role="user", content=text
    )

    reply: str | None = None
    status: ConversationStatusT = "failed"
    run: FlowRunResult | None = None
    try:
        raw_history = await service.get_history(
            conversation_id=conversation_id, limit=_HISTORY_LIMIT
        )
        message_history = [
            BasicMessage(conversation_id=conversation_id, **m)
            for m in raw_history or []
        ]
        input = FlowInput(
            conversation_id=conversation_id,
            current_message=BasicMessage(
                conversation_id=conversation_id, role="user", content=text
            ),
            message_history=message_history,
            state=ConversationState(conversation_id=conversation_id),
        )

        async for item in flow.stream(input):
            if isinstance(item, FlowRunResult):
                run = item
            else:
                yield item

        reply = (
            run.result.response.content
            if run and run.result.response
            else _EMPTY_REPLY
        )
        status = "completed"
    except Exception:
        logger.exception(
            "Conversation turn failed conversation_id=%s", conversation_id
        )
        reply = _FALLBACK_REPLY
    finally:
        if reply is not None:
            # The reply already exists; failing to store it must not cut the stream.
            try:
                await service.save_message(
                    conversation_id=conversation_id, role="assistant", content=reply
                )
            except Exception:
                logger.exception(
                    "Could not save the reply conversation_id=%s", conversation_id
                )

    yield ConversationResult(
        conversation_id=conversation_id,
        content=reply or _FALLBACK_REPLY,
        status=status,
        run=run,
    )
