import logging
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4
from collections.abc import AsyncIterable

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from src.api.deps import get_auth_token, get_postgres_client, get_service
from src.api.logger import configure_logging
from src.api.types import (
    ConversationIdFieldT,
    ConversationRequestInput,
    ConversationRequestOutput,
    HistoryResponse,
)
from src.config.config import Settings
from src.db.postgres import ConversationService
from src.flow.types import BasicMessage
from src.service.conversation import ConversationResult, run_conversation

settings = Settings()
configure_logging(settings.logging_settings)

logger = logging.getLogger("finagent.api")

from src.flow.flow import Flow

flow = Flow()

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    logger.info("API startup complete")
    yield
    db_client = get_postgres_client()
    if db_client.is_connected:
        await db_client.close()
    logger.info("API shutdown complete")


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/conversation")
@app.post("/conversation/{conversation_id}", response_model=ConversationRequestOutput)
@limiter.limit(settings.flow_settings.conversation_rate_limit)
async def conversation(
    request: Request,
    payload: ConversationRequestInput,
    auth_token: Annotated[str, Depends(get_auth_token)],
    service: Annotated[ConversationService, Depends(get_service)],
    conversation_id: ConversationIdFieldT | None = None,
):
    _ = auth_token
    result: ConversationResult | None = None
    async for item in run_conversation(
        flow=flow,
        service=service,
        conversation_id=conversation_id or str(uuid4()),
        text=payload.conversation,
    ):
        if isinstance(item, ConversationResult):
            result = item
    if result is None:
        raise RuntimeError("run_conversation did not yield a ConversationResult")
    return ConversationRequestOutput(
        conversation=result.content,
        conversation_id=result.conversation_id,
        status=result.status,
        sources=result.run.sources if result.run else [],
    )


@app.post("/stream/conversation", response_class=EventSourceResponse)
@app.post("/stream/conversation/{conversation_id}", response_class=EventSourceResponse)
@limiter.limit(settings.flow_settings.conversation_rate_limit)
async def conversation_stream(
    request: Request,
    payload: ConversationRequestInput,
    auth_token: Annotated[str, Depends(get_auth_token)],
    service: Annotated[ConversationService, Depends(get_service)],
    conversation_id: ConversationIdFieldT | None = None,
) -> AsyncIterable[ServerSentEvent]:
    """Same turn as /conversation, reported step by step over SSE.
    """
    _ = auth_token
    async for item in run_conversation(
        flow=flow,
        service=service,
        conversation_id=conversation_id or str(uuid4()),
        text=payload.conversation,
    ):
        if isinstance(item, ConversationResult):
            yield ServerSentEvent(
                event="done" if item.status == "completed" else "error",
                data=ConversationRequestOutput(
                    conversation=item.content,
                    conversation_id=item.conversation_id,
                    status=item.status,
                    sources=item.run.sources if item.run else [],
                ),
            )
        else:
            yield ServerSentEvent(event=item.type, data=item.model_dump())


@app.get("/history", response_model=HistoryResponse)
async def history(
    auth_token: Annotated[str, Depends(get_auth_token)],
    service: Annotated[ConversationService, Depends(get_service)],
    conversation_id: ConversationIdFieldT,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    before: Annotated[
        datetime | None,
        Query(description="Fetch messages older than this timestamp."),
    ] = None,
):
    message_history = await service.get_history(
        conversation_id=conversation_id, limit=limit, before=before
    )
    messages = [
        BasicMessage(conversation_id=conversation_id, **m) for m in message_history
    ]
    next_cursor = messages[0].created_at if len(messages) == limit else None
    return HistoryResponse(messages=messages, next_cursor=next_cursor)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.logging_settings.level.lower(),
    )
