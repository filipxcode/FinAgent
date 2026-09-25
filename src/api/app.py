import logging
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4
from collections.abc import AsyncIterable

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from src.api.deps import get_auth_token, get_postgres_client, get_service
from src.api.logger import configure_logging
from src.api.types import (
    ConversationIdFieldT,
    ConversationListResponse,
    ConversationRequestInput,
    ConversationRequestOutput,
    ConversationSummary,
    HistoryResponse,
)
from src.config.config import Settings
from src.db.postgres import ConversationService, DatabaseUnavailableError
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


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable(request: Request, exc: DatabaseUnavailableError) -> JSONResponse:
    logger.error("Database unavailable on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=503, content={"detail": "Database unavailable, try again later"}
    )

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

@app.get("/conversations", response_model=ConversationListResponse)
async def conversations(
    auth_token: Annotated[str, Depends(get_auth_token)],
    service: Annotated[ConversationService, Depends(get_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    before: Annotated[
        datetime | None,
        Query(description="Fetch conversations last active before this timestamp."),
    ] = None,
):
    _ = auth_token
    rows = await service.get_conversations(limit=limit, before=before)
    summaries = [ConversationSummary(**row) for row in rows]
    next_cursor = summaries[-1].updated_at if len(summaries) == limit else None
    return ConversationListResponse(conversations=summaries, next_cursor=next_cursor)


@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    auth_token: Annotated[str, Depends(get_auth_token)],
    service: Annotated[ConversationService, Depends(get_service)],
    conversation_id: ConversationIdFieldT,
) -> Response:
    _ = auth_token
    if not await service.delete_conversation(conversation_id=conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.logging_settings.level.lower(),
    )
