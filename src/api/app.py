import logging
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from datetime import datetime

from fastapi import Depends, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

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
from src.flow.types import BasicMessage, ConversationState, FlowInput

settings = Settings()
configure_logging(settings.logging_settings)

logger = logging.getLogger("finagent.api")

from src.flow.flow import Flow

flow = Flow()


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
async def conversation(
    request: ConversationRequestInput,
    auth_token: Annotated[str, Depends(get_auth_token)],
    service: Annotated[ConversationService, Depends(get_service)],
    conversation_id: ConversationIdFieldT | None = None,
):
    _ = auth_token
    if conversation_id is None:
        conversation_id = str(uuid4())

    logger.info("Conversation request for conversation_id=%s", conversation_id)
    await service.save_message(
        conversation_id=conversation_id,
        role="user",
        content=request.conversation,
    )
    state = ConversationState(conversation_id=conversation_id)
    try:
        message_history = await service.get_history(
            conversation_id=conversation_id, limit=20
        )
        message_history = (
            [
                BasicMessage(conversation_id=conversation_id, **m)
                for m in message_history
            ]
            if message_history
            else []
        )
        current_message = BasicMessage(
            conversation_id=conversation_id,
            role="user",
            content=request.conversation,
        )

        input = FlowInput(
            conversation_id=conversation_id,
            current_message=current_message,
            message_history=message_history,
            state=state,
        )
        response = await flow.run(input=input)
        reply_content = (
            response.result.response.content
            if response.result.response
            else "No response"
        )
        await service.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=reply_content,
        )
        return ConversationRequestOutput(
            conversation=reply_content,
            conversation_id=conversation_id,
            status="completed",
        )
    except Exception as e:
        logger.error("Error during running a flow %e", e)
        return ConversationRequestOutput(
            conversation="Sorry, something went wrong processing your request.",
            conversation_id=conversation_id,
            status="failed",
        )


@app.get("/history", response_model=HistoryResponse)
async def history(
    # auth_token: Annotated[str, Depends(get_auth_token)],
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
