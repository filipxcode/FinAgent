from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI

from src.api.logger import configure_logger
from src.api.deps import get_auth_token, get_postgres_client, get_service
from src.api.types import (
    ConversationIdFieldT,
    ConversationRequestInput,
    ConversationRequestOutput,
)
from src.db.postgres import ConversationService

logger = configure_logger("finagent.api")


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


@app.post("/conversation")
@app.post("/conversation/{conversation_id}")
async def conversation(
    request: ConversationRequestInput,
    auth_token: Annotated[str, Depends(get_auth_token)],
    service: Annotated[ConversationService, Depends(get_service)],
    conversation_id: ConversationIdFieldT | None = None,
) -> ConversationRequestOutput:
    _ = auth_token
    if conversation_id is None:
        conversation_id = str(uuid4())

    logger.info("Conversation request for conversation_id=%s", conversation_id)
    await service.save_message(
        conversation_id=conversation_id,
        role="user",
        content=request.conversation,
    )
    
    return ConversationRequestOutput(conversation=request.conversation)