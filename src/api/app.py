from contextlib import asynccontextmanager
from typing import Annotated
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI

from src.api.logger import configure_logger
from src.api.deps import get_auth_token, get_postgres_client, get_service
from src.api.types import (
    ConversationIdFieldT,
    ConversationRequestInput,
    ConversationRequestOutput,
)
from src.db.postgres import ConversationService
from src.config.config import Settings
from src.flow.types import FlowInput, ConversationState, BasicMessage

logger = configure_logger("finagent.api")

settings = Settings()
state = ConversationState()
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


@app.post("/conversation")
@app.post("/conversation/{conversation_id}")
async def conversation(
    request: ConversationRequestInput,
    #auth_token: Annotated[str, Depends(get_auth_token)],
    service: Annotated[ConversationService, Depends(get_service)],
    conversation_id: ConversationIdFieldT | None = None,
) -> ConversationRequestOutput:
    #_ = auth_token
    if conversation_id is None:
        conversation_id = str(uuid4())

    logger.info("Conversation request for conversation_id=%s", conversation_id)
    await service.save_message(
        conversation_id=conversation_id,
        role="user",
        content=request.conversation,
    )
    state.conversation_id = conversation_id
    try:
        message_history = await service.get_history(
            conversation_id=conversation_id, limit=20
        )
        message_history = (
            [BasicMessage(conversation_id=conversation_id, **m) for m in message_history]
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
            response.result.response.content if response.result.response else "No response"
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")