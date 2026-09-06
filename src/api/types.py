from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, Field

from src.flow.types import BasicMessage, ConversationStatusT

type ConversationRequestInputT = Annotated[
    str, Field(examples=["Hello, what is your name?"])
]

type ConversationRequestOutputT = Annotated[
    str, Field(examples=["Hello! I am FinAgent."])
]

type ConversationIdFieldT = Annotated[
    str, Field(examples=["4f32c027-3df7-44bc-8774-7d52fbb48cd0"])
]


class ConversationRequestInput(BaseModel):
    conversation: ConversationRequestInputT


class ConversationRequestOutput(BaseModel):
    conversation_id: ConversationIdFieldT
    conversation: ConversationRequestOutputT
    status: ConversationStatusT
    debug: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Temporary debug field: raw node output (reasoning, sub-agent "
            "traces, etc). Drop before shipping to real clients."
        ),
    )


class HistoryResponse(BaseModel):
    messages: list[BasicMessage] = Field(default_factory=list, description="Oldest first.")
    next_cursor: datetime | None = Field(
        default=None,
        description=(
            "Pass as `before` to fetch the page right before this one. "
            "None means there is no older history left."
        ),
    )
