from typing import Annotated

from pydantic import BaseModel, Field

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
    conversation: ConversationRequestOutputT
