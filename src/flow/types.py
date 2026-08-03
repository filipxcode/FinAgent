from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

type MessageContentT = Annotated[
    str,
    Field(examples=["My name is AI agent."])
]

MessageRoleT = Literal["user", "assistant", "system"]

ConversationStatusT = Literal["idle", "running", "waiting", "completed", "failed"]


class BasicMessage(BaseModel):
    message_id: str | None = None
    conversation_id: str
    role: MessageRoleT
    content: MessageContentT
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    conversation_id: str
    status: ConversationStatusT = "idle"
    active_node: str | None = None
    summary: str | None = None
    flags: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeInput(BaseModel):
    conversation_id: str
    current_message: BasicMessage
    message_history: list[BasicMessage] = Field(default_factory=list)
    state: ConversationState
    request_id: str | None = None
    trace_id: str | None = None
    


class NodeContext(BaseModel):
    conversation_id: str
    request_id: str | None = None
    trace_id: str | None = None
    logger: Any | None = None
    service: Any | None = None
    session: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunContext(BaseModel):
    conversation_id: str
    request_id: str | None = None
    trace_id: str | None = None
    logger: Any | None = None
    service: Any | None = None
    session: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class NodeOutput(BaseModel):
    output_messages: list[BasicMessage] = Field(default_factory=list)
    next_node: str | None = None
    updated_state: ConversationState | None = None
    should_persist: bool = True
    is_terminal: bool = False
    fallback_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowRequest(BaseModel):
    input: NodeInput
    context: NodeContext


class FlowResponse(BaseModel):
    result: NodeOutput


class NodeRunResult(BaseModel):
    output: NodeOutput
    analytics_params: dict[str, Any] = Field(default_factory=dict)


class FlowStepResult(BaseModel):
    node_name: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    output: NodeOutput
    analytics_params: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None


class FlowRunResult(BaseModel):
    result: NodeOutput
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    total_cost: float | None = None
    steps: list[FlowStepResult] = Field(default_factory=list)


class NodeABC(ABC):
    name: str

    @cached_property
    def cache_key(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}:{self.name}"

    def run(self, input: NodeInput, context: NodeContext) -> NodeRunResult:
        return self.run_node(input, context)

    @abstractmethod
    def run_node(self, input: NodeInput, context: NodeContext) -> NodeRunResult:
        raise NotImplementedError
    