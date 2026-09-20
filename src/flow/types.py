from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from functools import cached_property
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, Field, SerializeAsAny

type MessageContentT = Annotated[
    str,
    Field(examples=["My name is AI agent."])
]

MessageRoleT = Literal["user", "assistant", "system"]

ConversationStatusT = Literal["idle", "running", "waiting", "completed", "failed"]

StepStatusT = Literal["finished", "failed"]

DelegationStatusT = Literal["started", "finished", "failed"]

TNodeInput_t = TypeVar("TNodeInput", bound=BaseModel)

class PrecheckStatus(str, Enum):
    ALLOWED = "allowed"
    NOT_ALLOWED = "not_allowed"
    SMALL_TALK = "small_talk"


class LanguageEnum(str, Enum):
    ENG = "eng"
    PL = "pl"



class BasicMessage(BaseModel):
    message_id: str | None = None
    conversation_id: str
    role: MessageRoleT
    content: MessageContentT
    created_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    conversation_id: str | None = None
    status: ConversationStatusT = "idle"
    active_node: str | None = None
    summary: str | None = None


class FlowInput(BaseModel):
    conversation_id: str | None
    current_message: BasicMessage
    message_history: list[BasicMessage] = Field(default_factory=list)
    state: ConversationState
    request_id: str | None = None
    trace_id: str | None = None


class NodeOutput(BaseModel):
    response: BasicMessage | None = Field(
        default=None,
        description="The assistant reply this node produced, if it produced one.",
    )
    updated_state: ConversationState | None = None
    fallback_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowResponse(BaseModel):
    result: NodeOutput


class NodeRunResult[TNodeOutput: NodeOutput](BaseModel):
    output: TNodeOutput
    analytics_params: dict[str, Any] = Field(default_factory=dict)


class Delegation(BaseModel):
    """One hand-off from a node to a specialist agent."""

    type: Literal["delegation"] = "delegation"
    step: str | None = Field(
        default=None, description="Name of the node that made the delegation."
    )
    id: str
    agent: str
    status: DelegationStatusT
    task: str | None = None


class FlowStepResult(BaseModel):
    node_name: str
    status: StepStatusT
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    output: SerializeAsAny[NodeOutput]
    # analytics_params: debug/monitoring dict with reasoning, input, output summaries, latency, cost
    analytics_params: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None
    error: str | None = None
    delegations: list[Delegation] = Field(
        default_factory=list,
        description="Specialist hand-offs made during this step, in start order, "
        "each in its final status.",
    )


class StepStarted(BaseModel):
    type: Literal["step.started"] = "step.started"
    step: str


class StepFinished(BaseModel):
    """Wire-safe view of a finished step; the full result rides along unserialised."""

    type: Literal["step.finished"] = "step.finished"
    step: str
    status: StepStatusT
    duration_ms: float
    result: FlowStepResult = Field(exclude=True)

    @classmethod
    def of(cls, result: FlowStepResult) -> StepFinished:
        return cls(
            step=result.node_name,
            status=result.status,
            duration_ms=result.duration_ms,
            result=result,
        )


# Everything a running flow reports before its terminal FlowRunResult.
type FlowEvent = StepStarted | StepFinished | Delegation


class FlowRunResult(BaseModel):
    result: SerializeAsAny[NodeOutput]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: float | None = None
    total_cost: float | None = None
    # analytics: summary of all steps - reasoning, input, output, latency, cost per node
    analytics: dict[str, Any] = Field(
        default_factory=dict,
        description="Global flow analytics: step summaries, reasoning, latency, costs"
    )


class NodeABC[TNodeInput: TNodeInput_t, TNodeOutput: NodeOutput](ABC):
    name: str

    @cached_property
    def cache_key(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}:{self.name}"

    async def run(
        self, input: TNodeInput, state: ConversationState
    ) -> NodeRunResult[TNodeOutput]:
        return await self.run_node(input, state)

    @abstractmethod
    async def run_node(
        self, input: TNodeInput, state: ConversationState
    ) -> NodeRunResult[TNodeOutput]:
        raise NotImplementedError
    