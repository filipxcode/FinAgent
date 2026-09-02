from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.flow.agents.precheck import PrecheckNode
from src.flow.flow import Flow
from src.flow.types import (
    BasicMessage,
    ConversationState,
    FlowInput,
    NodeABC,
    NodeOutput,
    NodeRunResult,
)


class DemoOrchestratorInput(BaseModel):
    current_message: BasicMessage
    message_history: list[BasicMessage] = Field(default_factory=list)


@dataclass(kw_only=True)
class ConversationOrchestratorNode(NodeABC[DemoOrchestratorInput, NodeOutput]):
    name: str = "orchestrator_node"

    async def run_node(
        self, input: DemoOrchestratorInput, state: ConversationState
    ) -> NodeRunResult[NodeOutput]:
        _ = state
        user_text = input.current_message.content.strip().lower()

        if "help" in user_text or user_text.endswith("?"):
            reply_text = "Jasne, mogę pomóc. Powiedz dokładniej, czego potrzebujesz."
        else:
            reply_text = f"Odebrałem wiadomość: {input.current_message.content}"

        assistant_message = BasicMessage(
            conversation_id=input.current_message.conversation_id,
            role="assistant",
            content=reply_text,
        )

        return NodeRunResult(
            output=NodeOutput(response=assistant_message),
            analytics_params={"node": self.name, "message_count": 1, "cost": 0.0},
        )


def build_demo_flow() -> Flow:
    return Flow(
        precheck=PrecheckNode(),
        orchestrator=ConversationOrchestratorNode(),
    )


def build_demo_input(conversation_id: str, text: str) -> FlowInput:
    user_message = BasicMessage(
        conversation_id=conversation_id,
        role="user",
        content=text,
    )

    return FlowInput(
        current_message=user_message,
        message_history=[user_message],
        state=ConversationState(conversation_id=conversation_id),
    )
