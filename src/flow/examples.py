from __future__ import annotations

from dataclasses import dataclass

from src.config.config import FlowSettings
from src.flow.agents.precheck import PrecheckNode
from src.flow.flow import Flow
from src.flow.types import (
    BasicMessage,
    ConversationState,
    NodeABC,
    NodeContext,
    NodeInput,
    NodeOutput,
    NodeRunResult,
)


@dataclass(kw_only=True)
class ConversationOrchestratorNode(NodeABC[NodeInput, NodeContext, NodeOutput]):
    name: str = "orchestrator_node"

    async def run_node(self, input: NodeInput, context: NodeContext) -> NodeRunResult[NodeOutput]:
        user_text = input.current_message.content.strip().lower()

        if "help" in user_text or user_text.endswith("?"):
            reply_text = "Jasne, mogę pomóc. Powiedz dokładniej, czego potrzebujesz."
        else:
            reply_text = f"Odebrałem wiadomość: {input.current_message.content}"

        assistant_message = BasicMessage(
            conversation_id=input.conversation_id,
            role="assistant",
            content=reply_text,
        )

        return NodeRunResult(
            output=NodeOutput(response=assistant_message, next_node=None),
            analytics_params={"node": self.name, "message_count": 1, "cost": 0.0},
        )


def build_demo_flow(settings: FlowSettings | None = None) -> Flow:
    return Flow(
        precheck=PrecheckNode(),
        orchestrator=ConversationOrchestratorNode(),
        settings=settings or FlowSettings(),
    )


def build_demo_input(conversation_id: str, text: str) -> NodeInput:
    user_message = BasicMessage(
        conversation_id=conversation_id,
        role="user",
        content=text,
    )

    state = ConversationState(conversation_id=conversation_id)

    return NodeInput(
        conversation_id=conversation_id,
        current_message=user_message,
        message_history=[user_message],
        state=state,
    )
