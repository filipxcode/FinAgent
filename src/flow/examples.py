from __future__ import annotations

from dataclasses import dataclass

from src.flow.flow import Flow
from src.config.config import FlowSettings
from src.flow.types import (
    BasicMessage,
    ConversationState,
    NodeContext,
    NodeInput,
    NodeOutput,
    NodeRunResult,
    NodeABC,
)


@dataclass(kw_only=True)
class ConversationOrchestratorNode(NodeABC):
    name: str = "conversation_orchestrator"

    def run_node(self, input: NodeInput, context: NodeContext) -> NodeRunResult:
        logger = context.logger
        if logger is not None:
            logger.info("Running node=%s conversation_id=%s", self.name, input.conversation_id)

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

        next_state = input.state.model_copy(
            update={
                "status": "completed",
                "active_node": None,
            }
        )

        return NodeRunResult(
            output=NodeOutput(
                output_messages=[assistant_message],
                next_node=None,
                updated_state=next_state,
                should_persist=True,
                is_terminal=True,
            ),
            analytics_params={
                "node": self.name,
                "message_count": 1,
                "cost": 0.0,
            },
        )


def build_demo_flow(settings: FlowSettings | None = None) -> Flow:
    flow = Flow(entry_node="conversation_orchestrator", settings=settings or FlowSettings())
    flow.register_node(ConversationOrchestratorNode())
    return flow


def build_demo_input(conversation_id: str, text: str) -> NodeInput:
    user_message = BasicMessage(
        conversation_id=conversation_id,
        role="user",
        content=text,
    )

    state = ConversationState(conversation_id=conversation_id, status="running", active_node="conversation_orchestrator")

    return NodeInput(
        conversation_id=conversation_id,
        current_message=user_message,
        message_history=[user_message],
        state=state,
    )