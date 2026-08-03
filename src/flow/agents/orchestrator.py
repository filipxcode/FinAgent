from dataclasses import dataclass

from src.config.config import get_settings
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


@dataclass
class OrchestratorDeps:
    pass

ORCHESTRATOR_AGENT_KEY = "orchestrator"
agent = get_settings().get_agent(ORCHESTRATOR_AGENT_KEY)


@dataclass(kw_only=True)
class OrchestratorNode(NodeABC):
    name: str = "orchestrator_node"

    def run_node(self, input: NodeInput, context: NodeContext) -> NodeRunResult:
        return NodeRunResult(
            output=NodeOutput(
                output_messages=[],
                next_node=None,
                updated_state=input.state,
                should_persist=False,
                is_terminal=True,
            ),
            analytics_params={"node": self.name, "cost": 0.0},
        )
