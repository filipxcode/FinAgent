from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.flow.flow import Flow, NodeABC
from src.flow.types import (
    BasicMessage,
    ConversationState,
    NodeContext,
    NodeInput,
    NodeOutput,
)

@dataclass
class OrchestratorDeps:
    

@dataclass(kw_only=True)
class OrchestratorNode(NodeABC):
    name: str = "orchestrator_node"
    
    def run(self, input: NodeInput, context: NodeContext):
        