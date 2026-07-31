from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.flow.types import NodeContext, NodeInput, NodeOutput


class NodeABC(ABC):
    name: str

    @abstractmethod
    def run(self, input: NodeInput, context: NodeContext) -> NodeOutput:
        raise NotImplementedError


@dataclass(kw_only=True)
class Flow:
    nodes: dict[str, NodeABC] = field(default_factory=dict)
    entry_node: str | None = None

    def register_node(self, node: NodeABC) -> None:
        self.nodes[node.name] = node

    def get_node(self, node_name: str) -> NodeABC:
        try:
            return self.nodes[node_name]
        except KeyError as exc:
            raise KeyError(f"Unknown node: {node_name}") from exc

    def step(self, input: NodeInput, context: NodeContext) -> NodeOutput:
        node_name = input.state.active_node or self.entry_node
        if node_name is None:
            raise ValueError("Flow entry node is not set and state.active_node is empty.")

        node = self.get_node(node_name)
        output = node.run(input, context)

        next_state = output.updated_state or input.state
        if output.next_node is not None:
            next_state = next_state.model_copy(update={"active_node": output.next_node})

        return output.model_copy(update={"updated_state": next_state})