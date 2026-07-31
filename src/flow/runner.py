from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar

from src.flow.flow import Flow
from src.flow.types import NodeContext, NodeInput, NodeOutput, RunContext

OutputT = TypeVar("OutputT")


@dataclass(kw_only=True)
class RetryPolicy:
    max_attempts: int = 3
    delay_seconds: float = 0.0


@dataclass(kw_only=True)
class FlowRunner:
    flow: Flow
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    max_steps: int = 32

    def run(self, input: NodeInput, context: RunContext) -> NodeOutput:
        current_input = input
        current_context = NodeContext.model_validate(context.model_dump())

        for _ in range(self.max_steps):
            output = self._run_step(current_input, current_context)

            current_state = output.updated_state or current_input.state
            next_node = output.next_node

            if output.output_messages:
                current_message = output.output_messages[-1]
                message_history = [
                    *current_input.message_history,
                    *output.output_messages,
                ]
            else:
                current_message = current_input.current_message
                message_history = current_input.message_history

            current_input = current_input.model_copy(
                update={
                    "current_message": current_message,
                    "message_history": message_history,
                    "state": current_state,
                }
            )

            if output.is_terminal or next_node is None:
                return output

        raise RuntimeError(
            f"Flow exceeded max_steps={self.max_steps}. Check node transitions for loops."
        )

    def _run_step(self, input: NodeInput, context: NodeContext) -> NodeOutput:
        last_error: Exception | None = None

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                return self.flow.step(input, context)
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_policy.max_attempts:
                    break

        assert last_error is not None
        raise last_error