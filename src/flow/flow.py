from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import sleep

from src.config.config import FlowSettings
from src.flow.types import (
    FlowRunResult,
    FlowStepResult,
    NodeABC,
    NodeContext,
    NodeInput,
    NodeOutput,
)


@dataclass(kw_only=True)
class Flow:
    nodes: dict[str, NodeABC] = field(default_factory=dict)
    entry_node: str | None = None
    settings: FlowSettings = field(default_factory=FlowSettings)

    def register_node(self, node: NodeABC) -> None:
        self.nodes[node.name] = node

    def get_node(self, node_name: str) -> NodeABC:
        try:
            return self.nodes[node_name]
        except KeyError as exc:
            raise KeyError(f"Unknown node: {node_name}") from exc

    def run(self, input: NodeInput, context: NodeContext) -> FlowStepResult:
        node_name = input.state.active_node or self.entry_node
        if node_name is None:
            raise ValueError("Flow entry node is not set and state.active_node is empty.")

        started_at = datetime.now(timezone.utc)
        node = self.get_node(node_name)
        node_result = node.run(input, context)
        finished_at = datetime.now(timezone.utc)

        analytics_params = node_result.analytics_params
        cost = analytics_params.get("cost")
        if cost is not None:
            cost = float(cost)

        output = node_result.output
        next_state = output.updated_state or input.state
        if output.next_node is not None:
            next_state = next_state.model_copy(update={"active_node": output.next_node})

        updated_output = output.model_copy(update={"updated_state": next_state})
        duration_ms = (finished_at - started_at).total_seconds() * 1000

        return FlowStepResult(
            node_name=node_name,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            output=updated_output,
            analytics_params=analytics_params,
            cost=cost,
        )

    def _run_with_retry(self, input: NodeInput, context: NodeContext) -> FlowStepResult:
        last_error: Exception | None = None

        for attempt in range(1, self.settings.retry_max_attempts + 1):
            try:
                return self.run(input, context)
            except Exception as exc:
                last_error = exc
                if attempt >= self.settings.retry_max_attempts:
                    break
                if self.settings.retry_delay_seconds > 0:
                    sleep(self.settings.retry_delay_seconds)

        assert last_error is not None
        raise last_error

    def run_flow(
        self,
        input: NodeInput,
        context: NodeContext,
        *,
        max_steps: int | None = None,
    ) -> FlowRunResult:
        current_input = input
        steps: list[FlowStepResult] = []
        started_at = datetime.now(timezone.utc)
        step_limit = max_steps if max_steps is not None else self.settings.max_steps

        for _ in range(step_limit):
            step_result = self._run_with_retry(current_input, context)
            steps.append(step_result)

            output = step_result.output
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
                finished_at = datetime.now(timezone.utc)
                duration_ms = (finished_at - started_at).total_seconds() * 1000
                total_cost = sum(
                    step.cost for step in steps if step.cost is not None
                ) or None
                return FlowRunResult(
                    result=output,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    total_cost=total_cost,
                    steps=steps,
                )

        raise RuntimeError(
            f"Flow exceeded max_steps={step_limit}. Check node transitions for loops."
        )