from __future__ import annotations

import logging
from asyncio import sleep
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from src.config.config import FlowSettings
from src.flow.agents.orchestrator import OrchestratorNode
from src.flow.agents.precheck import PrecheckNode, PrecheckNodeOutput
from src.flow.types import (
    FlowRunResult,
    FlowStepResult,
    NodeABC,
    NodeContext,
    NodeInput,
    NodeOutput,
    PrecheckStatus,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Flow:
    settings: FlowSettings = field(default_factory=FlowSettings)
    precheck: PrecheckNode = field(default_factory=PrecheckNode)
    orchestrator: OrchestratorNode = field(default_factory=OrchestratorNode)

    async def run_flow(
        self,
        input: NodeInput,
        context: NodeContext,
    ) -> FlowRunResult:
        """Run the agents in order, branching on what precheck decides.

        precheck -> small_talk / not_allowed  => stop here
                 -> allowed                    => orchestrator
        """
        steps: list[FlowStepResult] = []
        started_at = datetime.now(UTC)
        current = input
        
        step, current = await self._step(self.precheck, current, context, steps)
        if step.status == "failed":
            return self._build_result(steps, started_at, "failed", input)

        precheck_output = cast(PrecheckNodeOutput, step.output)
        if precheck_output.status in (PrecheckStatus.SMALL_TALK, PrecheckStatus.NOT_ALLOWED):
            return self._build_result(steps, started_at, "completed", input)

        step, current = await self._step(self.orchestrator, current, context, steps)
        if step.status == "failed":
            return self._build_result(steps, started_at, "failed", input)

        return self._build_result(steps, started_at, "completed", input)

    async def _step(
        self,
        node: NodeABC[Any, Any, Any],
        input: NodeInput,
        context: NodeContext,
        steps: list[FlowStepResult],
    ) -> tuple[FlowStepResult, NodeInput]:
        """Run one node, record its step, and thread the state forward."""
        step = await self._run_with_retry(node, input, context)
        steps.append(step)
        next_input = input.model_copy(update={"state": step.output.updated_state})
        return step, next_input

    def _build_result(
        self,
        steps: list[FlowStepResult],
        started_at: datetime,
        final_status: str,
        input: NodeInput,
    ) -> FlowRunResult:
        finished_at = datetime.now(UTC)
        duration_ms = (finished_at - started_at).total_seconds() * 1000
        total_cost = sum(s.cost for s in steps if s.cost is not None) or None

        last_state = steps[-1].output.updated_state if steps else input.state
        final_state = (last_state or input.state).model_copy(
            update={"status": final_status, "active_node": None}
        )
        result = (
            steps[-1].output.model_copy(update={"updated_state": final_state})
            if steps
            else NodeOutput(updated_state=final_state)
        )

        return FlowRunResult(
            result=result,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            total_cost=total_cost,
            steps=steps,
        )

    async def _run_with_retry(
        self,
        node: NodeABC[Any, Any, Any],
        input: NodeInput,
        context: NodeContext,
    ) -> FlowStepResult:
        """Run a node with retries and record its lifecycle (running -> finished/failed)."""
        run_logger = context.logger or logger
        last_error: Exception | None = None

        running_state = input.state.model_copy(
            update={"status": "running", "active_node": node.name}
        )
        step_input = input.model_copy(update={"state": running_state})

        for attempt in range(1, self.settings.retry_max_attempts + 1):
            started_at = datetime.now(UTC)
            try:
                node_result = await node.run(step_input, context)
                finished_at = datetime.now(UTC)

                cost = node_result.analytics_params.get("cost")
                cost = float(cost) if cost is not None else None

                output = node_result.output
                # Node may have changed business fields; keep the flow-owned status.
                business_state = output.updated_state or running_state
                next_state = business_state.model_copy(
                    update={"status": "running", "active_node": node.name}
                )

                return FlowStepResult(
                    node_name=node.name,
                    status="finished",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=(finished_at - started_at).total_seconds() * 1000,
                    output=output.model_copy(update={"updated_state": next_state}),
                    analytics_params=node_result.analytics_params,
                    cost=cost,
                )
            except Exception as exc:
                last_error = exc
                run_logger.exception(
                    "Node run failed node=%s attempt=%s/%s conversation_id=%s",
                    node.name,
                    attempt,
                    self.settings.retry_max_attempts,
                    step_input.conversation_id,
                )
                if attempt < self.settings.retry_max_attempts:
                    if self.settings.retry_delay_seconds > 0:
                        await sleep(self.settings.retry_delay_seconds)
                    continue

                finished_at = datetime.now(UTC)
                failed_state = running_state.model_copy(
                    update={"status": "failed", "active_node": None}
                )
                return FlowStepResult(
                    node_name=node.name,
                    status="failed",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=(finished_at - started_at).total_seconds() * 1000,
                    output=NodeOutput(
                        updated_state=failed_state,
                        next_node=None,
                        fallback_reason=str(exc),
                    ),
                    analytics_params={"node": node.name, "cost": 0.0},
                    error=str(exc),
                )

        assert last_error is not None
        raise last_error
