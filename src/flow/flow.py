from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import BaseModel

from src.flow.agents.orchestrator import OrchestratorInput, OrchestratorNode
from src.flow.agents.precheck import PrecheckInput, PrecheckNode, PrecheckNodeOutput
from src.flow.types import (
    BasicMessage,
    ConversationState,
    FlowInput,
    FlowRunResult,
    FlowStepResult,
    NodeABC,
    NodeOutput,
    PrecheckStatus,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Flow:
    precheck: PrecheckNode = field(default_factory=PrecheckNode)
    orchestrator: OrchestratorNode = field(default_factory=OrchestratorNode)

    async def run(self, input: FlowInput) -> FlowRunResult:
        started_at = datetime.now(UTC)
        history = input.message_history[:-1]
        steps, final_state = await self._run_flow(input.current_message, history, input.state)
        finished_at = datetime.now(UTC)
        duration_ms = (finished_at - started_at).total_seconds() * 1000
        total_cost = sum(s.cost for s in steps if s.cost is not None) or None
        result = (
            steps[-1].output.model_copy(update={"updated_state": final_state})
            if steps
            else NodeOutput(updated_state=final_state)
        )
        
        # Debug logging: print analytics params from all steps
        logger.debug("=" * 60)
        logger.debug("FLOW EXECUTION TRACE")
        logger.debug("=" * 60)
        for i, step in enumerate(steps, 1):
            logger.debug(
                "Step %d - node=%s status=%s latency_ms=%.2f cost=%.4f",
                i, step.node_name, step.status, step.duration_ms, step.cost or 0.0
            )
            if step.analytics_params:
                logger.debug("  analytics: %s", step.analytics_params)
            if step.error:
                logger.debug("  error: %s", step.error)
        logger.debug("=" * 60)
        
        # Build analytics dict for FlowRunResult - includes full reasoning, input, output
        analytics = {
            "node_count": len(steps),
            "nodes": [s.node_name for s in steps],
            "status": steps[-1].status if steps else "no_steps",
            "total_latency_ms": duration_ms,
            "step_latency_ms": [s.duration_ms for s in steps],
            "total_cost": total_cost,
            "step_costs": [s.cost for s in steps],
            "step_reasoning": [s.analytics_params.get("reasoning", "N/A") for s in steps],
            "step_input": [s.analytics_params.get("input", "N/A") for s in steps],
            "step_output": [s.output.response.content if s.output.response else "no_response" for s in steps],
            "step_tokens": [s.analytics_params for s in steps],
        }
        
        return FlowRunResult(
            result=result,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            total_cost=total_cost,
            analytics=analytics,
        )

    async def _run_flow(
        self,
        current_message: BasicMessage,
        message_history: list[BasicMessage],
        state: ConversationState,
    ) -> tuple[list[FlowStepResult], ConversationState]:
        """precheck -> small_talk / not_allowed => stop here
                     -> allowed                  => orchestrator

        No retry loop here - transient failures are retried on the HTTP
        transport each agent's model client uses (see config.Settings.get_agent).
        """
        steps: list[FlowStepResult] = []

        precheck_input = PrecheckInput(
            current_message=current_message, message_history=message_history
        )
        step, state = await self._run_node(self.precheck, precheck_input, state)
        steps.append(step)
        if step.status == "failed":
            return steps, state

        precheck_output = cast(PrecheckNodeOutput, step.output)
        if precheck_output.status in (PrecheckStatus.SMALL_TALK, PrecheckStatus.NOT_ALLOWED):
            logger.info("Precheck stopped the flow: status=%s", precheck_output.status)
            return steps, state

        orchestrator_input = OrchestratorInput(
            current_message=current_message,
            message_history=message_history,
            language=precheck_output.language,
        )
        step, state = await self._run_node(self.orchestrator, orchestrator_input, state)
        steps.append(step)
        return steps, state

    async def _run_node(
        self, node: NodeABC[Any, Any], input: BaseModel, state: ConversationState
    ) -> tuple[FlowStepResult, ConversationState]:
        """Run one node once and build its FlowStepResult - the only place that does.

        Generic over node type: only touches the common NodeOutput shape
        (updated_state, analytics_params), never a node-specific field.
        """
        running_state = state.model_copy(update={"status": "running", "active_node": node.name})
        started_at = datetime.now(UTC)
        try:
            node_result = await node.run(input, running_state)
            finished_at = datetime.now(UTC)

            output = node_result.output
            next_state = (output.updated_state or running_state).model_copy(
                update={"status": "running", "active_node": node.name}
            )
            cost = node_result.analytics_params.get("cost")

            step = FlowStepResult(
                node_name=node.name,
                status="finished",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at).total_seconds() * 1000,
                output=output.model_copy(update={"updated_state": next_state}),
                analytics_params=node_result.analytics_params,
                cost=float(cost) if cost is not None else None,
            )
            return step, next_state
        except Exception as exc:
            logger.exception(
                "Node run failed node=%s conversation_id=%s", node.name, state.conversation_id
            )
            finished_at = datetime.now(UTC)
            failed_state = running_state.model_copy(
                update={"status": "failed", "active_node": None}
            )
            step = FlowStepResult(
                node_name=node.name,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=(finished_at - started_at).total_seconds() * 1000,
                output=NodeOutput(updated_state=failed_state, fallback_reason=str(exc)),
                analytics_params={"node": node.name, "cost": 0.0},
                error=str(exc),
            )
            return step, failed_state