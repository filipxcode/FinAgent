from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
import asyncio
from pydantic import BaseModel

from src.flow.agents.answer import AnswerInput, AnswerNode
from src.flow.agents.history_summarizer import HistorySummarizerInput, HistorySummarizerNode
from src.flow.agents.orchestrator import (
    OrchestratorInput,
    OrchestratorNode,
    OrchestratorNodeOutput,
)
from src.flow.agents.precheck import PrecheckInput, PrecheckNode
from src.flow.types import (
    BasicMessage,
    ConversationState,
    FlowInput,
    FlowEvent,
    FlowRunResult,
    FlowStepResult,
    NodeABC,
    NodeOutput,
    PrecheckStatus,
    StepFinished,
    StepStarted,
)
from src.flow.events import DONE, bind_queue, publish, step_scope
logger = logging.getLogger(__name__)


_RECENT_HISTORY_SIZE = 20
_SUMMARIZE_TRIGGER = 25
_MAX_SUMMARIZE_BATCH = 25


@dataclass(kw_only=True)
class Flow:
    precheck: PrecheckNode = field(default_factory=PrecheckNode)
    orchestrator: OrchestratorNode = field(default_factory=OrchestratorNode)
    answer: AnswerNode = field(default_factory=AnswerNode)
    history_summarizer: HistorySummarizerNode = field(default_factory=HistorySummarizerNode)

    async def run(self, input: FlowInput) -> FlowRunResult:
        """Drain stream() and hand back only its terminal result."""
        result: FlowRunResult | None = None
        async for item in self.stream(input):
            if isinstance(item, FlowRunResult):
                result = item
        if result is None:
            raise RuntimeError("Flow.stream did not yield a terminal FlowRunResult")
        return result

    async def _run_flow(
        self,
        current_message: BasicMessage,
        message_history: list[BasicMessage],
        state: ConversationState,
    ) -> AsyncIterator[FlowStepResult]:
        """precheck -> small_talk / not_allowed => stop here
                     -> allowed                  => orchestrator

        No retry loop here - transient failures are retried on the HTTP
        transport each agent's model client uses (see config.Settings.get_agent).

        Yields every step as it finishes - accumulating them is stream()'s job.
        """
        
        if len(message_history) >= _SUMMARIZE_TRIGGER:
            older, message_history = (
                message_history[-(_RECENT_HISTORY_SIZE + _MAX_SUMMARIZE_BATCH) : -_RECENT_HISTORY_SIZE],
                message_history[-_RECENT_HISTORY_SIZE:],
            )
            summarizer_input = HistorySummarizerInput(
                message_history=older, previous_summary=state.summary
            )
            _summarizer_output, step, state = await self._run_node(
                self.history_summarizer, summarizer_input, state
            )
            yield step
            if step.status == "failed":
                return

        precheck_input = PrecheckInput(
            current_message=current_message, message_history=message_history
        )
        precheck_output, step, state = await self._run_node(
            self.precheck, precheck_input, state
        )
        yield step
        if step.status == "failed":
            return

        if precheck_output.status in (
            PrecheckStatus.SMALL_TALK,
            PrecheckStatus.NOT_ALLOWED,
        ):
            logger.info("Precheck stopped the flow: status=%s", precheck_output.status)
            return

        orchestrator_input = OrchestratorInput(
            current_message=current_message,
            message_history=message_history,
            language=precheck_output.language,
        )
        orchestrator_output, step, state = await self._run_node(
            self.orchestrator, orchestrator_input, state
        )
        yield step
        if step.status == "failed":
            return

        answer_input = AnswerInput(
            current_message=current_message,
            message_history=message_history,
            language=precheck_output.language,
            task_result=orchestrator_output.task_result,
            missing_informations=orchestrator_output.missing_informations,
            tool_limitations=orchestrator_output.tool_limitations,
        )
        _answer_output, step, state = await self._run_node(self.answer, answer_input, state)
        yield step

    async def _run_node[TNodeOutput: NodeOutput](
        self, node: NodeABC[Any, TNodeOutput], input: BaseModel, state: ConversationState
    ) -> tuple[TNodeOutput, FlowStepResult, ConversationState]:
        """Run one node once and build its FlowStepResult - the only place that does.
        """
        running_state = state.model_copy(
            update={"status": "running", "active_node": node.name}
        )
        publish(StepStarted(step=node.name))
        started_at = datetime.now(UTC)
        with step_scope(node.name) as delegations:
            try:
                node_result = await node.run(input, running_state)
                finished_at = datetime.now(UTC)

                output = node_result.output
                next_state = (output.updated_state or running_state).model_copy(
                    update={"status": "running", "active_node": node.name}
                )
                cost = node_result.analytics_params.get("cost")

                output = output.model_copy(update={"updated_state": next_state})
                step = FlowStepResult(
                    node_name=node.name,
                    status="finished",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=(finished_at - started_at).total_seconds() * 1000,
                    output=output,
                    analytics_params=node_result.analytics_params,
                    cost=float(cost) if cost is not None else None,
                    delegations=list(delegations.values()),
                )
                return output, step, next_state
            except Exception as exc:
                logger.exception(
                    "Node run failed node=%s conversation_id=%s",
                    node.name,
                    state.conversation_id,
                )
                finished_at = datetime.now(UTC)
                failed_state = running_state.model_copy(
                    update={"status": "failed", "active_node": None}
                )
                fallback_output = cast(
                    TNodeOutput,
                    NodeOutput(updated_state=failed_state, fallback_reason=str(exc)),
                )
                step = FlowStepResult(
                    node_name=node.name,
                    status="failed",
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=(finished_at - started_at).total_seconds() * 1000,
                    output=fallback_output,
                    analytics_params={"node": node.name, "cost": 0.0},
                    error=str(exc),
                    delegations=list(delegations.values()),
                )
                return fallback_output, step, failed_state

    async def stream(
        self, input: FlowInput
    ) -> AsyncIterator[FlowEvent | FlowRunResult]:
        """Yield flow events as they happen - a step starting, a delegation
        moving, a step finishing - then one terminal FlowRunResult.
        """
        
        started_at = datetime.now(UTC)
        history = input.message_history[:-1]
        steps: list[FlowStepResult] = []
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self._queue_put(queue, input.current_message, history, input.state))
        try:
            while True:
                item = await queue.get()
                if item is DONE:
                    break
                if isinstance(item, StepFinished):
                    steps.append(item.result)
                yield item
            await task
        finally:
            task.cancel()
        finished_at = datetime.now(UTC)
        duration_ms = (finished_at - started_at).total_seconds() * 1000
        total_cost = sum(s.cost for s in steps if s.cost is not None) or None
        # last step carries the final state - no re-stamping needed here.
        result = (
            steps[-1].output if steps else NodeOutput(updated_state=input.state)
        )

        # Trace logging: print analytics params from all steps
        logger.info("=" * 60)
        logger.info("FLOW EXECUTION TRACE")
        logger.info("=" * 60)
        for i, step in enumerate(steps, 1):
            logger.info(
                "Step %d - node=%s status=%s latency_ms=%.2f cost=%.4f",
                i,
                step.node_name,
                step.status,
                step.duration_ms,
                step.cost or 0.0,
            )
            if step.analytics_params:
                logger.info("  analytics: %s", step.analytics_params)
            if step.error:
                logger.info("  error: %s", step.error)
        logger.info("=" * 60)

        analytics = {
            "node_count": len(steps),
            "nodes": [s.node_name for s in steps],
            "status": steps[-1].status if steps else "no_steps",
            "total_latency_ms": duration_ms,
            "step_latency_ms": [s.duration_ms for s in steps],
            "total_cost": total_cost,
            "step_costs": [s.cost for s in steps],
            "step_reasoning": [
                s.analytics_params.get("reasoning", "N/A") for s in steps
            ],
            "step_input": [s.analytics_params.get("input", "N/A") for s in steps],
            "step_output": [
                s.output.response.content if s.output.response else "no_response"
                for s in steps
            ],
            "step_tokens": [s.analytics_params for s in steps],
        }

        sources = next(
            (s.output.sources for s in steps if isinstance(s.output, OrchestratorNodeOutput)),
            [],
        )

        yield FlowRunResult(
            result=result,
            sources=sources,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            total_cost=total_cost,
            analytics=analytics,
        )

    async def _queue_put(
        self,
        queue: asyncio.Queue,
        current_message: BasicMessage,
        message_history: list[BasicMessage],
        state: ConversationState,
    ) -> None:
        """Producer task: run the flow, feeding its events into 'queue'"""
        bind_queue(queue)
        try:
            async for step in self._run_flow(current_message, message_history, state):
                publish(StepFinished.of(step))
        finally:
            queue.put_nowait(DONE)
