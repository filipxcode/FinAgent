from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

from src.flow.types import Delegation, DelegationStatusT, FlowEvent

_queue: ContextVar[asyncio.Queue | None] = ContextVar("flow_events", default=None)
_current_step: ContextVar[str | None] = ContextVar("flow_current_step", default=None)
_step_delegations: ContextVar[dict[str, Delegation] | None] = ContextVar(
    "flow_step_delegations", default=None
)

DONE = object()


def bind_queue(queue: asyncio.Queue) -> None:
    _queue.set(queue)


def publish(event: FlowEvent) -> None:
    """Put an event on the bound queue"""
    queue = _queue.get()
    if queue is not None:
        queue.put_nowait(event)


@contextmanager
def step_scope(step: str) -> Generator[dict[str, Delegation]]:
    delegations: dict[str, Delegation] = {}
    step_token = _current_step.set(step)
    delegations_token = _step_delegations.set(delegations)
    try:
        yield delegations
    finally:
        _step_delegations.reset(delegations_token)
        _current_step.reset(step_token)


def _emit(id: str, agent: str, status: DelegationStatusT, task: str | None) -> None:
    event = Delegation(
        step=_current_step.get(), id=id, agent=agent, status=status, task=task
    )
    delegations = _step_delegations.get()
    if delegations is not None:
        delegations[id] = event
    publish(event)


@contextmanager
def delegation(agent: str, task: str) -> Generator[None]:
    id = uuid4().hex
    _emit(id, agent, "started", task)
    try:
        yield
    except Exception:
        _emit(id, agent, "failed", task)
        raise
    _emit(id, agent, "finished", task)
