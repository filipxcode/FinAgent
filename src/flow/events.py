import asyncio
from contextvars import ContextVar
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class DelegationEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    agent: str
    status: Literal["started", "finished", "failed"]
    task: str | None = None


_queue: ContextVar[asyncio.Queue | None] = ContextVar("flow_events", default=None)

def emit(event: DelegationEvent) -> None:
    queue = _queue.get()
    if queue is not None:
        queue.put_nowait(event)