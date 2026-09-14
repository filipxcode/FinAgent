from __future__ import annotations

from dataclasses import dataclass, field
from inspect import cleandoc
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.config.config import get_settings
from src.flow.agents.prompt import current_date
from src.flow.agents.usage import run_cost
from src.flow.messages import to_model_messages
from src.flow.types import (
    BasicMessage,
    ConversationState,
    LanguageEnum,
    NodeABC,
    NodeOutput,
    NodeRunResult,
)

_SUMMARY_MIN_CHARS = 400
_SUMMARY_MAX_CHARS = 1200


class HistorySummarizerInput(BaseModel):
    message_history: list[BasicMessage] = Field(default_factory=list)
    previous_summary: str | None = None
    language: LanguageEnum = LanguageEnum.ENG


class HistorySummarizerAgentOutput(BaseModel):
    """Structured rolling summary the LLM must produce for the older turns."""

    reasoning: str = Field(
        description=cleandoc("""
            One to two sentences on what you carried over from previous_summary
            versus the new turns, and what you dropped as no longer
            load-bearing. For developers reading traces - never shown to the user.
        """),
    )
    summary: str = Field(
        min_length=_SUMMARY_MIN_CHARS,
        max_length=_SUMMARY_MAX_CHARS,
        description=cleandoc(f"""
            Rolling summary of the conversation so far, {_SUMMARY_MIN_CHARS}-
            {_SUMMARY_MAX_CHARS} characters. Keep every coin, wallet address,
            figure, date and open question a follow-up might still need; drop
            greetings, retries and anything a later turn already superseded.
            Dense notes, not prose - this text replaces the raw turns for every
            future agent call and is never shown to the user.
        """),
    )


class HistorySummarizerNodeOutput(NodeOutput):
    """Flow-level result of one history-compaction pass."""

    reasoning: str = ""
    summary: str = ""


@dataclass
class HistorySummarizerDeps:
    """Runtime parameters for one history-compaction pass."""

    language: LanguageEnum = LanguageEnum.ENG
    previous_summary: str | None = None
    analytics_params: dict[str, Any] = field(default_factory=dict)


HISTORY_SUMMARIZER_AGENT_KEY = "history_summarizer"
agent = get_settings().get_agent(
    HISTORY_SUMMARIZER_AGENT_KEY,
    deps_type=HistorySummarizerDeps,
    output_type=HistorySummarizerAgentOutput,
)


@agent.instructions
async def get_agent_instructions(ctx: RunContext[HistorySummarizerDeps]) -> str:
    prompt = """
    # CURRENT DATE
    {current_date}

    # ROLE
    You are the History Summarizer Agent inside a specialized Cryptocurrency &
    Financial Research Assistant. You do NOT talk to the end user and you have
    no tools - you read the turns handed to you as message history and
    compress them into one rolling summary that stands in for those turns from
    now on.

    # HOW TO WORK
    - Fold 'previous_summary' (the compression of everything before this batch)
      together with the new turns in your message history into one updated
      summary - never summarize the new turns alone and lose what came before.
    - Keep what a later turn could still need: coins and wallets discussed,
      figures and dates already given to the user, open questions, and any
      caveat or limitation already surfaced. Drop greetings, retries and
      anything superseded by a later, more complete answer in the same history.
    - Never invent or silently drop a figure, date or address - if you cut
      something, it is because a later turn made it irrelevant, not because it
      didn't fit the length.
    - Write in {language}.

    # PREVIOUS SUMMARY
    {previous_summary}
    """
    return cleandoc(prompt).format(
        current_date=current_date(),
        language=ctx.deps.language.value,
        previous_summary=ctx.deps.previous_summary or "None - this is the first compaction.",
    )


@dataclass(kw_only=True)
class HistorySummarizerNode(NodeABC[HistorySummarizerInput, HistorySummarizerNodeOutput]):
    name: str = "history_summarizer_node"

    async def run_node(
        self, input: HistorySummarizerInput, state: ConversationState
    ) -> NodeRunResult[HistorySummarizerNodeOutput]:
        deps = HistorySummarizerDeps(
            language=input.language, previous_summary=input.previous_summary
        )
        run = await agent.run(
            message_history=to_model_messages(input.message_history),
            deps=deps,
        )
        result = run.output
        run_usage = run.usage if run.usage else None

        deps.analytics_params.update(
            {
                "node": self.name,
                "reasoning": result.reasoning,
                "cost": run_cost(run),
                "input": input.previous_summary or "None",
                "output": result.summary,
                "total_tokens": run_usage.total_tokens if run_usage else 0,
                "input_tokens": run_usage.input_tokens if run_usage else 0,
                "output_tokens": run_usage.output_tokens if run_usage else 0,
            }
        )
        return NodeRunResult(
            output=HistorySummarizerNodeOutput(
                updated_state=state.model_copy(update={"summary": result.summary}),
                reasoning=result.reasoning,
                summary=result.summary,
            ),
            analytics_params=deps.analytics_params,
        )
