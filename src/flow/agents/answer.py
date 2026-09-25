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


class AnswerInput(BaseModel):
    current_message: BasicMessage
    message_history: list[BasicMessage] = Field(default_factory=list)
    language: LanguageEnum = LanguageEnum.ENG
    task_result: str = "None"
    missing_informations: str = "None"
    tool_limitations: str = "None"


class AnswerAgentOutput(BaseModel):
    """Structured report the LLM must produce for the finished turn."""

    reasoning: str = Field(
        description=cleandoc("""
            One to two sentences on how you turned task_result into the report
            below - what you kept, what caveat you folded in and where. For
            developers reading traces - never shown to the user.
        """),
    )
    report: str = Field(
        description=cleandoc("""
            The final, report shown to the user, written in Markdown.
            Ground every claim in
            task_result - keep its figures, dates, addresses and sources exactly
            as given, never add one that isn't already there. Weave in
            missing_informations and tool_limitations as natural caveats where
            they matter, not as a bolted-on disclaimer section, and never name
            internal tools or agents. If task_result is "None" (small talk, no
            data gathered), keep this short and conversational instead of
            forcing structure onto it.
            Formatting: one level-2 heading (##) per topic, short paragraphs or
            hyphen bullets under it, bold for key figures, a table when comparing
            assets or periods. Small talk stays plain text with no headings. End
            with the last substantive section, never with a summary that repeats
            the sections above.
        """),
    )


class AnswerNodeOutput(NodeOutput):
    """Flow-level result of the answer/reporting turn."""

    reasoning: str = ""
    report: str = ""


@dataclass
class AnswerDeps:
    """Runtime parameters for one answer/report synthesis."""

    language: LanguageEnum = LanguageEnum.ENG
    task_result: str = "None"
    missing_informations: str = "None"
    tool_limitations: str = "None"
    analytics_params: dict[str, Any] = field(default_factory=dict)


ANSWER_AGENT_KEY = "answer"
agent = get_settings().get_agent(
    ANSWER_AGENT_KEY,
    deps_type=AnswerDeps,
    output_type=AnswerAgentOutput,
)


@agent.instructions
async def get_agent_instructions(ctx: RunContext[AnswerDeps]) -> str:
    """Build the system prompt."""
    prompt = """
    # CURRENT DATE
    {current_date}

    # ROLE
    You are the Answer Agent inside a specialized Cryptocurrency & Financial
    Research Assistant. You have no tools and gather no data yourself - the
    Orchestrator already delegated to its specialists and produced a summary of
    the turn. Your only job is to turn that summary into one well-written,
    accurate report.

    # WHAT YOU RECEIVE
    - 'task_result': the synthesized findings from this turn - your source of
      truth.
    - 'missing_informations': what the user's question needed but no specialist
      could supply.
    - 'tool_limitations': constraints on the data behind the answer.
    You do not see raw tool calls or specialist traces, and you cannot call any
    tool - everything you write must come from what's given below and the
    conversation history.

    TASK RESULT:
    {task_result}

    MISSING INFORMATIONS:
    {missing_informations}

    TOOL LIMITATIONS:
    {tool_limitations}

    # HOW TO WORK
    - Never invent a figure, date, address or source absent from 'task_result'.
    - Fold 'missing_informations' and 'tool_limitations' in naturally, only
      where they change how a claim should be read - not as a bolted-on
      disclaimer, and never name internal tools or agents.
    - Use the conversation history only to match tone and avoid repeating what
      the user already knows - do not summarize the whole conversation.
    - Write in {language}.
    - Format the answer in Markdown as described in the 'report' field.
    - Keep every qualifier that changes the meaning of a finding exactly as
      'task_result' states it: who acted, what kind of act or measurement it
      is, its direction, its status and the period it covers. Do not shorten
      a finding in a way that changes any of these.
    - Be concrete: lead with the figures and findings, cut filler and generic
      statements.
    """
    return cleandoc(prompt).format(
        current_date=current_date(),
        task_result=ctx.deps.task_result,
        missing_informations=ctx.deps.missing_informations,
        tool_limitations=ctx.deps.tool_limitations,
        language=ctx.deps.language.value,
    )


@dataclass(kw_only=True)
class AnswerNode(NodeABC[AnswerInput, AnswerNodeOutput]):
    name: str = "answer_node"

    async def run_node(
        self, input: AnswerInput, state: ConversationState
    ) -> NodeRunResult[AnswerNodeOutput]:
        _ = state
        deps = AnswerDeps(
            language=input.language,
            task_result=input.task_result,
            missing_informations=input.missing_informations,
            tool_limitations=input.tool_limitations,
        )
        run = await agent.run(
            input.current_message.content,
            message_history=to_model_messages(input.message_history),
            deps=deps,
        )
        answer = run.output
        run_usage = run.usage if run.usage else None

        assistant_message = BasicMessage(
            conversation_id=input.current_message.conversation_id,
            role="assistant",
            content=answer.report,
        )

        deps.analytics_params.update(
            {
                "node": self.name,
                "reasoning": answer.reasoning,
                "cost": run_cost(run),
                "input": str(input.current_message.content),
                "output": answer.report,
                "total_tokens": run_usage.total_tokens if run_usage else 0,
                "input_tokens": run_usage.input_tokens if run_usage else 0,
                "output_tokens": run_usage.output_tokens if run_usage else 0,
            }
        )
        return NodeRunResult(
            output=AnswerNodeOutput(
                response=assistant_message,
                reasoning=answer.reasoning,
                report=answer.report,
            ),
            analytics_params=deps.analytics_params,
        )
