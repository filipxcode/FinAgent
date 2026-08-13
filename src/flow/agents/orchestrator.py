from __future__ import annotations

from dataclasses import dataclass
from inspect import cleandoc

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.config.config import get_settings
from src.flow.agents.prompt import current_date
from src.flow.agents.researcher import ResearcherAgentOutput, ResearcherDeps
from src.flow.agents.researcher import agent as researcher_agent
from src.flow.agents.usage import run_cost
from src.flow.agents.whale_tracker import WhaleTrackerAgentOutput, WhaleTrackerDeps
from src.flow.agents.whale_tracker import agent as whale_tracker_agent
from src.flow.messages import to_model_messages
from src.flow.types import (
    BasicMessage,
    LanguageEnum,
    NodeABC,
    NodeContext,
    NodeInput,
    NodeOutput,
    NodeRunResult,
)


class OrchestratorInput(NodeInput):
    language: LanguageEnum


class OrchestratorAgentOutput(BaseModel):
    """Structured recap the orchestrator produces after it has answered the user.

    Filled by a second, tool-free run over the same conversation: the first run
    talks to the user, this one looks back at what actually happened and records
    it for downstream consumers (logging, analytics, follow-up routing).
    """

    reasoning: str = Field(
        description=cleandoc("""
            Brief explanation of how you arrived at the answer: which specialists
            you delegated to and why, what you answered from the conversation
            alone, and where the evidence was thin. 2-4 sentences, for developers
            reading traces - it is never shown to the user.
        """),
    )
    task_result: str = Field(
        default="None",
        description=cleandoc("""
            Summarize all used tool results using the gathered knowledge, keeping
            every detail that matters to the user's question. Do not alter any
            information - treat what the specialists returned as ground truth,
            including their dates and figures. Leave out anything the user did
            not ask about, and never name technical errors, tools or agents here.
            "None" when no tool was used (small talk or a purely conversational
            answer).
        """),
    )
    missing_informations: str = Field(
        default="None",
        description=cleandoc("""
            What the user's question needed but no specialist could supply - data
            that was not covered, a period with no readings, an ambiguity the user
            still has to resolve. Phrase it as the gap itself, not as a failure.
            "None" when the answer is complete.
        """),
    )
    tool_limitations: str = Field(
        default="None",
        description=cleandoc("""
            Constraints of the data behind the answer that shape how it should be
            read - e.g. on-chain readings are daily and close a day late, a
            confidence score came back low, a source returned partial results.
            "None" when nothing limited the answer.
        """),
    )


class OrchestratorNodeOutput(NodeOutput):
    """Flow-level result of one orchestrator turn."""

    reasoning: str = ""
    task_result: str = "None"
    missing_informations: str = "None"
    tool_limitations: str = "None"


class OrchestratorContext(NodeContext):
    pass


@dataclass
class OrchestratorDeps:
    """Runtime parameters for one orchestrator turn.

    The conversation itself travels as ``message_history`` on ``agent.run``, so
    it stays out of the instructions and out of every sub-agent. Deps carry only
    what the instructions and the delegation tools read at call time.
    """

    language: LanguageEnum = LanguageEnum.ENG


ORCHESTRATOR_AGENT_KEY = "orchestrator"
agent = get_settings().get_agent(
    ORCHESTRATOR_AGENT_KEY,
    deps_type=OrchestratorDeps,
    output_type=str,
)


@agent.instructions
async def get_agent_instructions(ctx: RunContext[OrchestratorDeps]) -> str:
    """Build the system prompt.

    Static by design apart from the date and language: the conversation arrives
    as ``message_history``, which keeps this prefix stable and cacheable across
    turns instead of being rewritten on every request.
    """
    prompt = """
    # CURRENT DATE
    {current_date}

    # ROLE
    You are the Orchestrator of a specialized Cryptocurrency & Financial
    Research Assistant. You are the only agent that talks to the user. You hold
    the conversation; your specialists do not — they answer one isolated task at
    a time and forget it immediately.

    # YOUR SPECIALISTS
    - delegate_research — market data and everything around it: prices, market
      caps, volumes, rankings, the Fear & Greed index, news, narratives, macro.
    - delegate_whale_tracking — on-chain movement of large holders: coins
      flowing onto or off exchanges, whether a day is unusual against its own
      history, and what one known wallet address holds and has moved.

    # HOW TO WORK
    - Answer directly, without delegating, when the question is about the
      conversation itself or about something a specialist already reported.
    - Delegate anything that needs fresh data. When a question spans both market
      data and on-chain movement, call both specialists in parallel rather than
      making one guess outside its area.
    - Specialists see NO conversation history. Every 'task' you send must stand
      on its own: resolve "it", "that coin", "the same period" into explicit
      names, tickers and dates first. "And its volume?" is useless to them;
      "What is Solana's 24h trading volume in USD?" works.
    - Never invent figures, dates, addresses or sources. Report only what a
      specialist actually returned, and keep the date it gave you.
    - Separate observation from interpretation, and say plainly when the data
      does not support an answer instead of filling the gap.
    - Reply to the user in {language}.

    # AFTER THE ANSWER
    Once the user has been answered you may be asked, with no new user message,
    to fill a structured recap of the turn. Then do not write to the user and do
    not call any tool: read back over what just happened and fill each field
    exactly as its description asks, using only what the specialists returned.
    """
    return cleandoc(prompt).format(
        current_date=current_date(),
        language=ctx.deps.language.value,
    )


@agent.tool
async def delegate_research(
    ctx: RunContext[OrchestratorDeps],
    task: str,
    background: str | None = None,
) -> ResearcherAgentOutput:
    R"""Delegate one self-contained market-research task to the Research Agent.

    Covers prices, market caps, volumes, rankings, market sentiment, news,
    narratives and macro context. For what large holders are doing on-chain,
    use `delegate_whale_tracking` instead.

    Args:
        task: A standalone research question. The agent sees none of this
            conversation, so name the coins, figures and time ranges explicitly.
        background: At most 1-3 sentences of earlier context, and only when it
            changes the answer — e.g. the user is comparing against a figure
            from an earlier turn. Leave unset otherwise.

    Returns the report with its sources and a 0-1 confidence. Low confidence
    means the evidence was thin — say so rather than presenting it as settled.
    """
    run = await researcher_agent.run(
        task,
        deps=ResearcherDeps(language=ctx.deps.language, background=background),
        usage=ctx.usage,
    )
    return run.output


@agent.tool
async def delegate_whale_tracking(
    ctx: RunContext[OrchestratorDeps],
    task: str,
    background: str | None = None,
) -> WhaleTrackerAgentOutput:
    R"""Delegate one self-contained on-chain task to the Whale Tracker Agent.

    Covers whether large holders are moving coins onto exchanges (positioning to
    sell) or off them (moving into storage), whether the latest day is unusual
    against its own history, and what one known wallet address holds and has
    moved. For prices, market caps or news, use `delegate_research` instead.

    Args:
        task: A standalone on-chain question. The agent sees none of this
            conversation, so name the asset, wallet address and comparison
            period explicitly.
        background: At most 1-3 sentences of earlier context, and only when it
            changes the answer. Leave unset otherwise.

    On-chain values are daily and published after a day closes, so the newest
    reading is normally yesterday's. Keep the returned 'latest_date' in your
    answer and never present the figures as intraday.
    """
    run = await whale_tracker_agent.run(
        task,
        deps=WhaleTrackerDeps(language=ctx.deps.language, background=background),
        usage=ctx.usage,
    )
    return run.output


@dataclass(kw_only=True)
class OrchestratorNode(NodeABC[OrchestratorInput, OrchestratorContext, OrchestratorNodeOutput]):
    name: str = "orchestrator_node"

    async def run_node(
        self,
        input: OrchestratorInput,
        context: OrchestratorContext,
    ) -> NodeRunResult[OrchestratorNodeOutput]:
        """Answer the user, then recap the turn in a second pass."""
        deps = OrchestratorDeps(language=input.language)
        answer_run = await agent.run(
            input.current_message.content,
            message_history=to_model_messages(input.message_history[:-1]),
            deps=deps,
        )
        recap_run = await agent.run(
            message_history=answer_run.all_messages(),
            output_type=OrchestratorAgentOutput,
            deps=deps,
        )
        recap = recap_run.output

        assistant_message = BasicMessage(
            conversation_id=input.current_message.conversation_id,
            role="assistant",
            content=answer_run.output,
        )
        return NodeRunResult(
            output=OrchestratorNodeOutput(
                response=assistant_message,
                reasoning=recap.reasoning,
                task_result=recap.task_result,
                missing_informations=recap.missing_informations,
                tool_limitations=recap.tool_limitations,
            ),
            analytics_params={
                "node": self.name,
                "cost": run_cost(answer_run, recap_run),
            },
        )
