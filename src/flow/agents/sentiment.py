from __future__ import annotations

from dataclasses import dataclass
from inspect import cleandoc
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.config.config import get_settings
from src.flow.agents.http import get_json
from src.flow.agents.prompt import current_date
from src.flow.types import LanguageEnum



class SentimentAgentOutput(BaseModel):
    """Structured verdict the LLM must produce for the message being classified."""

    reasoning: str = Field(
        description=cleandoc("""
            Brief explanation of your decision: what the message is asking, which
            earlier turn you resolved it against when it was ambiguous, and why
            that lands on this status rather than a neighbouring one. 1-3
            sentences, for developers reading traces - it never reaches the user.
        """),
    )



@dataclass
class SentimentDeps:
    language: LanguageEnum = LanguageEnum.ENG


PRECHECK_AGENT_KEY = "sentiment"
agent = get_settings().get_agent(
    PRECHECK_AGENT_KEY,
    deps_type=SentimentDeps,
    output_type=SentimentAgentOutput,
)


@agent.instructions
async def get_agent_instructions() -> str:
    prompt = """
    
    """
    return cleandoc(prompt).format(current_date=current_date())


@agent.tool_plain
async def search_reddit(query: str):
    