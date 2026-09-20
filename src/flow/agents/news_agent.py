from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from inspect import cleandoc

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from tavily import AsyncTavilyClient

from src.config.config import get_settings
from src.flow.agents.prompt import current_date
from src.flow.agents.types import ExtractedArticle, ExtractResponse, FeedToolResponse
from src.flow.agents.utils import get_feed, matches_keywords
from src.flow.types import LanguageEnum

logger = logging.getLogger(__name__)


class NewsAgentOutput(BaseModel):
    """Structured answer the LLM must produce for one delegated macro/news task."""

    reasoning: str = Field(
        description=cleandoc("""
            Brief explanation of how you got to the report: which headlines you
            judged relevant to the task and why, and what drove the confidence
            you gave. 2-4 sentences, for developers reading traces - it never
            reaches the end user.
        """),
    )
    report: str = Field(
        description=cleandoc("""
            Concise, factual synthesis that answers the task. Give every
            headline its date and source. State observations before any
            interpretation, and name the gap outright when no relevant news
            was found.
        """),
    )
    sources: list[str] = Field(
        default_factory=list,
        description="URLs the report actually rests on - only ones you really retrieved.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How well the gathered headlines support the report, 0-1.",
    )


@dataclass
class NewsAgentDeps:
    """Runtime parameters for one delegated news task."""

    language: LanguageEnum = LanguageEnum.ENG
    background: str | None = None


NEWS_AGENT_KEY = "news_agent"
agent = get_settings().get_agent(
    NEWS_AGENT_KEY,
    deps_type=NewsAgentDeps,
    output_type=NewsAgentOutput,
)


@agent.instructions
async def get_agent_instructions(ctx:RunContext[NewsAgentDeps]) -> str:
    prompt = """
    # ROLE
    You are the Macro & Regulatory News Agent inside a specialized
    Cryptocurrency & Financial Research Assistant. You do NOT talk to the end
    user directly - the Orchestrator delegates a single, self-contained task
    to you and expects a factual, well-sourced answer back.

    # HOW TO WORK
    - Call 'get_macro_news' to get the latest headlines from official macro
      (Federal Reserve) and regulatory (SEC) feeds.
    - Judge which returned headlines are actually relevant to the task - most
      are not, since these feeds cover everything the source publishes, not
      only crypto/markets.
    - A headline and date are rarely enough to answer the task. For any
      headline that looks directly relevant, call 'extract_article' with its
      exact URL to read the full story before writing your report - do not
      guess at the content from the title alone.
    - Never fabricate headlines, dates or URLs. Only cite sources you actually
      retrieved. Put every URL you relied on into 'sources'.
    - If a tool returns a non-empty 'error' field, note the gap, lower your
      'confidence', and answer with whatever reliable evidence you have. If
      nothing relevant was found, say so plainly rather than stretching an
      unrelated headline to fit.
    - Your only sources are the Fed and SEC feeds. When the task is about
      something outside them (legislation, Congress, other agencies, media
      coverage), say so explicitly in 'reasoning' and keep 'confidence' low -
      the Orchestrator uses that to route the topic to web research.
    - Current date is {current_date}
    - Language of your response {language}
    
    """
    if ctx.deps.background:
        prompt += cleandoc(
            """

            # BACKGROUND
            Context the Orchestrator judged relevant. The task itself stays
            authoritative — use this only to disambiguate it:
            {background}
            """
        ).format(background=ctx.deps.background)
    return cleandoc(prompt).format(
        current_date=current_date(),
        language=ctx.deps.language)


@agent.tool_plain
async def get_macro_news() -> FeedToolResponse:
    R"""Get the latest macro & regulatory headlines that plausibly move markets.

    Pulls from official sources only - Federal Reserve press releases (rate
    decisions, FOMC statements) and SEC press releases (enforcement,
    rulemaking) - then keeps only entries whose title matches a configured
    macro/crypto keyword list, since these sources publish on everything they
    handle, not just what's relevant here. Use this for "what's the latest on
    rates/regulation" style questions. For crypto-native news, legislation,
    other agencies or anything else these feeds don't cover, you have no other
    source: report low confidence and name what the feeds did not cover, so the
    Orchestrator can send the topic to web research.

    Each entry gives a title, link and published date - not the full article.
    If a headline looks worth digging into for the task, follow up on that URL
    rather than guessing at the story from the title alone.

    Results are newest first. On any transport/API failure the tool returns
    whatever feeds succeeded, with 'error' naming the ones that failed - it
    never raises.
    """
    settings = get_settings().agent_utils_settings
    fetched = await asyncio.gather(*(get_feed(u) for u in settings.rss_urls))

    results = [item for sublist, _ in fetched for item in sublist]
    results = [item for item in results if matches_keywords(item.title, settings.news_keywords)]
    results.sort(key=lambda item: item.published or datetime.min.replace(tzinfo=UTC), reverse=True)

    errors = [err for _, err in fetched if err]
    return FeedToolResponse(results=results, error="; ".join(errors) or None)


@agent.tool_plain
async def extract_article(urls: list[str]) -> ExtractResponse:
    R"""Fetch the full text of one or more article URLs from 'get_macro_news'.

    Use this once a headline looks directly relevant to the task, to read the
    actual story instead of just the title. Pass exact URLs you got back from
    'get_macro_news' - never a guessed or reconstructed link.

    Args:
        urls: One or more article URLs to fetch. Keep this small - only the
            headlines that actually matter to the task, not every one
            'get_macro_news' returned.

    On any transport/API failure the tool returns a response with 'error' set.
    A URL that fails extraction individually (paywalled, blocked, etc.) is
    listed in 'failed_urls' instead of failing the whole call.
    """
    api_key = get_settings().agent_utils_settings.tavily_api_key
    if not api_key:
        return ExtractResponse(error="TAVILY_API_KEY not configured")

    try:
        client = AsyncTavilyClient(api_key=api_key)
        data = await client.extract(urls=urls, extract_depth="basic", format="text")
    except Exception as exc:
        logger.exception("Tavily extract failed urls=%s", urls)
        return ExtractResponse(error=f"tavily server error: {exc}")

    articles = [
        ExtractedArticle(
            url=item.get("url", ""),
            title=item.get("title", ""),
            content=item.get("raw_content", ""),
        )
        for item in data.get("results", [])
    ]
    failed_urls = [item.get("url", "") for item in data.get("failed_results", [])]
    return ExtractResponse(articles=articles, failed_urls=failed_urls)
