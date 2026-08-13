from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from inspect import cleandoc
from typing import Annotated, Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from tavily import AsyncTavilyClient

from src.config.config import get_settings
from src.flow.agents.http import get_json
from src.flow.agents.prompt import current_date
from src.flow.agents.usage import run_cost
from src.flow.types import (
    BasicMessage,
    LanguageEnum,
    NodeABC,
    NodeContext,
    NodeInput,
    NodeOutput,
    NodeRunResult,
)

logger = logging.getLogger(__name__)


class ResearcherInput(NodeInput):
    task: str
    language: LanguageEnum


class ResearcherContext(NodeContext):
    pass


class ResearcherAgentOutput(BaseModel):
    """Structured answer the LLM must produce for one delegated research task."""

    reasoning: str = Field(
        description=cleandoc("""
            Brief explanation of how you got to the report: which tools you chose
            and why, how you reconciled figures that disagreed, and what drove the
            confidence you gave. 2-4 sentences, for developers reading traces - it
            never reaches the end user.
        """),
    )
    report: str = Field(
        description=cleandoc("""
            Concise, factual synthesis that answers the task. Give every figure
            with its context (date and source) and keep monetary amounts in USD.
            State observations before any interpretation, and name the gap
            outright when the evidence does not settle the question.
        """),
    )
    sources: list[str] = Field(
        default_factory=list,
        description=cleandoc("""
            URLs and references the report actually rests on - only sources you
            really retrieved, never reconstructed ones. Data read from the crypto
            tools can be cited by tool name.
        """),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=cleandoc("""
            How well the gathered evidence supports the report, 0-1. Above 0.8 the
            figures came straight from a working data source; around 0.5 they are
            partial or second-hand; below 0.3 a source failed or the evidence is
            too thin to lean on, and the Orchestrator will say so to the user.
        """),
    )


class ResearcherNodeOutput(NodeOutput):
    """Flow-level result of one research task, mirroring the agent output."""

    reasoning: str = ""
    report: str = ""
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class TavilyResult(BaseModel):
    title: str
    url: str
    content: str
    score: float | None = None


class TavilySearchResponse(BaseModel):
    query: str
    answer: str | None = None
    results: list[TavilyResult] = Field(default_factory=list)
    error: str | None = None


class AltMeSort(StrEnum):
    ID = "id"
    RANK = "rank"
    NAME = "name"
    PRICE = "price"
    VOLUME_24H = "volume_24h"
    PERCENT_CHANGE_1H = "percent_change_1h"
    PERCENT_CHANGE_24H = "percent_change_24h"
    PERCENT_CHANGE_7D = "percent_change_7d"
    CIRCULATING_SUPPLY = "circulating_supply"


class CoinTicker(BaseModel):
    """One coin, with its USD quote flattened onto the top level."""

    id: int
    name: str
    symbol: str
    website_slug: str = Field(description="Identifier to pass to exact_crypto_tool.")
    rank: int | None = None
    price_usd: float | None = None
    volume_24h_usd: float | None = None
    market_cap_usd: float | None = None
    percentage_change_1h: float | None = None
    percentage_change_24h: float | None = None
    percentage_change_7d: float | None = None
    circulating_supply: float | None = None
    max_supply: float | None = None
    last_updated: datetime | None = None


class MarketOverview(BaseModel):
    """Aggregate state of the whole crypto market, in USD."""

    total_market_cap_usd: float | None = None
    total_volume_24h_usd: float | None = None
    bitcoin_percentage_of_market_cap: float | None = None
    active_cryptocurrencies: int | None = None
    last_updated: datetime | None = None


class CryptoMarketResponse(BaseModel):
    market: MarketOverview | None = None
    coins: list[CoinTicker] = Field(default_factory=list)
    error: str | None = None


class CoinDetailResponse(BaseModel):
    coin: CoinTicker | None = None
    error: str | None = None


class FearGreedEntry(BaseModel):
    value: int = Field(description="0 = extreme fear, 100 = extreme greed.")
    value_classification: str = Field(
        description="Extreme Fear / Fear / Neutral / Greed / Extreme Greed.",
    )
    timestamp: datetime


class FearGreedResponse(BaseModel):
    entries: list[FearGreedEntry] = Field(
        default_factory=list,
        description="Newest first; one entry per day.",
    )
    error: str | None = None


def _parse_coin(raw: dict[str, Any]) -> CoinTicker:
    """Map one raw alternative.me ticker entry onto :class:`CoinTicker`."""
    quote = (raw.get("quotes") or {}).get("USD") or {}
    return CoinTicker(
        id=raw["id"],
        name=raw.get("name", ""),
        symbol=raw.get("symbol", ""),
        website_slug=raw.get("website_slug", ""),
        rank=raw.get("rank"),
        price_usd=quote.get("price"),
        volume_24h_usd=quote.get("volume_24h"),
        market_cap_usd=quote.get("market_cap"),
        percentage_change_1h=quote.get("percentage_change_1h"),
        percentage_change_24h=quote.get("percentage_change_24h"),
        percentage_change_7d=quote.get("percentage_change_7d"),
        circulating_supply=raw.get("circulating_supply"),
        max_supply=raw.get("max_supply"),
        last_updated=raw.get("last_updated"),
    )

@dataclass
class ResearcherDeps:
    """Runtime parameters for one delegated research task."""

    language: LanguageEnum = LanguageEnum.ENG
    background: str | None = None


RESEARCHER_AGENT_KEY = "researcher"
agent = get_settings().get_agent(
    RESEARCHER_AGENT_KEY,
    deps_type=ResearcherDeps,
    output_type=ResearcherAgentOutput,
)


@agent.instructions
async def get_agent_instructions(ctx: RunContext[ResearcherDeps]) -> str:
    prompt = """
    # ROLE
    You are the Research Agent inside a specialized Cryptocurrency & Financial
    Research Assistant. You do NOT talk to the end user directly — the
    Orchestrator agent delegates a single, self-contained research task to you
    and expects a factual, well-sourced answer back.

    # WHAT YOU DO
    Given the task, you gather evidence and synthesize it into a concise report.
    You can use toose all tools paralell, depends on task.

    # HOW TO WORK
    - Decide which tool(s) the task actually needs; you may call them multiple
      times with refined queries. Market overview / top coins ->
      base_crypto_tool. A single named coin -> exact_crypto_tool. Market mood ->
      fear_greed_index_tool. News, narratives, macro or anything the market
      endpoints do not cover -> tavily_search.
    - Every monetary figure the crypto tools return is denominated in USD.
      Always state amounts as USD; never convert to another currency, and never
      imply a figure is in anything else — not even when the user writes in
      another language.
    - If a tool returns a non-empty 'error' field, it means the data source
      failed. Do not retry blindly — note the gap, lower your 'confidence', and
      answer with whatever reliable evidence you have.
    - Never fabricate numbers, headlines, or URLs. Only cite sources you
      actually retrieved. Put every URL / reference you relied on into 'sources'.
    - Be precise and neutral. Report figures with their context (date, source).
    - Write the 'report' in {language} (matching the user's language).
    - Current date is {current_date}
    """
    prompt = cleandoc(prompt).format(
        current_date=current_date(),
        language=ctx.deps.language.value,
    )
    if ctx.deps.background:
        prompt += cleandoc(
            """

            # BACKGROUND
            Context the Orchestrator judged relevant. The task itself stays
            authoritative — use this only to disambiguate it:
            {background}
            """
        ).format(background=ctx.deps.background)
    return prompt


@agent.tool_plain
async def tavily_search(query: str, max_results: int = 5) -> TavilySearchResponse:
    R"""Search the public web for fresh information via the Tavily API.

    Use this for news, prices, narratives, project background, macro context —
    anything that is not a dedicated crypto-news feed. Prefer specific queries
    (include ticker, protocol name, date range). Returns ranked results with a
    short content snippet and, when available, a synthesized 'answer'.

    Args:
        query: The search query in natural language.
        max_results: How many results to return (1-10, default 5).

    On any transport/API failure the tool returns a response with `error` set
    (server-error fallback) instead of raising, so keep working with what you have.
    """
    api_key = get_settings().agent_utils_settings.tavily_api_key
    if not api_key:
        return TavilySearchResponse(query=query, error="TAVILY_API_KEY not configured")

    try:
        client = AsyncTavilyClient(api_key=api_key)
        data = await client.search(
            query,
            max_results=max(1, min(max_results, 10)),
            search_depth="advanced",
            include_answer=True,
        )
    except Exception as exc:
        logger.exception("Tavily search failed query=%s", query)
        return TavilySearchResponse(query=query, error=f"tavily server error: {exc}")

    results = [
        TavilyResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            content=item.get("content", ""),
            score=item.get("score"),
        )
        for item in data.get("results", [])
    ]
    answer = data.get("answer")
    return TavilySearchResponse(query=query, answer=answer, results=results)


@agent.tool_plain
async def base_crypto_tool(
    top_k_crypto: Annotated[int, Field(ge=1, le=50)] = 5,
    sort: AltMeSort = AltMeSort.RANK,
) -> CryptoMarketResponse:
    R"""Get a market-wide snapshot: total cap, BTC dominance, and the top coins.

    Use this as the starting point for any "how is the market / which coins"
    question. Sort by whatever the task implies; keep the default otherwise.

    Args:
        top_k_crypto: How many coins to return (1-50, default 5).
        sort: Field the coin list is ordered by (default: market-cap rank).

    All amounts are USD. On any API failure the response comes back with `error`
    set instead of raising.
    """
    url = get_settings().agent_utils_settings.alternativeme_url
    coins_data, coins_error = await get_json(
        f"{url}/v2/ticker/",
        params={
            "limit": top_k_crypto,
            "sort": sort.value,
            "convert": "USD",
            "structure": "array",
        },
    )
    market_data, market_error = await get_json(
        f"{url}/v2/global/", params={"convert": "USD"}
    )

    coins = [_parse_coin(raw) for raw in (coins_data or {}).get("data", [])]

    market: MarketOverview | None = None
    if raw_market := (market_data or {}).get("data"):
        quote = (raw_market.get("quotes") or {}).get("USD") or {}
        market = MarketOverview(
            total_market_cap_usd=quote.get("total_market_cap"),
            total_volume_24h_usd=quote.get("total_volume_24h"),
            bitcoin_percentage_of_market_cap=raw_market.get(
                "bitcoin_percentage_of_market_cap"
            ),
            active_cryptocurrencies=raw_market.get("active_cryptocurrencies"),
            last_updated=raw_market.get("last_updated"),
        )

    errors = [e for e in (coins_error, market_error) if e]
    return CryptoMarketResponse(
        market=market, coins=coins, error="; ".join(errors) or None
    )


@agent.tool_plain
async def exact_crypto_tool(website_slug: str) -> CoinDetailResponse:
    R"""Get the current USD figures for one specific coin.

    Use this once you know which coin the task is about. The `website_slug` is
    the lowercase name form ("bitcoin", "ethereum")

    Args:
        website_slug: Coin identifier, e.g. "bitcoin".

    All amounts are USD. On any API failure the response comes back with `error`
    set instead of raising.
    """
    url = get_settings().agent_utils_settings.alternativeme_url
    data, error = await get_json(
        f"{url}/v2/ticker/{website_slug}/", params={"convert": "USD"}
    )
    if error:
        return CoinDetailResponse(error=error)

    entries = list((data or {}).get("data", {}).values())
    if not entries:
        return CoinDetailResponse(error=f"no coin found for slug '{website_slug}'")
    return CoinDetailResponse(coin=_parse_coin(entries[0]))


@agent.tool_plain
async def fear_greed_index_tool(
    limit: Annotated[int, Field(ge=1, le=365)] = 1,
) -> FearGreedResponse:
    R"""Get the crypto Fear & Greed Index — one 0-100 score for market sentiment.

    Use this for questions about market mood or as sentiment context on top of
    price data. Ask for more than one entry only when the task needs a trend.

    Args:
        limit: How many daily readings to return, newest first (1-365, default 1).

    On any API failure the response comes back with 'error' set instead of raising.
    """
    url = get_settings().agent_utils_settings.alternativeme_url
    data, error = await get_json(f"{url}/fng/", params={"limit": limit})
    if error:
        return FearGreedResponse(error=error)

    entries = [FearGreedEntry(**entry) for entry in (data or {}).get("data", [])]
    return FearGreedResponse(entries=entries)


@dataclass(kw_only=True)
class ResearcherNode(NodeABC[ResearcherInput, ResearcherContext, ResearcherNodeOutput]):
    name: str = "researcher_node"

    async def run_node(
        self,
        input: ResearcherInput,
        context: ResearcherContext,
    ) -> NodeRunResult[ResearcherNodeOutput]:
        deps = ResearcherDeps(language=input.language)
        run = await agent.run(input.task, deps=deps)
        report = run.output

        assistant_message = BasicMessage(
            conversation_id=input.current_message.conversation_id,
            role="assistant",
            content=report.report,
        )
        return NodeRunResult(
            output=ResearcherNodeOutput(
                response=assistant_message,
                reasoning=report.reasoning,
                report=report.report,
                sources=report.sources,
                confidence=report.confidence,
            ),
            analytics_params={"node": self.name, "cost": run_cost(run)},
        )
