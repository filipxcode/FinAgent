from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from inspect import cleandoc
from typing import Annotated
import asyncio

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.config.config import get_settings
from src.flow.agents.http import get_json
from src.flow.agents.prompt import current_date
from src.flow.agents.types import (
    Lookback,
    WalletAccount,
    WalletLookup,
    WhaleAsset,
    WhaleFlowResponse,
    WhaleMetric,
)
from src.flow.agents.utils import METRIC_CODE, fetch_wallets, metric_series, metric_trend
from src.flow.types import LanguageEnum

logger = logging.getLogger(__name__)


class WhaleTrackerAgentOutput(BaseModel):
    """Structured answer the LLM must produce for one delegated whale task."""

    reasoning: str = Field(
        description=cleandoc("""
            Brief explanation of how you got to the report: which tools and
            lookback you chose and why, how you read the zscore, and what you
            deliberately did not conclude. 2-4 sentences, for developers reading
            traces - it never reaches the end user.
        """),
    )
    report: str = Field(
        description=cleandoc("""
            Concise, evidence-backed answer to the task. Name the date of every
            reading you cite, judge unusualness by zscore rather than by the raw
            value, and keep observation and interpretation apart - "inflows were
            2.4x their 30-day average" is an observation, "whales are about to
            sell" is a guess and must be marked as one.
            Describe exchange flows by direction in plain words, never as a bare
            "outflow" or "inflow": say whether coins were withdrawn from
            exchanges or deposited to them, together with the usual reading of
            that direction. Say whether each figure is one day (name the date)
            or summed over the whole window (name its length).
        """),
    )
    latest_date: date | None = Field(
        default=None,
        description=cleandoc("""
            Day the on-chain readings the report leans on belong to. These are
            daily values published after a day closes, so this is normally
            yesterday and never today. Leave unset only when no dated on-chain
            reading backs the report (e.g. a wallet lookup alone).
        """),
    )


_LOOKBACK_DAYS: dict[Lookback, int] = {
    Lookback.WEEK: 7,
    Lookback.MONTH: 30,
    Lookback.TWO_MONTHS: 60,
    Lookback.QUARTER: 90,
    Lookback.HALF_YEAR: 180,
    Lookback.YEAR: 365,
}


_DEFAULT_METRICS: list[WhaleMetric] = [
    WhaleMetric.EXCHANGE_INFLOW_USD,
    WhaleMetric.EXCHANGE_OUTFLOW_USD,
    WhaleMetric.COINS_HELD_ON_EXCHANGES,
]


@dataclass
class WhaleTrackerDeps:
    """Runtime parameters for one delegated whale-tracking task."""

    language: LanguageEnum = LanguageEnum.ENG
    background: str | None = None


WHALE_TRACKER_AGENT_KEY = "whaletracker"
agent = get_settings().get_agent(
    WHALE_TRACKER_AGENT_KEY,
    deps_type=WhaleTrackerDeps,
    output_type=WhaleTrackerAgentOutput,
)

@agent.instructions
async def get_agent_instructions(ctx: RunContext[WhaleTrackerDeps]) -> str:
    prompt = """
    # CURRENT DATE
    {current_date}

    # ROLE
    You are the Whale Tracker Agent inside a specialized Cryptocurrency &
    Financial Research Assistant. You do NOT talk to the end user directly -
    the Orchestrator agent delegates a single, self-contained task to you and
    expects a factual, evidence-backed answer back.

    # WHAT YOU DO
    You answer what large holders are doing on-chain: whether coins are moving
    onto exchanges (positioning to sell) or off them (moving into storage),
    whether the current day is unusual against its own history, and what a
    specific wallet has been doing. You do NOT cover prices, market caps,
    valuations or news — another agent owns those. If the task is really about
    market data rather than on-chain movement, say so instead of guessing.

    # HOW TO WORK
    - Chain-wide questions ("are whales moving", "anything unusual on-chain",
      "are they accumulating") -> 'coinmetrics_whale_flows'. Questions about one
      known wallet address -> 'wallet_activity'. You may call tools in parallel.
    - Pick 'lookback' from the question: "unusual today" -> '30d', "over the
      last two months" -> '60d'. It sets the comparison baseline, not the
      reported day.
    - Judge unusualness by 'zscore', never by eyeballing the raw value. Inside
      -2..2 the day is ordinary — say so plainly rather than inventing a story.
      Outside it, state the direction and by how much.
    - Read 'coins_held_on_exchanges' as a direction over 'buckets', not as a
      level: falling means accumulation, rising means growing sell-side supply.
    - The newest reading is a closed day, normally yesterday. Never present it
      as intraday, "right now", or "today so far". Always name the date.
    - Call a tool only with argument values its schema allows. When the task
      asks about an asset, period or metric the tools do not support, do not
      call them for it: state that gap in the report and answer the rest.
    - If a tool returns a non-empty 'error' field, the data source failed. Do
      not retry blindly — state the gap and answer with what you reliably have.
    - Never fabricate figures, addresses or transaction hashes. Report only what
      a tool actually returned, with its date.
    - Net exchange flow: negative means net withdrawal from exchanges
      (accumulation), positive means net deposits to exchanges (sell-side
      pressure). Your report is read by agents that never see the tools, and a
      bare "outflow" gets misread as money leaving the market, so spell out the
      direction and its usual reading in words.
    - 'net_exchange_flow_usd_latest' is a single day, 'net_exchange_flow_usd_window'
      is the sum over the whole lookback. Always say which one a figure is.
    - Distinguish observation from inference. "Exchange inflows were 2.4x their
      30-day average" is an observation; "whales are about to sell" is a guess.
      Give the observation first, and mark any interpretation as such.
    - Write your answer in {language}.
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


_MAX_BATCH_ADDRESSES = get_settings().agent_utils_settings.wallet_batch_max_addresses
_MAX_CONCURRENCY = get_settings().agent_utils_settings.wallet_batch_max_concurrency


@agent.tool_plain
async def wallet_activity(
    addresses: Annotated[list[WalletLookup], Field(max_length=_MAX_BATCH_ADDRESSES)],
    tx_limit: int = 8,
) -> tuple[list[WalletAccount], str]:
    R"""Inspect what one or more known wallets hold and have recently moved.

    Use this when at least one address is already known — verifying a reported
    whale move, or following an entity's wallet(s). For "what are whales doing"
    in general, with no address in hand, use 'coinmetrics_whale_flows' instead.

    Args:
        addresses: Up to _MAX_BATCH_ADDRESSES (address, chain) pairs to inspect.
            chain must match the address format: Bitcoin addresses start with
            1, 3 or bc1, every other chain here uses 0x… addresses. Lookups run
            concurrently, capped at _MAX_CONCURRENCY in flight.
        tx_limit: How many recent transactions to return per wallet (default 8).

    Balances and values are in each chain's own currency, named in
    'native_symbol' — never assume USD. On Bitcoin a transaction has many
    senders and recipients, so 'value' is its largest output (the actual
    payment, excluding change) and the addresses are the dominant parties, not
    the only ones. A wallet that fails is dropped from the returned list and
    named instead in the second, error-summary return value - it never raises.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    tasks = [fetch_wallets(semaphore, a, tx_limit) for a in addresses]
    tasks_processed = await asyncio.gather(*tasks)
    errors = ""
    wallets = []
    for t in tasks_processed:
        if t[0]:
            wallets.append(t[0])
        else:
            errors += f"\t{t[1]}"
    return wallets, errors


@agent.tool_plain
async def coinmetrics_whale_flows(
    asset: WhaleAsset = WhaleAsset.BTC,
    lookback: Lookback = Lookback.MONTH,
    metrics: list[WhaleMetric] | None = None,
) -> WhaleFlowResponse:
    R"""Detect what large holders are doing with their coins, chain-wide.

    Answers questions like "are whales moving right now", "is anything unusual
    happening on-chain", "are they accumulating or preparing to sell". 

    How to read the result. Every metric comes back as its newest daily value
    plus statistics against the baseline window. 'zscore' is the judgement call:
    within -2..2 the day is ordinary and should not be called unusual, beyond it
    the day genuinely stands out. 'pct_vs_window_mean' says by how much, and
    'buckets' show whether the trend has been building or fading. On top of that
    'net_exchange_flow_usd_latest' gives the single most useful number: positive
    means coins moved net onto exchanges (holders positioning to sell), negative
    means net withdrawal (moving into storage).

    Args:
        asset: Chain to inspect. Only Bitcoin and Ethereum have free on-chain data.
        lookback: How far back the comparison baseline reaches. Pick it from the
            question: "unusual today" is served by '30d', "over the last two
            months" by '60d'.
        metrics: What to measure. The default — exchange inflow, outflow and
            coins held on exchanges — answers the great majority of whale
            questions. Add 'active_addresses' or 'transfer_count' only to check
            whether a move was whale-specific or the whole chain being busier.

    Values are daily and published after a day closes, so the newest reading is
    normally yesterday's — never describe it as intraday or as "right now". On
    any API failure the response comes back with 'error' set instead of raising.
    """
    supported = list(dict.fromkeys(metrics or _DEFAULT_METRICS))
    days = _LOOKBACK_DAYS[lookback]
    start_time = datetime.now(UTC) - timedelta(days=days)
    url = get_settings().agent_utils_settings.coinmetrics_url
    data, error = await get_json(
        f"{url}/timeseries/asset-metrics",
        params={
            "assets": asset.value,
            "metrics": ",".join(METRIC_CODE[metric] for metric in supported),
            "frequency": "1d",
            "start_time": start_time.date().isoformat(),
            "page_size": 10000,
        },
    )
    if error:
        return WhaleFlowResponse(
            asset=asset,
            lookback=lookback,
            days=days,
            error=error,
        )

    rows = (data or {}).get("data", [])
    series = {metric: metric_series(rows, metric) for metric in supported}
    trends = [metric_trend(metric, series[metric]) for metric in supported]

    inflow = dict(series.get(WhaleMetric.EXCHANGE_INFLOW_USD, []))
    outflow = dict(series.get(WhaleMetric.EXCHANGE_OUTFLOW_USD, []))
    shared = sorted(inflow.keys() & outflow.keys())
    net_latest = inflow[shared[-1]] - outflow[shared[-1]] if shared else None
    net_window = sum(inflow[day] - outflow[day] for day in shared) if shared else None

    latest_dates = [trend.latest_date for trend in trends if trend.latest_date]
    return WhaleFlowResponse(
        asset=asset,
        lookback=lookback,
        days=days,
        latest_date=max(latest_dates) if latest_dates else None,
        net_exchange_flow_usd_latest=net_latest,
        net_exchange_flow_usd_window=net_window,
        trends=trends,
    )

