from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from inspect import cleandoc
from math import ceil
from statistics import fmean, pstdev
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

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

WEI_PER_ETH = 1e18
SATOSHI_PER_BTC = 1e8


class WhaleTrackerInput(NodeInput):
    task: str
    language: LanguageEnum


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


class WhaleTrackerNodeOutput(NodeOutput):
    """Flow-level result of one whale-tracking task, mirroring the agent output."""

    reasoning: str = ""
    report: str = ""
    latest_date: date | None = None


class WhaleTrackerContext(NodeContext):
    pass



class WhaleAsset(StrEnum):
    BTC = "btc"
    ETH = "eth"


class Lookback(StrEnum):
    """How far back the comparison baseline reaches.

    This does not change which day is reported - the newest day is always
    reported. It sets what that day is measured against, so "is today unusual
    compared to the last two months" is TWO_MONTHS. Short windows react fast
    but call ordinary weekly swings unusual; long windows only flag moves that
    are rare on that timescale.
    """

    WEEK = "7d"
    MONTH = "30d"
    TWO_MONTHS = "60d"
    QUARTER = "90d"
    HALF_YEAR = "180d"
    YEAR = "1y"


_LOOKBACK_DAYS: dict[Lookback, int] = {
    Lookback.WEEK: 7,
    Lookback.MONTH: 30,
    Lookback.TWO_MONTHS: 60,
    Lookback.QUARTER: 90,
    Lookback.HALF_YEAR: 180,
    Lookback.YEAR: 365,
}


class WhaleMetric(StrEnum):
    """On-chain measurements of where large holders are moving their coins.

    Large holders have to use exchanges to sell, so coins moving onto exchanges
    is the earliest visible sign of intent to sell, and coins leaving is a sign
    of moving to long-term storage. What each value measures:

    EXCHANGE_INFLOW_USD
        Value sent to exchange wallets that day. A spike means holders are
        positioning to sell — this is the main early-warning signal.
    EXCHANGE_OUTFLOW_USD
        Value withdrawn from exchanges that day. A spike means coins are being
        moved into self-custody, which usually means holding rather than selling.
    COINS_HELD_ON_EXCHANGES
        Total coins sitting on exchanges. Read the direction, not the level:
        a falling trend is accumulation, a rising trend is building sell-side
        supply. Slower and more reliable than daily flows.
    ACTIVE_ADDRESSES
        Distinct addresses active that day. Use it to tell a whale-specific move
        apart from general network-wide activity.
    TRANSFER_COUNT
        On-chain transfers that day. Same purpose as active addresses: baseline
        context showing whether the whole chain got busier.
    """

    EXCHANGE_INFLOW_USD = "exchange_inflow_usd"
    EXCHANGE_OUTFLOW_USD = "exchange_outflow_usd"
    COINS_HELD_ON_EXCHANGES = "coins_held_on_exchanges"
    ACTIVE_ADDRESSES = "active_addresses"
    TRANSFER_COUNT = "transfer_count"


# Descriptive enum values keep the tool schema readable; these are the provider
# ids they translate to on the wire.
_METRIC_CODE: dict[WhaleMetric, str] = {
    WhaleMetric.EXCHANGE_INFLOW_USD: "FlowInExUSD",
    WhaleMetric.EXCHANGE_OUTFLOW_USD: "FlowOutExUSD",
    WhaleMetric.COINS_HELD_ON_EXCHANGES: "SplyExNtv",
    WhaleMetric.ACTIVE_ADDRESSES: "AdrActCnt",
    WhaleMetric.TRANSFER_COUNT: "TxTfrCnt",
}

_DEFAULT_METRICS: list[WhaleMetric] = [
    WhaleMetric.EXCHANGE_INFLOW_USD,
    WhaleMetric.EXCHANGE_OUTFLOW_USD,
    WhaleMetric.COINS_HELD_ON_EXCHANGES,
]


class MetricBucket(BaseModel):
    """Mean value over one slice of the window, for reading the trend shape."""

    start: date
    end: date
    mean: float


class MetricTrend(BaseModel):
    """One metric's newest reading, measured against its own recent history."""

    metric: WhaleMetric
    latest: float | None = Field(default=None, description="Newest daily value.")
    latest_date: date | None = None
    window_mean: float | None = Field(
        default=None, description="Average over the baseline window."
    )
    window_min: float | None = None
    window_max: float | None = None
    pct_vs_window_mean: float | None = Field(
        default=None,
        description="How far the newest value sits above the window average; +150.0 means 2.5x it.",
    )
    zscore: float | None = Field(
        default=None,
        description=(
            "Standard deviations from the window average. Between -2 and 2 is "
            "normal variation; beyond that the day is a genuine outlier worth "
            "reporting as unusual."
        ),
    )
    buckets: list[MetricBucket] = Field(
        default_factory=list,
        description="The window in equal slices, oldest first, for reading trend direction.",
    )


class WhaleFlowResponse(BaseModel):
    asset: WhaleAsset
    lookback: Lookback
    days: int = Field(description="Calendar days the baseline window covers.")
    latest_date: date | None = Field(
        default=None,
        description="Day the newest readings belong to — usually yesterday, never today.",
    )
    net_exchange_flow_usd_latest: float | None = Field(
        default=None,
        description=(
            "Inflow minus outflow on the newest day. Positive means coins moved "
            "net onto exchanges (sell-side pressure), negative means net withdrawal."
        ),
    )
    net_exchange_flow_usd_window: float | None = Field(
        default=None,
        description="The same net flow summed across the whole window, showing the longer trend.",
    )
    trends: list[MetricTrend] = Field(default_factory=list)
    error: str | None = None


class Chain(StrEnum):
    """Chains a wallet can be inspected on.

    Must match the address format: Bitcoin addresses start with 1, 3 or bc1,
    every other chain here uses 0x… addresses.
    """

    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    BASE = "base"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"


_NATIVE_SYMBOL: dict[Chain, str] = {
    Chain.BITCOIN: "BTC",
    Chain.ETHEREUM: "ETH",
    Chain.BASE: "ETH",
    Chain.ARBITRUM: "ETH",
    Chain.OPTIMISM: "ETH",
}


class WalletTx(BaseModel):
    hash: str
    timestamp: datetime | None = None
    value: float | None = Field(
        default=None, description="Amount moved, in the chain's native currency."
    )
    from_address: str | None = None
    to_address: str | None = None
    from_label: str | None = Field(
        default=None, description="Known name of the sender, when the explorer has one."
    )
    to_label: str | None = None


class WalletAccount(BaseModel):
    address: str
    chain: Chain
    native_symbol: str = Field(
        description="Currency that `balance` and every `value` are denominated in."
    )
    label: str | None = Field(
        default=None, description="Known name of this address, when the explorer has one."
    )
    balance: float | None = None
    transactions: list[WalletTx] = Field(
        default_factory=list, description="Most recent transactions, newest first."
    )
    error: str | None = None


@dataclass
class WhaleTrackerDeps:
    """Runtime parameters for one delegated whale-tracking task.

    Deliberately carries no conversation history: the Orchestrator owns the
    conversation and hands down a self-contained ``task`` plus, when it actually
    changes the answer, a short ``background`` it wrote itself.
    """

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
      "are they accumulating") -> coinmetrics_whale_flows. Questions about one
      known wallet address -> wallet_activity. You may call tools in parallel.
    - Pick 'lookback' from the question: "unusual today" -> 30d, "over the last
      two months" -> 60d. It sets the comparison baseline, not the reported day.
    - Judge unusualness by 'zscore', never by eyeballing the raw value. Inside
      -2..2 the day is ordinary — say so plainly rather than inventing a story.
      Outside it, state the direction and by how much.
    - Read 'coins_held_on_exchanges' as a direction over 'buckets', not as a
      level: falling means accumulation, rising means growing sell-side supply.
    - The newest reading is a closed day, normally yesterday. Never present it
      as intraday, "right now", or "today so far". Always name the date.
    - If a tool returns a non-empty 'error' field, the data source failed. Do
      not retry blindly — state the gap and answer with what you reliably have.
    - Never fabricate figures, addresses or transaction hashes. Report only what
      a tool actually returned, with its date.
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


async def _evm_wallet(address: str, chain: Chain, tx_limit: int) -> WalletAccount:
    """Balance and recent transfers for an EVM address, via Blockscout."""
    account = WalletAccount(
        address=address, chain=chain, native_symbol=_NATIVE_SYMBOL[chain]
    )
    host = get_settings().agent_utils_settings.blockscout_urls[chain.value]
    info, info_error = await get_json(f"{host}/api/v2/addresses/{address}")
    tx_data, tx_error = await get_json(f"{host}/api/v2/addresses/{address}/transactions")

    if balance := (info or {}).get("coin_balance"):
        account.balance = int(balance) / WEI_PER_ETH
    account.label = (info or {}).get("name")
    confirmed = [
        tx for tx in ((tx_data or {}).get("items") or []) if tx.get("block_number")
    ]
    for tx in confirmed[:tx_limit]:
        sender = tx.get("from") or {}
        recipient = tx.get("to") or {}
        account.transactions.append(
            WalletTx(
                hash=tx.get("hash", ""),
                timestamp=tx.get("timestamp"),
                value=int(tx.get("value") or 0) / WEI_PER_ETH,
                from_address=sender.get("hash"),
                to_address=recipient.get("hash"),
                from_label=sender.get("name"),
                to_label=recipient.get("name"),
            )
        )

    account.error = "; ".join(e for e in (info_error, tx_error) if e) or None
    return account


async def _bitcoin_wallet(address: str, tx_limit: int) -> WalletAccount:
    """Balance and recent transfers for a Bitcoin address, via mempool.space."""
    account = WalletAccount(
        address=address,
        chain=Chain.BITCOIN,
        native_symbol=_NATIVE_SYMBOL[Chain.BITCOIN],
    )
    url = get_settings().agent_utils_settings.mempool_url
    stats, stats_error = await get_json(f"{url}/address/{address}")
    txs, txs_error = await get_json(f"{url}/address/{address}/txs")

    if chain_stats := (stats or {}).get("chain_stats"):
        funded = chain_stats.get("funded_txo_sum", 0)
        spent = chain_stats.get("spent_txo_sum", 0)
        account.balance = (funded - spent) / SATOSHI_PER_BTC

    for tx in (txs or [])[:tx_limit]:
        outputs = tx.get("vout") or []
        payment = max(outputs, key=lambda out: out.get("value", 0), default={})
        inputs = tx.get("vin") or []
        sender = (inputs[0].get("prevout") or {}).get("scriptpubkey_address") if inputs else None
        block_time = (tx.get("status") or {}).get("block_time")
        account.transactions.append(
            WalletTx(
                hash=tx.get("txid", ""),
                timestamp=datetime.fromtimestamp(block_time, tz=UTC) if block_time else None,
                value=payment.get("value", 0) / SATOSHI_PER_BTC,
                from_address=sender,
                to_address=payment.get("scriptpubkey_address"),
            )
        )

    account.error = "; ".join(e for e in (stats_error, txs_error) if e) or None
    return account


@agent.tool_plain
async def wallet_activity(
    address: str,
    chain: Chain = Chain.ETHEREUM,
    tx_limit: int = 10,
) -> WalletAccount:
    R"""Inspect what one specific wallet holds and has recently moved.

    Use this when an address is already known — verifying a reported whale move,
    or following an entity's wallet. For "what are whales doing" in general,
    with no address in hand, use "coinmetrics_whale_flows" instead.

    Args:
        address: The wallet to inspect, on the chain given below.
        chain: Which chain the address belongs to. Must match the address
            format: Bitcoin addresses start with 1, 3 or bc1, every other chain
            uses 0x… addresses.
        tx_limit: How many recent transactions to return (default 10).

    Balances and values are in the chain's own currency, named in
    'native_symbol' — never assume USD. On Bitcoin a transaction has many
    senders and recipients, so 'value' is its largest output (the actual
    payment, excluding change) and the addresses are the dominant parties, not
    the only ones. On any API failure the response comes back with 'error' set
    instead of raising.
    """
    if chain is Chain.BITCOIN:
        return await _bitcoin_wallet(address, tx_limit)
    return await _evm_wallet(address, chain, tx_limit)


def _series(rows: list[dict[str, Any]], metric: WhaleMetric) -> list[tuple[date, float]]:
    """Pull one metric out of the mixed rows, dropping days it is missing from."""
    points: list[tuple[date, float]] = []
    for row in rows:
        raw = row.get(_METRIC_CODE[metric])
        if raw is None:
            continue
        try:
            points.append((datetime.fromisoformat(row["time"]).date(), float(raw)))
        except (ValueError, KeyError):
            continue
    return sorted(points)


def _bucket(points: list[tuple[date, float]], count: int = 10) -> list[MetricBucket]:
    """Compress a daily series into at most 'count' equal slices."""
    if not points:
        return []
    size = max(1, ceil(len(points) / count))
    return [
        MetricBucket(
            start=chunk[0][0],
            end=chunk[-1][0],
            mean=fmean(value for _, value in chunk),
        )
        for chunk in (points[i : i + size] for i in range(0, len(points), size))
    ]


def _trend(metric: WhaleMetric, points: list[tuple[date, float]]) -> MetricTrend:
    """Turn a daily series into latest-vs-baseline statistics."""
    trend = MetricTrend(metric=metric)
    if not points:
        return trend

    values = [value for _, value in points]
    latest_date, latest = points[-1]
    mean = fmean(values)
    spread = pstdev(values) if len(values) > 1 else 0.0

    trend.latest = latest
    trend.latest_date = latest_date
    trend.window_mean = mean
    trend.window_min = min(values)
    trend.window_max = max(values)
    trend.pct_vs_window_mean = (latest / mean - 1) * 100 if mean else None
    trend.zscore = (latest - mean) / spread if spread else None
    trend.buckets = _bucket(points)
    return trend


@agent.tool_plain
async def coinmetrics_whale_flows(
    asset: WhaleAsset = WhaleAsset.BTC,
    lookback: Lookback = Lookback.MONTH,
    metrics: list[WhaleMetric] | None = None,
) -> WhaleFlowResponse:
    R"""Detect what large holders are doing with their coins, chain-wide.

    Answers questions like "are whales moving right now", "is anything unusual
    happening on-chain", "are they accumulating or preparing to sell". It looks
    at aggregate flows across the whole chain, not at individual wallets — use
    `wallet_activity` when a specific address is already known.

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
            question: "unusual today" is served by 30d, "over the last two
            months" by 60d.
        metrics: What to measure. The default — exchange inflow, outflow and
            coins held on exchanges — answers the great majority of whale
            questions. Add active_addresses or transfer_count only to check
            whether a move was whale-specific or the whole chain being busier.

    Values are daily and published after a day closes, so the newest reading is
    normally yesterday's — never describe it as intraday or as "right now". On
    any API failure the response comes back with `error` set instead of raising.
    """
    supported = list(dict.fromkeys(metrics or _DEFAULT_METRICS))
    days = _LOOKBACK_DAYS[lookback]
    start_time = datetime.now(UTC) - timedelta(days=days)
    url = get_settings().agent_utils_settings.coinmetrics_url
    data, error = await get_json(
        f"{url}/timeseries/asset-metrics",
        params={
            "assets": asset.value,
            "metrics": ",".join(_METRIC_CODE[metric] for metric in supported),
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
    series = {metric: _series(rows, metric) for metric in supported}
    trends = [_trend(metric, series[metric]) for metric in supported]

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



@dataclass(kw_only=True)
class WhaleTrackerNode(NodeABC[WhaleTrackerInput, WhaleTrackerContext, WhaleTrackerNodeOutput]):
    name: str = "whale_tracker_node"

    async def run_node(
        self,
        input: WhaleTrackerInput,
        context: WhaleTrackerContext,
    ) -> NodeRunResult[WhaleTrackerNodeOutput]:
        deps = WhaleTrackerDeps(language=input.language)
        run = await agent.run(input.task, deps=deps)
        report = run.output

        assistant_message = BasicMessage(
            conversation_id=input.current_message.conversation_id,
            role="assistant",
            content=report.report,
        )
        return NodeRunResult(
            output=WhaleTrackerNodeOutput(
                response=assistant_message,
                reasoning=report.reasoning,
                report=report.report,
                latest_date=report.latest_date,
            ),
            analytics_params={"node": self.name, "cost": run_cost(run)},
        )
