"""Models for the tools the agents call and the helpers behind them.

The agents' own types (outputs, deps, node inputs) stay in each agent's module.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# --- researcher tools ---


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
    website_slug: str = Field(description="Identifier to pass to 'exact_crypto_tool'.")
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


# --- whale tracker tools ---


class WhaleAsset(StrEnum):
    BTC = "btc"
    ETH = "eth"


class Lookback(StrEnum):
    """How far back the comparison baseline reaches.

    This does not change which day is reported - the newest day is always
    reported. It sets what that day is measured against, so "is today unusual
    compared to the last two months" is '60d'. Short windows react fast
    but call ordinary weekly swings unusual; long windows only flag moves that
    are rare on that timescale.
    """

    WEEK = "7d"
    MONTH = "30d"
    TWO_MONTHS = "60d"
    QUARTER = "90d"
    HALF_YEAR = "180d"
    YEAR = "1y"


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
        description="Currency that 'balance' and every 'value' are denominated in."
    )
    label: str | None = Field(
        default=None, description="Known name of this address, when the explorer has one."
    )
    balance: float | None = None
    transactions: list[WalletTx] = Field(
        default_factory=list, description="Most recent transactions, newest first."
    )
    error: str | None = None


class WalletLookup(BaseModel):
    address: str
    chain: Chain


# --- news agent tools ---


class FeedOut(BaseModel):
    title: str
    url: str
    published: datetime | None = None
    source: str


class FeedToolResponse(BaseModel):
    results: list[FeedOut] = Field(
        default_factory=list,
        description="Newest first, across all configured feeds.",
    )
    error: str | None = None


class ExtractedArticle(BaseModel):
    url: str
    title: str = ""
    content: str = ""


class ExtractResponse(BaseModel):
    articles: list[ExtractedArticle] = Field(default_factory=list)
    failed_urls: list[str] = Field(
        default_factory=list,
        description="URLs Tavily could not extract - paywalled, blocked, or otherwise failed.",
    )
    error: str | None = None
