"""Helpers the agents' tools lean on: parsing, fetching and light statistics."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime
from math import ceil
from statistics import fmean, pstdev
from typing import Any

import feedparser
import httpx

from src.config.config import get_settings
from src.flow.agents.http import get_json
from src.flow.agents.types import (
    Chain,
    CoinTicker,
    FeedOut,
    MetricBucket,
    MetricTrend,
    WalletAccount,
    WalletLookup,
    WalletTx,
    WhaleMetric,
)

logger = logging.getLogger(__name__)


# --- orchestrator ---


def add_sources(known: list[str], sources: list[str]) -> None:
    """Keep the links a specialist cited (not tool names it may list), once each."""
    for source in sources:
        if source.startswith(("http://", "https://")) and source not in known:
            known.append(source)


def record_failure(
    analytics_params: dict[str, Any], agent_name: str, task: str, reason: str, error: Exception
) -> None:
    """Log a specialist that failed its task and keep it in the turn's analytics."""
    logger.warning("Specialist %s failed on task %r: %s", agent_name, task, error)
    analytics_params.setdefault("delegations", []).append(
        {
            "agent": agent_name,
            "task": task,
            "orchestrator_reason": reason,
            "error": str(error),
        }
    )


# --- researcher ---


def parse_coin(raw: dict[str, Any]) -> CoinTicker:
    """Map one raw alternative.me ticker entry onto 'CoinTicker'."""
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


# --- whale tracker ---

WEI_PER_ETH = 1e18
SATOSHI_PER_BTC = 1e8

# Descriptive enum values keep the tool schema readable
METRIC_CODE: dict[WhaleMetric, str] = {
    WhaleMetric.EXCHANGE_INFLOW_USD: "FlowInExUSD",
    WhaleMetric.EXCHANGE_OUTFLOW_USD: "FlowOutExUSD",
    WhaleMetric.COINS_HELD_ON_EXCHANGES: "SplyExNtv",
    WhaleMetric.ACTIVE_ADDRESSES: "AdrActCnt",
    WhaleMetric.TRANSFER_COUNT: "TxTfrCnt",
}

_NATIVE_SYMBOL: dict[Chain, str] = {
    Chain.BITCOIN: "BTC",
    Chain.ETHEREUM: "ETH",
    Chain.BASE: "ETH",
    Chain.ARBITRUM: "ETH",
    Chain.OPTIMISM: "ETH",
}


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


async def fetch_wallets(semaphore: asyncio.Semaphore, address: WalletLookup, tx_limit: int) -> tuple[WalletAccount | None, str | None]:
    async with semaphore:
        try:
            if address.chain is Chain.BITCOIN:
                wallet_acc = await _bitcoin_wallet(address.address, tx_limit)
            else:
                wallet_acc = await _evm_wallet(address.address, address.chain, tx_limit)
            return wallet_acc, None
        except Exception:
            logger.exception("Error during wallet lookup address=%s", address)
            return None, f"Error during address = {address}"


def metric_series(rows: list[dict[str, Any]], metric: WhaleMetric) -> list[tuple[date, float]]:
    """Pull one metric out of the mixed rows, dropping days it is missing from."""
    points: list[tuple[date, float]] = []
    for row in rows:
        raw = row.get(METRIC_CODE[metric])
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


def metric_trend(metric: WhaleMetric, points: list[tuple[date, float]]) -> MetricTrend:
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


# --- news agent ---


async def get_feed(url: str) -> tuple[list[FeedOut], str | None]:
    """Fetch and parse one RSS/Atom feed into entries.
    """
    source = httpx.URL(url).host or url
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:
        logger.exception("RSS fetch failed url=%s", url)
        return [], f"{source} request error: {exc}"

    feed = feedparser.parse(response.content)
    results = []
    for entry in feed.entries:
        published = None
        if parsed := entry.get("published_parsed"):
            published = datetime(*parsed[:6], tzinfo=UTC)
        results.append(
            FeedOut(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                published=published,
                source=source,
            )
        )
    return results, None


def matches_keywords(title: str, keywords: list[str]) -> bool:
    """True if any keyword appears in the title, case-insensitive."""
    lowered = title.lower()
    return any(keyword.lower() in lowered for keyword in keywords)
