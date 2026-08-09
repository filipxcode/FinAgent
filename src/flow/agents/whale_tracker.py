from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from inspect import cleandoc

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from src.config.config import get_settings
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


class WhaleTrackerInput(NodeInput):
    language: LanguageEnum


class WhaleTrackerOutput(NodeOutput):
    pass


class WhaleTrackerContext(NodeContext):
    pass



class EtherscanTx(BaseModel):
    hash: str
    from_address: str
    to_address: str
    value_eth: float
    timestamp: datetime | None = None


class EtherscanAccount(BaseModel):
    address: str
    balance_eth: float | None = None
    transactions: list[EtherscanTx] = Field(default_factory=list)
    error: str | None = None


@dataclass
class WhaleTrackerDeps:
    messages: list[BasicMessage]


WHALE_TRACKER_AGENT_KEY = "whaletracker"
agent = get_settings().get_agent(
    WHALE_TRACKER_AGENT_KEY,
    deps_type=WhaleTrackerDeps,
    output_type=WhaleTrackerOutput,
)

@agent.instructions
async def get_agent_instructions(ctx: RunContext[WhaleTrackerDeps]) -> str:
    prompt = """
    ###Input
    Last user message:
    ```python
    {last_user_message}
    ```
    Message history
    ```python
    {messages}
    ```
    """
    prompt = cleandoc(prompt)
    last_user_message = ctx.deps.messages[-1].content if ctx.deps.messages else ""
    return prompt.format(
        last_user_message=last_user_message,
        messages=[m.model_dump() for m in ctx.deps.messages[:-1]],
    )


@agent.tool_plain
async def etherscan_account(address: str, tx_limit: int = 10) -> EtherscanAccount:
    R"""Look up on-chain Ethereum data for a single address via the Etherscan API.

    Use this to verify wallet activity — e.g. whale / tracked-entity movements:
    the current ETH balance and the most recent normal transactions. Only the
    important fields are returned (hash, counterparties, ETH value, timestamp);
    gas/nonce/input and other low-level fields are intentionally dropped.

    Args:
        address: The 0x… Ethereum address to inspect.
        tx_limit: Max recent transactions to return (default 10).

    On any transport/API failure the tool returns a response with `error` set
    (server-error fallback) instead of raising.
    """
    api_key = get_settings().agent_utils_settings.etherscan_api_key
    if not api_key:
        return EtherscanAccount(address=address, error="ETHERSCAN_API_KEY not configured")

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            balance_response = await client.get
                params={
                    "module": "account",
                    "action": "balance",
                    "address": address,
                    "tag": "latest",
                    "apikey": api_key,
                },
            )
            balance_response.raise_for_status()
            balance_data = balance_response.json()

            tx_response = await client.get(
                ETHERSCAN_API_URL,
                params={
                    "module": "account",
                    "action": "txlist",
                    "address": address,
                    "startblock": 0,
                    "endblock": 99999999,
                    "page": 1,
                    "offset": max(1, tx_limit),
                    "sort": "desc",
                    "apikey": api_key,
                },
            )
            tx_response.raise_for_status()
            tx_data = tx_response.json()
    except Exception as exc:
        logger.exception("Etherscan request failed address=%s", address)
        return EtherscanAccount(address=address, error=f"etherscan server error: {exc}")

    balance_eth: float | None = None
    if balance_data.get("status") == "1":
        balance_eth = int(balance_data["result"]) / WEI_PER_ETH

    transactions: list[EtherscanTx] = []
    if tx_data.get("status") == "1":
        for tx in tx_data.get("result", [])[:tx_limit]:
            timestamp = (
                datetime.fromtimestamp(int(tx["timeStamp"]), tz=UTC)
                if tx.get("timeStamp")
                else None
            )
            transactions.append(
                EtherscanTx(
                    hash=tx.get("hash", ""),
                    from_address=tx.get("from", ""),
                    to_address=tx.get("to", ""),
                    value_eth=int(tx.get("value", "0")) / WEI_PER_ETH,
                    timestamp=timestamp,
                )
            )

    return EtherscanAccount(
        address=address, balance_eth=balance_eth, transactions=transactions
    )



@dataclass(kw_only=True)
class WhaleTrackerNode(NodeABC[WhaleTrackerInput, WhaleTrackerContext, WhaleTrackerOutput]):
    name: str = "whale_tracker_node"

    async def run_node(
        self,
        input: WhaleTrackerInput,
        context: WhaleTrackerContext,
    ) -> NodeRunResult[NodeOutput]:
        return NodeRunResult(
            output=NodeOutput(output_messages=[], next_node=None),
            analytics_params={"node": self.name, "cost": 0.0},
        )
