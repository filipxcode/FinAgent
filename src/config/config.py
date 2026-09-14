import logging
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import quote_plus

from httpx import AsyncClient, HTTPStatusError, TransportError
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import before_sleep_log, retry_if_exception_type, stop_after_attempt

retry_logger = logging.getLogger("finagent.retry")


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class FlowSettings(BaseSettings):
    max_steps: int = 32
    retry_max_attempts: int = 3
    conversation_rate_limit: str = "20/minute"


class AgentUtilsSettings(BaseSettings):
    """External API keys / config used by agent tools."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    tavily_api_key: str | None = None

    alternativeme_url: str = "https://api.alternative.me"
    coinmetrics_url: str = "https://community-api.coinmetrics.io/v4"
    mempool_url: str = "https://mempool.space/api"
    blockscout_urls: dict[str, str] = Field(
        default_factory=lambda: {
            "ethereum": "https://eth.blockscout.com",
            "base": "https://base.blockscout.com",
            "arbitrum": "https://arbitrum.blockscout.com",
            "optimism": "https://explorer.optimism.io",
        }
    )
    rss_urls: list[str]=["https://www.federalreserve.gov/feeds/press_all.xml", "https://www.sec.gov/news/pressreleases.rss"]
    wallet_batch_max_addresses: int = 10
    wallet_batch_max_concurrency: int = 5
    news_keywords: list[str] = Field(
        default_factory=lambda: [
            "crypto", "bitcoin", "ethereum", "digital asset", "stablecoin",
            "cpi", "inflation", "interest rate", "fomc", "federal funds",
            "rate cut", "rate hike", "etf",
        ]
    )


class AgentSettings(BaseModel):
    provider: Literal["openai"] = "openai"
    name: str = "orchestrator"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int | None = None

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"


class DatabaseSettings(BaseSettings):
    host: str = "localhost"
    port: int = 5432
    database: str = "finagent"
    user: str = "postgres"
    password: str = "postgres"
    ssl_mode: str = "prefer"

    @property
    def sqlalchemy_url(self) -> str:
        user = quote_plus(self.user)
        password = quote_plus(self.password)
        return (
            f"postgresql+asyncpg://{user}:{password}@{self.host}:{self.port}/"
            f"{self.database}"
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    logging_settings: LoggingSettings = Field(default_factory=LoggingSettings)
    flow_settings: FlowSettings = Field(default_factory=FlowSettings)
    agent_utils_settings: AgentUtilsSettings = Field(default_factory=AgentUtilsSettings)
    agent_settings: dict[str, AgentSettings] = Field(
        default_factory=lambda: {
            "precheck": AgentSettings(
                provider="openai",
                name="precheck",
                model="gpt-4.1-mini",
            ),
            "orchestrator": AgentSettings(
                provider="openai",
                name="orchestrator",
                model="gpt-4.1-mini",
            ),
            "researcher": AgentSettings(
                provider="openai",
                name="researcher",
                model="gpt-4.1-mini",
            ),
            "whaletracker": AgentSettings(
                provider="openai",
                name="whaletracker",
                model="gpt-4.1-mini",
            ),
            "news_agent": AgentSettings(
                provider="openai",
                name="news_agent",
                model="gpt-4.1-mini",
            ),
            "answer": AgentSettings(
                provider="openai",
                name="answer",
                model="gpt-4.1-mini",
            ),
            "history_summarizer": AgentSettings(
                provider="openai",
                name="history_summarizer",
                model="gpt-4.1-mini",
            ),
        }
    )
    db_settings: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api_key: str

    def get_agent_settings(self, agent_key: str) -> AgentSettings:
        try:
            return self.agent_settings[agent_key]
        except KeyError as exc:
            available = ", ".join(sorted(self.agent_settings)) or "<none>"
            raise KeyError(
                f"Unknown agent_settings key: {agent_key}. Available keys: {available}"
            ) from exc

    def get_agent(
        self,
        agent_key: str,
        *,
        deps_type: type[Any] | None = None,
        output_type: type[Any] | None = None,
    ) -> Agent[Any, Any]:
        """Build an agent whose HTTP client retries transient failures itself.
        """
        agent_settings = self.get_agent_settings(agent_key)
        transport = AsyncTenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type((HTTPStatusError, TransportError)),
                wait=wait_retry_after(max_wait=30),
                stop=stop_after_attempt(self.flow_settings.retry_max_attempts),
                reraise=True,
                before_sleep=before_sleep_log(retry_logger, logging.WARNING),
            ),
            validate_response=lambda r: r.raise_for_status(),
        )
        provider = OpenAIProvider(http_client=AsyncClient(transport=transport))
        model = OpenAIChatModel(agent_settings.model, provider=provider)
        return Agent(
            model=model,
            name=agent_settings.name,
            deps_type=deps_type,
            output_type=output_type,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


