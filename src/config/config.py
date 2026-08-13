from functools import lru_cache
from typing import Any, Literal
from urllib.parse import quote_plus

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class FlowSettings(BaseSettings):
    max_steps: int = 32
    retry_max_attempts: int = 3
    retry_delay_seconds: float = 0.0


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
            f"{self.database}?sslmode={self.ssl_mode}"
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
        }
    )
    db_settings: DatabaseSettings = Field(default_factory=DatabaseSettings)

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
        agent_settings = self.get_agent_settings(agent_key)
        return Agent(
            model=agent_settings.model_id,
            name=agent_settings.name,
            deps_type=deps_type,
            output_type=output_type,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


