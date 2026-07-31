from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    db_settings: DatabaseSettings = Field(default_factory=DatabaseSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()


