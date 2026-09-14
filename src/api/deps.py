from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from src.config.config import Settings
from src.config.config import get_settings as load_settings
from src.db.postgres import PostgresClient, get_postgres_client


async def get_auth_token(
	authorization: Annotated[str | None, Header(alias="Authorization")] = None,
	x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
) -> str:
	expected_token = get_settings().api_key

	token = None
	if x_api_key:
		token = x_api_key
	elif authorization:
		token = authorization.removeprefix("Bearer ").strip()

	if not token:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Missing authorization token",
		)

	if token != expected_token:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid token",
		)

	return token


@lru_cache
def get_settings() -> Settings:
	return load_settings()


async def get_database(
	settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[PostgresClient]:
	_ = settings
	client = get_postgres_client()
	if not client.is_connected:
		await client.connect()
	yield client


def get_service(
	database: Annotated[PostgresClient, Depends(get_database)],
) -> object:
	from src.db.postgres import get_service as build_service

	return build_service(database=database)