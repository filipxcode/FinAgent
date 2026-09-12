from fastapi.testclient import TestClient

from src.api import deps
from src.api.app import app


class _StubService:
    async def save_message(self, **kwargs):
        return None

    async def get_history(self, **kwargs):
        return []


app.dependency_overrides[deps.get_service] = lambda: _StubService()

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_conversation_without_auth_header_is_rejected():
    response = client.post("/conversation", json={"conversation": "hello"})
    assert response.status_code == 401


def test_conversation_with_wrong_token_is_rejected():
    response = client.post(
        "/conversation",
        json={"conversation": "hello"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401


def test_history_without_conversation_id_is_rejected():
    response = client.get("/history")
    assert response.status_code == 422
