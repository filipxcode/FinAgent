import json

from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api import deps
from src.api.app import app
from src.flow.types import BasicMessage, FlowRunResult, NodeOutput


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


def test_history_without_auth_header_is_rejected():
    response = client.get("/history", params={"conversation_id": "c1"})
    assert response.status_code == 401


def test_history_without_conversation_id_is_rejected():
    response = client.get("/history", headers={"x-api-key": "test-token"})
    assert response.status_code == 422


class _StubFlow:
    def __init__(self, sources):
        self._sources = sources

    async def stream(self, input):
        reply = BasicMessage(conversation_id="c1", role="assistant", content="answer")
        yield FlowRunResult(result=NodeOutput(response=reply), sources=self._sources)


def _sse_events(text):
    events = {}
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events[fields["event"]] = json.loads(fields["data"])
    return events


def test_stream_done_event_carries_the_sources_agents_cited(monkeypatch):
    urls = ["https://a.com/1", "https://b.com/2"]
    monkeypatch.setattr(app_module, "flow", _StubFlow(urls))
    response = client.post(
        "/stream/conversation",
        json={"conversation": "hi"},
        headers={"x-api-key": "test-token"},
    )
    assert response.status_code == 200
    assert _sse_events(response.text)["done"]["sources"] == urls


def test_conversation_response_carries_sources_and_defaults_to_none(monkeypatch):
    headers = {"x-api-key": "test-token"}
    monkeypatch.setattr(app_module, "flow", _StubFlow(["https://a.com/1"]))
    with_sources = client.post("/conversation", json={"conversation": "hi"}, headers=headers)
    assert with_sources.json()["sources"] == ["https://a.com/1"]

    monkeypatch.setattr(app_module, "flow", _StubFlow([]))
    without = client.post("/conversation", json={"conversation": "hi"}, headers=headers)
    assert without.json()["sources"] == []
