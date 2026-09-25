import json

from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api import deps
from src.api.app import app
from src.db.postgres import DatabaseUnavailableError
from src.flow.types import BasicMessage, FlowRunResult, NodeOutput


class _StubService:
    conversations: list[dict] = []
    conversations_kwargs: dict = {}

    async def save_message(self, **kwargs):
        return None

    async def get_history(self, **kwargs):
        return []

    async def get_conversations(self, **kwargs):
        _StubService.conversations_kwargs = kwargs
        return _StubService.conversations

    async def delete_conversation(self, *, conversation_id):
        return conversation_id == "exists"


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


def _conversation(id, updated_at):
    return {
        "conversation_id": id,
        "title": f"title {id}",
        "created_at": "2026-09-01T10:00:00+00:00",
        "updated_at": updated_at,
    }


def test_conversations_requires_auth():
    assert client.get("/conversations").status_code == 401


def test_conversations_full_page_returns_a_cursor_to_the_next_one(monkeypatch):
    rows = [_conversation("a", "2026-09-03T10:00:00+00:00"), _conversation("b", "2026-09-02T10:00:00+00:00")]
    monkeypatch.setattr(_StubService, "conversations", rows)

    response = client.get("/conversations", params={"limit": 2}, headers={"x-api-key": "test-token"})

    body = response.json()
    assert [c["conversation_id"] for c in body["conversations"]] == ["a", "b"]
    assert body["conversations"][0]["title"] == "title a"
    assert body["next_cursor"] == "2026-09-02T10:00:00Z"
    assert _StubService.conversations_kwargs == {"limit": 2, "before": None}


def test_conversations_short_page_has_no_cursor_and_forwards_before(monkeypatch):
    monkeypatch.setattr(_StubService, "conversations", [_conversation("a", "2026-09-03T10:00:00+00:00")])

    response = client.get(
        "/conversations",
        params={"limit": 5, "before": "2026-09-04T10:00:00+00:00"},
        headers={"x-api-key": "test-token"},
    )

    assert response.json()["next_cursor"] is None
    assert _StubService.conversations_kwargs["before"].isoformat() == "2026-09-04T10:00:00+00:00"


def test_stream_still_ends_with_done_when_saving_the_reply_fails(monkeypatch, caplog):
    class _FailingSave(_StubService):
        saves = 0

        async def save_message(self, **kwargs):
            _FailingSave.saves += 1
            if _FailingSave.saves == 2:  
                raise RuntimeError("db is down")

    monkeypatch.setattr(app_module, "flow", _StubFlow([]))
    app.dependency_overrides[deps.get_service] = lambda: _FailingSave()
    try:
        response = client.post(
            "/stream/conversation",
            json={"conversation": "hi"},
            headers={"x-api-key": "test-token"},
        )
    finally:
        app.dependency_overrides[deps.get_service] = lambda: _StubService()

    assert response.status_code == 200
    assert _sse_events(response.text)["done"]["conversation"] == "answer"
    assert "Could not save the reply" in caplog.text
    assert "db is down" in caplog.text  