"""The Ollama wrapper: what it sends, and how it fails."""

import json

import pytest
import requests

from ollama_client import OllamaClient, OllamaError


class FakeResponse:
    def __init__(self, content="", status_code=200):
        self.status_code = status_code
        self._content = content

    def json(self):
        return {"message": {"content": self._content}}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")


@pytest.fixture
def sent(monkeypatch):
    """Capture the payload rather than talking to Ollama."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "payload": json, "timeout": timeout})
        return FakeResponse(calls[0].get("reply", '{"ok": true}'))

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def client(**kw):
    return OllamaClient(**{"model": "phi4-mini:3.8b", **kw})


def reply_with(monkeypatch, content, status_code=200):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse(content, status_code)
    )


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


class TestWhatItSends:
    def test_posts_to_the_chat_endpoint_of_the_configured_host(self, sent):
        client(base_url="http://gpu-box:11434/").chat([{"role": "user", "content": "hola"}])
        assert sent[0]["url"] == "http://gpu-box:11434/api/chat"  # trailing slash trimmed

    def test_sends_only_role_and_content(self, sent):
        # Messages carry a timestamp internally; Ollama rejects unknown keys.
        client().chat([{"role": "user", "content": "hola", "timestamp": "2026-01-01"}])
        assert sent[0]["payload"]["messages"] == [{"role": "user", "content": "hola"}]

    def test_asks_for_one_whole_answer_rather_than_a_stream(self, sent):
        client().chat([{"role": "user", "content": "hola"}])
        assert sent[0]["payload"]["stream"] is False

    def test_a_plain_chat_asks_for_no_particular_format(self, sent):
        client().chat([{"role": "user", "content": "hola"}])
        assert "format" not in sent[0]["payload"]

    def test_a_json_chat_sends_the_schema(self, sent):
        # This is what stops a small model wrapping its answer in prose.
        client().chat_json([{"role": "user", "content": "hola"}], SCHEMA)
        assert sent[0]["payload"]["format"] == SCHEMA


class TestJsonReplies:
    def test_parses_the_answer(self, monkeypatch):
        reply_with(monkeypatch, json.dumps({"ok": True}))
        assert client().chat_json([], SCHEMA) == {"ok": True}

    def test_gives_up_rather_than_guessing_when_the_answer_is_not_json(self, monkeypatch):
        reply_with(monkeypatch, "I think the sentence is fine!")
        assert client().chat_json([], SCHEMA) is None

    def test_gives_up_when_the_answer_is_json_but_not_an_object(self, monkeypatch):
        reply_with(monkeypatch, "[1, 2, 3]")
        assert client().chat_json([], SCHEMA) is None

    def test_gives_up_when_the_server_fails(self, monkeypatch):
        reply_with(monkeypatch, "", status_code=500)
        assert client().chat_json([], SCHEMA) is None

    def test_gives_up_when_the_server_cannot_be_reached(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "post", boom)
        assert client().chat_json([], SCHEMA) is None


class TestFailingPlainly:
    def test_a_missing_model_says_which_one_and_how_to_get_it(self, monkeypatch):
        reply_with(monkeypatch, "", status_code=404)
        answer = client(model="not-pulled:1b").chat([])
        assert "not-pulled:1b" in answer
        assert "ollama pull not-pulled:1b" in answer

    def test_a_timeout_suggests_what_to_do_about_it(self, monkeypatch):
        def slow(*a, **k):
            raise requests.exceptions.Timeout()

        monkeypatch.setattr(requests, "post", slow)
        answer = client(timeout=5).chat([])
        assert "5s" in answer
        assert "smaller model" in answer

    def test_conversation_errors_come_back_as_text_not_exceptions(self, monkeypatch):
        # An exception here would lose the message the learner just typed.
        reply_with(monkeypatch, "", status_code=500)
        assert client().chat([]).startswith("Error:")

    def test_a_missing_model_raises_inside_the_json_path(self, monkeypatch):
        reply_with(monkeypatch, "", status_code=404)
        with pytest.raises(OllamaError):
            client(model="not-pulled:1b")._post([])


class TestListingModels:
    def test_reports_what_is_pulled(self, monkeypatch):
        class Tags:
            status_code = 200

            def json(self):
                return {"models": [{"name": "phi4-mini:3.8b"}, {"name": "qwen3:4b"}]}

            def raise_for_status(self):
                pass

        monkeypatch.setattr(requests, "get", lambda *a, **k: Tags())
        assert client().available_models() == ["phi4-mini:3.8b", "qwen3:4b"]

    def test_an_unreachable_server_lists_nothing_rather_than_raising(self, monkeypatch):
        def boom(*a, **k):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", boom)
        assert client().available_models() == []
