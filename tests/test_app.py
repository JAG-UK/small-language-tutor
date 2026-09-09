"""The two halves of a turn: the reply, and the critique that follows it."""

import pytest

import app as app_module


@pytest.fixture
def client(monkeypatch):
    """A test client whose models are stand-ins that record what they were asked."""
    calls = {"chat": [], "check": [], "hints": []}

    def fake_chat(messages, language="en"):
        calls["chat"].append(messages)
        return "¡Claro!"

    def fake_check(message, context, language, tone="friendly"):
        calls["check"].append({"message": message, "context": list(context),
                               "language": language, "tone": tone})
        return {"has_errors": True, "corrected": message.upper(), "explanation": "capitals"}

    def fake_hints(message, context, language, tone="friendly"):
        calls["hints"].append({"message": message, "context": list(context),
                               "language": language, "tone": tone})
        return {"has_hints": True, "hints": ["try it another way"]}

    monkeypatch.setattr(app_module.ollama, "chat", fake_chat)
    monkeypatch.setattr(app_module.grammar_checker, "check_message", fake_check)
    monkeypatch.setattr(app_module.grammar_checker, "get_hints", fake_hints)
    app_module.conversations.clear()

    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        c.calls = calls
        yield c


def send(client, message, session="s1", language="es", tone="friendly"):
    return client.post("/api/chat", json={"session_id": session, "message": message,
                                          "language": language, "tone": tone}).get_json()


class TestTheReplyComesBackAlone:
    def test_answers_with_the_reply(self, client):
        assert send(client, "hola")["response"] == "¡Claro!"

    def test_costs_exactly_one_model_call(self, client):
        # The point of the split: three round trips used to stand between the
        # learner pressing Send and seeing a word back.
        send(client, "hola")
        assert len(client.calls["chat"]) == 1
        assert client.calls["check"] == []
        assert client.calls["hints"] == []

    def test_hands_back_the_turn_to_be_reviewed(self, client):
        assert send(client, "hola")["turn"] == 0
        assert send(client, "adiós")["turn"] == 2  # user, assistant, user

    def test_keeps_the_reply_in_the_transcript(self, client):
        messages = send(client, "hola")["messages"]
        assert [m["content"] for m in messages] == ["hola", "¡Claro!"]


class TestTheCritiqueThatFollows:
    def review(self, client, turn, session="s1"):
        return client.post("/api/review", json={"session_id": session, "turn": turn})

    def test_returns_the_learning_points_for_that_turn(self, client):
        turn = send(client, "hola")["turn"]
        body = self.review(client, turn).get_json()
        assert body["correction"]["corrected"] == "HOLA"
        assert body["hints"]["hints"] == ["try it another way"]

    def test_judges_the_named_turn_not_the_latest_one(self, client):
        first = send(client, "hola")["turn"]
        send(client, "adiós")
        self.review(client, first)
        assert client.calls["check"][0]["message"] == "hola"

    def test_shows_the_critic_what_came_before_but_not_what_came_after(self, client):
        # The reply the model gave is not evidence about the sentence that
        # prompted it, and would push that sentence out of view.
        send(client, "hola")
        turn = send(client, "vivo en madrid")["turn"]
        self.review(client, turn)
        context = [m["content"] for m in client.calls["check"][0]["context"]]
        assert context == ["hola", "¡Claro!", "vivo en madrid"]

    def test_accumulates_across_the_conversation(self, client):
        self.review(client, send(client, "hola")["turn"])
        body = self.review(client, send(client, "adiós")["turn"]).get_json()
        assert [c["message"] for c in body["all_corrections"]] == ["hola", "adiós"]

    def test_reviewing_a_turn_twice_does_not_file_it_twice(self, client):
        turn = send(client, "hola")["turn"]
        self.review(client, turn)
        body = self.review(client, turn).get_json()
        assert len(body["all_corrections"]) == 1
        assert len(client.calls["check"]) == 1

    def test_uses_the_language_and_tone_the_turn_was_sent_with(self, client):
        turn = send(client, "salut", language="fr", tone="professional")["turn"]
        self.review(client, turn)
        assert client.calls["check"][0]["language"] == "fr"
        assert client.calls["check"][0]["tone"] == "professional"

    def test_a_quiet_critic_files_nothing(self, client, monkeypatch):
        monkeypatch.setattr(app_module.grammar_checker, "check_message",
                            lambda *a, **k: {"has_errors": False, "corrected": "hola",
                                             "explanation": ""})
        monkeypatch.setattr(app_module.grammar_checker, "get_hints",
                            lambda *a, **k: {"has_hints": False, "hints": []})
        body = self.review(client, send(client, "hola")["turn"]).get_json()
        assert body["correction"] is None and body["hints"] is None
        assert body["all_corrections"] == [] and body["all_hints"] == []


class TestWhenThereIsNothingToReview:
    """Conversations live in memory. A restart loses them, and the client will
    still ask about a turn that is gone."""

    def test_an_unknown_session_is_refused_not_crashed(self, client):
        assert client.post("/api/review", json={"session_id": "gone", "turn": 0}).status_code == 404

    def test_a_turn_past_the_end_is_refused(self, client):
        send(client, "hola")
        assert client.post("/api/review", json={"session_id": "s1", "turn": 99}).status_code == 404

    @pytest.mark.parametrize("turn", [None, "0", -1, 1.5])
    def test_a_turn_that_is_not_an_index_is_refused(self, client, turn):
        send(client, "hola")
        assert client.post("/api/review", json={"session_id": "s1", "turn": turn}).status_code == 404

    def test_the_model_s_own_turn_is_not_the_learner_s_to_answer_for(self, client):
        send(client, "hola")  # turn 1 is the reply
        assert client.post("/api/review", json={"session_id": "s1", "turn": 1}).status_code == 400
