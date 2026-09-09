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

    def test_returns_the_panel_for_that_turn(self, client):
        turn = send(client, "hola")["turn"]
        panel = self.review(client, turn).get_data(as_text=True)
        assert "HOLA" in panel
        assert "try it another way" in panel

    def test_answers_with_the_panel_rather_than_its_ingredients(self, client):
        """The page used to be sent the data and build the panel itself, which
        meant two renderers that had to agree about the shape of a hint. When
        the shape changed, every already-open browser drew "[object Object]"."""
        response = self.review(client, send(client, "hola")["turn"])
        assert "text/html" in response.content_type
        assert response.get_data(as_text=True).lstrip().startswith("<")

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
        panel = self.review(client, send(client, "adiós")["turn"]).get_data(as_text=True)
        assert "HOLA" in panel and "ADIÓS" in panel

    def test_reviewing_a_turn_twice_does_not_file_it_twice(self, client):
        turn = send(client, "hola")["turn"]
        self.review(client, turn)
        panel = self.review(client, turn).get_data(as_text=True)
        assert panel.count("HOLA") == 1
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
        panel = self.review(client, send(client, "hola")["turn"]).get_data(as_text=True)
        assert "No learning points yet" in panel


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


class TestTheOneRenderer:
    """The panel is built in exactly one place now. These are the shapes it has
    to cope with, including ones written by older versions of the app."""

    def panel(self, corrections=(), hints=()):
        return app_module.render_learning_points(
            {"corrections": list(corrections), "hints": list(hints)}
        )

    def test_an_empty_conversation_says_so(self):
        assert "No learning points yet" in self.panel()

    def test_a_missing_conversation_is_not_a_crash(self):
        # The page asks on load, before anything has been said.
        assert "No learning points yet" in app_module.render_learning_points(None)

    def test_shows_a_hint_as_a_suggestion_and_a_reason(self):
        panel = self.panel(hints=[{"message": "hola", "timestamp": "t", "hints": [
            {"suggestion": "¿Qué tal?", "why": "Warmer between friends."}]}])
        assert "¿Qué tal?" in panel
        assert "Warmer between friends." in panel

    def test_still_renders_hints_saved_in_the_old_flat_shape(self):
        # Conversations in the database predate the split into two fields.
        panel = self.panel(hints=[{"message": "hola", "timestamp": "t",
                                   "hints": ["Try 'qué tal'"]}])
        assert "Try &#x27;qué tal&#x27;" in panel or "Try 'qué tal'" in panel
        assert "object Object" not in panel

    def test_escapes_what_the_learner_typed(self):
        panel = self.panel(corrections=[{"message": "<script>alert(1)</script>",
                                         "corrected": "x", "explanation": "y", "timestamp": "t"}])
        assert "<script>" not in panel
        assert "&lt;script&gt;" in panel

    def test_puts_the_newest_first(self):
        panel = self.panel(
            corrections=[{"message": "older", "corrected": "OLDER", "explanation": "",
                          "timestamp": "2026-01-01T00:00:00"}],
            hints=[{"message": "newer", "timestamp": "2026-06-01T00:00:00",
                    "hints": [{"suggestion": "s", "why": "w"}]}],
        )
        assert panel.index("newer") < panel.index("older")
