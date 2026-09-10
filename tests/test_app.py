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


class TestExplanationsThatAreReallyLists:
    """Asked what changed, a model answers with a markdown list: a lead-in, then
    one point per line, each opening with a bullet. HTML collapses newlines, so
    it all arrived as one run-on paragraph with stray asterisks in it."""

    LIST = (
        "Here's a breakdown of the changes:\n"
        "\n"
        "*   \"ameliorar\" changed to \"mejorar\": a French word.\n"
        "*   \"un buen idea\" changed to \"una buena idea\": idea is feminine.\n"
    )

    def test_keeps_the_lines_and_drops_the_bullets(self):
        assert app_module.explanation_lines(self.LIST) == [
            "Here's a breakdown of the changes:",
            '"ameliorar" changed to "mejorar": a French word.',
            '"un buen idea" changed to "una buena idea": idea is feminine.',
        ]

    @pytest.mark.parametrize("marker", ["-", "*", "•", "*  "])
    def test_recognises_the_bullets_models_actually_use(self, marker):
        text = f"{marker} first point.\n{marker} second point."
        assert app_module.explanation_lines(text) == ["first point.", "second point."]

    def test_drops_a_bullet_the_model_opened_and_abandoned(self):
        assert app_module.explanation_lines("- a real point.\n-\n") == ["a real point."]

    @pytest.mark.parametrize(
        "text",
        [
            "The word needs an accent.",
            "A well-known change from A -> B.",
            # One sentence with a dash in it is not a two-item list.
            "Changed 'tu' to 'tú' - the pronoun takes an accent.",
        ],
    )
    def test_leaves_a_single_sentence_alone(self, text):
        assert app_module.explanation_lines(text) == [text]

    def test_nothing_to_say_is_no_lines_at_all(self):
        assert app_module.explanation_lines("") == []
        assert app_module.explanation_lines(None) == []

    def test_the_panel_puts_each_point_on_its_own_line(self):
        panel = app_module.render_learning_points({"corrections": [{
            "message": "m", "corrected": "c", "timestamp": "t", "explanation": self.LIST,
        }], "hints": []})
        assert "a French word.<br>" in panel
        assert "*" not in panel.split("explanation")[1][:400]  # no bullets left in the markup


class TestConversationsSurviveTheServer:
    """They used to live only in memory, and the Save button that was meant to
    rescue them returned 400. Every turn is written through now."""

    def test_the_reply_says_which_conversation_it_belongs_to(self, client):
        """The page stores this and reopens it on the next load. Without it a
        refresh left you on an empty screen with the conversation saved but
        nowhere in sight."""
        first = send(client, "hola")
        assert first["conversation_id"] is not None
        assert send(client, "adiós")["conversation_id"] == first["conversation_id"]

    def test_a_turn_is_saved_as_it_happens(self, client):
        send(client, "hola")
        import store
        assert [row["title"] for row in store.recent()] == ["hola"]

    def test_the_whole_exchange_is_there_without_anyone_pressing_anything(self, client):
        send(client, "hola")
        send(client, "adiós")
        import store
        assert store.recent()[0]["message_count"] == 4  # two turns each way

    def test_a_conversation_is_one_row_however_long_it_runs(self, client):
        for message in ("hola", "adiós", "hasta luego"):
            send(client, message)
        import store
        assert len(store.recent()) == 1

    def test_learning_points_are_saved_with_it(self, client):
        turn = send(client, "hola")["turn"]
        client.post("/api/review", json={"session_id": "s1", "turn": turn})
        import store
        conv_id = store.recent()[0]["id"]
        assert store.load(conv_id)["corrections"][0]["corrected"] == "HOLA"

    def test_reopening_after_the_server_forgets_everything(self, client):
        send(client, "hola")
        import store
        conv_id = store.recent()[0]["id"]

        app_module.conversations.clear()  # what a restart looks like

        opened = client.post(f"/api/conversations/{conv_id}/open",
                             json={"session_id": "fresh"}).get_json()
        assert [m["content"] for m in opened["messages"]] == ["hola", "¡Claro!"]

    def test_and_the_conversation_can_then_be_continued(self, client):
        """The point of putting it back on the server: the partner reads its
        history from there, so a transcript alone would only be something to
        look at."""
        send(client, "vivo en Madrid")
        import store
        conv_id = store.recent()[0]["id"]
        app_module.conversations.clear()

        client.post(f"/api/conversations/{conv_id}/open", json={"session_id": "fresh"})
        client.calls["chat"].clear()
        send(client, "¿y tú?", session="fresh")

        sent = [m["content"] for m in client.calls["chat"][0]]
        assert any("vivo en Madrid" in content for content in sent)

    def test_reopening_does_not_critique_the_whole_history_again(self, client):
        turn = send(client, "hola")["turn"]
        client.post("/api/review", json={"session_id": "s1", "turn": turn})
        import store
        conv_id = store.recent()[0]["id"]
        app_module.conversations.clear()
        client.calls["check"].clear()

        client.post(f"/api/conversations/{conv_id}/open", json={"session_id": "fresh"})
        client.post("/api/review", json={"session_id": "fresh", "turn": turn})

        assert client.calls["check"] == []

    def test_the_language_and_tone_come_back_too(self, client):
        send(client, "salut", language="fr", tone="professional")
        import store
        conv_id = store.recent()[0]["id"]
        app_module.conversations.clear()

        opened = client.post(f"/api/conversations/{conv_id}/open",
                             json={"session_id": "fresh"}).get_json()
        assert (opened["language"], opened["tone"]) == ("fr", "professional")

    def test_opening_something_that_is_not_there(self, client):
        assert client.post("/api/conversations/9999/open",
                           json={"session_id": "fresh"}).status_code == 404


class TestTheConversationList:
    def test_nothing_saved_says_so(self, client):
        assert "Nothing saved yet" in client.get("/api/conversations").get_data(as_text=True)

    def test_shows_what_was_said_and_when(self, client):
        send(client, "hola amigo")
        listing = client.get("/api/conversations").get_data(as_text=True)
        assert "hola amigo" in listing
        assert "2 messages" in listing

    def test_escapes_the_title(self, client):
        send(client, "<script>alert(1)</script>")
        listing = client.get("/api/conversations").get_data(as_text=True)
        assert "<script>" not in listing
        assert "&lt;script&gt;" in listing

    def test_deleting_one_removes_it(self, client):
        send(client, "hola")
        import store
        conv_id = store.recent()[0]["id"]

        assert client.delete(f"/api/conversations/{conv_id}").status_code == 200
        assert store.recent() == []

    def test_deleting_it_also_forgets_it_in_memory(self, client):
        """Otherwise the page still holding that session would write the deleted
        conversation straight back on its next turn."""
        send(client, "hola")
        import store
        conv_id = store.recent()[0]["id"]

        client.delete(f"/api/conversations/{conv_id}")
        send(client, "otra vez")

        assert [row["title"] for row in store.recent()] == ["otra vez"]

    def test_deleting_what_is_not_there(self, client):
        assert client.delete("/api/conversations/9999").status_code == 404


class TestWhichModelsItUses:
    """The critic is a different model by default. phi4-mini:3.8b caught 11 of
    21 errors where translategemma:12b caught 20, and an app whose point is
    catching mistakes should not ship missing half of them to save a pull."""

    def teardown_method(self):
        app_module.use_critic(app_module.DEFAULT_CRITIC)

    def test_the_critic_is_not_the_conversation_model_by_default(self):
        assert app_module.DEFAULT_CRITIC != app_module.MODEL

    def test_nothing_is_said_when_both_are_pulled(self):
        assert app_module.check_models([app_module.MODEL, app_module.DEFAULT_CRITIC]) == []

    def test_a_missing_conversation_model_says_what_to_type(self):
        notes = app_module.check_models([app_module.DEFAULT_CRITIC])
        assert any(f"ollama pull {app_module.MODEL}" in note for note in notes)

    def test_a_missing_critic_falls_back_rather_than_marking_nothing(self):
        notes = app_module.check_models([app_module.MODEL])

        assert any("fall back" in note for note in notes)
        assert app_module.CRITIC_MODEL == app_module.MODEL
        assert app_module.grammar_checker.ollama is app_module.ollama

    def test_the_fallback_is_a_working_checker_not_a_stub(self):
        app_module.check_models([app_module.MODEL])
        assert hasattr(app_module.grammar_checker, "check_message")

    def test_an_ollama_that_answers_nothing_is_left_alone(self):
        # Not running yet, or still starting. The first real request will say so
        # far better than a guess made here.
        assert app_module.check_models([]) == []
        assert app_module.CRITIC_MODEL == app_module.DEFAULT_CRITIC

    def test_choosing_the_conversation_model_as_critic_shares_one_client(self):
        # One model in memory, for a machine that cannot hold two.
        app_module.use_critic(app_module.MODEL)
        assert app_module.critic is app_module.ollama
