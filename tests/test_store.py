"""Keeping conversations, and getting them back."""

import json

import pytest
from sqlalchemy import create_engine, text

import store
from models import Base, Conversation, Session, ensure_schema


def conversation(**overrides):
    conv = store.new_conversation("es", "friendly")
    conv["messages"] = [
        {"role": "user", "content": "hola", "timestamp": "t"},
        {"role": "assistant", "content": "¡Hola!", "timestamp": "t"},
    ]
    conv.update(overrides)
    return conv


class TestSaving:
    def test_a_saved_conversation_gets_an_id(self):
        conv = conversation()
        assert conv["id"] is None
        assert store.save(conv) == conv["id"] is not None

    def test_saving_again_updates_rather_than_duplicates(self):
        # The old Save button inserted every time, so pressing it twice left two
        # copies of the same conversation.
        conv = conversation()
        store.save(conv)
        conv["messages"].append({"role": "user", "content": "adiós", "timestamp": "t"})
        store.save(conv)

        assert len(store.recent()) == 1
        assert store.load(conv["id"])["messages"][-1]["content"] == "adiós"

    def test_the_tone_survives(self):
        # It was not stored at all, so a flirty conversation reopened as friendly.
        conv = conversation(tone="flirty")
        store.save(conv)
        assert store.load(conv["id"])["tone"] == "flirty"

    def test_learning_points_survive_in_their_current_shape(self):
        conv = conversation()
        conv["corrections"] = [{"message": "hola", "corrected": "¡Hola!",
                                "explanation": "punctuation", "timestamp": "t"}]
        conv["hints"] = [{"message": "hola", "timestamp": "t",
                          "hints": [{"suggestion": "¿Qué tal?", "why": "Warmer."}]}]
        store.save(conv)

        back = store.load(conv["id"])
        assert back["corrections"][0]["corrected"] == "¡Hola!"
        assert back["hints"][0]["hints"][0]["why"] == "Warmer."

    def test_stored_as_json_rather_than_json_inside_json(self):
        """The columns are JSON, so SQLAlchemy encodes them. The old code
        encoded them again first, and only round-tripped because the read
        decoded twice to match."""
        conv = conversation()
        store.save(conv)

        session = Session()
        raw = session.execute(
            text("SELECT messages FROM conversations WHERE id = :id"), {"id": conv["id"]}
        ).scalar()
        session.close()

        assert isinstance(json.loads(raw), list)  # not a string that needs decoding again


class TestTitles:
    def test_the_first_thing_the_learner_said(self):
        assert store.title_for(conversation()) == "hola"

    def test_a_long_opening_is_cut_at_a_word(self):
        conv = conversation(messages=[{"role": "user", "timestamp": "t", "content":
                                       "hola " * 40}])
        title = store.title_for(conv)
        assert len(title) <= store.TITLE_LENGTH + 1
        assert title.endswith("…")
        assert not title.rstrip("…").endswith("hol")  # not chopped mid-word

    def test_what_the_model_said_first_is_not_the_title(self):
        conv = conversation(messages=[
            {"role": "assistant", "content": "¡Bienvenido!", "timestamp": "t"},
            {"role": "user", "content": "gracias", "timestamp": "t"},
        ])
        assert store.title_for(conv) == "gracias"

    def test_a_conversation_with_nothing_in_it(self):
        assert store.title_for(store.new_conversation("es", "friendly")) == "Untitled"


class TestLoading:
    def test_an_unknown_id_is_none_rather_than_an_error(self):
        assert store.load(9999) is None

    def test_everything_already_said_counts_as_reviewed(self):
        """Reopening should not re-critique the whole history: that is a model
        call per turn, and it would file every learning point a second time."""
        conv = conversation()
        conv["messages"].append({"role": "user", "content": "adiós", "timestamp": "t"})
        store.save(conv)

        back = store.load(conv["id"])
        assert back["reviewed"] == {0, 2}  # the learner's turns, not the model's

    def test_reads_rows_written_before_the_encoding_was_fixed(self):
        # A JSON string inside the JSON column, as the old code left it.
        session = Session()
        row = Conversation(title="old", language="es", tone="friendly",
                           messages=json.dumps([{"role": "user", "content": "viejo"}]),
                           corrections=json.dumps([]), hints=json.dumps([]))
        session.add(row)
        session.commit()
        conv_id = row.id
        session.close()

        assert store.load(conv_id)["messages"][0]["content"] == "viejo"


class TestListing:
    def test_most_recently_touched_first(self):
        first, second = conversation(), conversation()
        store.save(first)
        store.save(second)
        store.save(first)  # touching the older one moves it up

        assert [row["id"] for row in store.recent()] == [first["id"], second["id"]]

    def test_counts_the_messages(self):
        conv = conversation()
        store.save(conv)
        assert store.recent()[0]["message_count"] == 2

    def test_nothing_saved_is_an_empty_list(self):
        assert store.recent() == []


class TestDeleting:
    def test_removes_it(self):
        conv = conversation()
        store.save(conv)
        assert store.delete(conv["id"]) is True
        assert store.load(conv["id"]) is None

    def test_deleting_what_is_not_there_says_so(self):
        assert store.delete(9999) is False


class TestTheSchemaCatchesUp:
    """`hints` was added to the model after the database existed. create_all()
    creates missing tables and nothing else, so the column was simply absent and
    every save failed with `no such column: conversations.hints`."""

    def test_a_column_added_since_the_database_was_made_is_added_to_it(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path}/old.db")
        with engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE conversations (id INTEGER PRIMARY KEY, title VARCHAR(200))"))

        added = ensure_schema(engine)

        assert "hints" in added and "tone" in added
        with engine.begin() as connection:
            # The query that used to fail.
            connection.execute(text("SELECT hints, tone FROM conversations"))

    def test_running_it_again_changes_nothing(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path}/new.db")
        Base.metadata.create_all(engine)
        assert ensure_schema(engine) == []

    def test_a_database_that_does_not_exist_yet_is_left_to_create_all(self, tmp_path):
        engine = create_engine(f"sqlite:///{tmp_path}/empty.db")
        assert ensure_schema(engine) == []
