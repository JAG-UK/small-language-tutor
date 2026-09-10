"""The report card: what is counted, what is asked, and what is refused."""

import pytest

import report
from report import build, repeated_slips, word_changes


class FakeOllama:
    """Answers with whatever the test set, and records what it was asked."""

    model = "translategemma:12b"

    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []

    def chat_json(self, messages, schema):
        self.calls.append({"messages": messages, "schema": schema})
        return self.answer


@pytest.fixture
def model():
    return FakeOllama()


PROFILE = __import__("model_profiles").profile_for("translategemma:12b")


def corrections(*pairs):
    return [
        {"message": wrote, "corrected": correct, "timestamp": f"t{i}",
         "language": "es", "conversation_id": 1}
        for i, (wrote, correct) in enumerate(pairs)
    ]


ACCENTS = corrections(
    ("tengo dos anos", "tengo dos años"),
    ("hace tres anos", "hace tres años"),
    ("vivo en madrid", "vivo en Madrid"),
    ("me gusta el comida", "me gusta la comida"),
    ("estudio espanol", "estudio español"),
)


class TestWhatChanged:
    def test_finds_the_words_that_were_swapped(self):
        assert word_changes("vivo en madrid desde hace dos anos",
                            "vivo en Madrid desde hace dos años") == [
            ("madrid", "Madrid"), ("anos", "años")]

    def test_ignores_a_wholesale_rewrite(self):
        """A correction that replaces the sentence says something about that
        sentence and nothing countable about a habit."""
        assert word_changes("hola", "Buenos días, ¿cómo se encuentra usted hoy?") == []

    def test_an_unchanged_sentence_changed_nothing(self):
        assert word_changes("ya está bien", "ya está bien") == []


class TestWhatIsCounted:
    def test_a_word_put_right_twice_is_a_habit(self):
        slips = repeated_slips(ACCENTS)
        assert {"wrote": "anos", "should_be": "años", "times": 2} in slips

    def test_a_word_put_right_once_is_not(self):
        assert all(slip["wrote"] != "madrid" for slip in repeated_slips(ACCENTS))

    def test_commonest_first(self):
        many = corrections(*[("dos anos", "dos años")] * 3, ("el comida", "la comida"),
                           ("la comida", "el comida"))
        assert repeated_slips(many, minimum=1)[0]["times"] == 3


class TestAskingTheModel:
    def test_the_mistakes_go_in_as_pairs(self, model):
        build(model, PROFILE, ACCENTS, "es")
        asked = " ".join(m["content"] for m in model.calls[0]["messages"])
        assert "tengo dos anos" in asked and "tengo dos años" in asked

    def test_the_explanations_are_left_out(self, model):
        """They are the least reliable thing the critic produces, and a wrong
        one repeated here would be a wrong lesson with a heading on it."""
        with_prose = [dict(c, explanation="Utter nonsense about the subjunctive")
                      for c in ACCENTS]
        build(model, PROFILE, with_prose, "es")
        asked = " ".join(m["content"] for m in model.calls[0]["messages"])
        assert "subjunctive" not in asked

    def test_the_report_is_asked_for_in_english(self, model):
        build(model, PROFILE, ACCENTS, "es")
        asked = " ".join(m["content"] for m in model.calls[0]["messages"])
        assert "English" in asked and "Spanish" in asked


class TestThemesMustBeGroundedInRealMistakes:
    """Asked to find patterns, a model will illustrate one with a plausible
    mistake that nobody made. A report card that invents your mistakes is worse
    than no report card."""

    def theme(self, **overrides):
        theme = {"area": "Missing accents", "what_happens": "Accents get dropped.",
                 "examples": ["anos"], "practise": "Learn the ñ key."}
        theme.update(overrides)
        return theme

    def test_a_theme_drawn_from_real_mistakes_is_kept(self, model):
        model.answer = {"themes": [self.theme()]}
        report_card = build(model, PROFILE, ACCENTS, "es")
        assert report_card["themes"][0]["area"] == "Missing accents"

    def test_an_invented_example_is_dropped(self, model):
        model.answer = {"themes": [self.theme(examples=["anos", "subjuntivo imperfecto"])]}
        assert build(model, PROFILE, ACCENTS, "es")["themes"][0]["examples"] == ["anos"]

    def test_a_theme_with_nothing_real_behind_it_goes(self, model):
        model.answer = {"themes": [self.theme(examples=["haber estado cantando"])]}
        report_card = build(model, PROFILE, ACCENTS, "es")
        assert report_card["themes"] == []
        assert "Nothing stood out" in report_card["note"]

    def test_an_example_is_matched_past_its_accents(self, model):
        # The quote and the original differ by exactly the thing being taught.
        model.answer = {"themes": [self.theme(examples=["años"])]}
        assert build(model, PROFILE, ACCENTS, "es")["themes"][0]["examples"] == ["años"]

    def test_more_themes_than_asked_for_are_trimmed(self, model):
        model.answer = {"themes": [self.theme(area=f"Area {i}") for i in range(9)]}
        assert len(build(model, PROFILE, ACCENTS, "es")["themes"]) == report.MAX_THEMES


class TestWhenThereIsNotEnoughToGoOn:
    def test_too_few_corrections_is_not_worth_a_model_call(self, model):
        report_card = build(model, PROFILE, ACCENTS[:2], "es")
        assert model.calls == []
        assert "Keep talking" in report_card["note"]

    def test_but_the_counting_still_happens(self, model):
        """The tally and the repeats are arithmetic. They are true whether or
        not anything is answering."""
        report_card = build(model, PROFILE, ACCENTS[:2], "es")
        assert report_card["total"] == 2

    def test_no_corrections_at_all(self, model):
        report_card = build(model, PROFILE, [], "es")
        assert report_card["total"] == 0 and report_card["themes"] == []

    def test_a_model_that_does_not_answer_leaves_the_counting_intact(self, model):
        model.answer = None
        report_card = build(model, PROFILE, ACCENTS, "es")
        assert report_card["themes"] == []
        assert report_card["repeats"], "the repeats are counted, not asked for"
        assert "Could not" in report_card["note"]

    def test_counts_the_conversations_it_drew_on(self, model):
        spread = [dict(c, conversation_id=i % 3) for i, c in enumerate(ACCENTS)]
        assert build(model, PROFILE, spread, "es")["conversations"] == 3
