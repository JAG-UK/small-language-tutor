"""Corrections and hints: what gets asked for, and what happens when the model
does not cooperate."""

import pytest

from grammar_checker import CORRECTION_SCHEMA, HINTS_SCHEMA, GrammarChecker


class FakeOllama:
    """Answers with whatever the test set, and records what it was asked."""

    def __init__(self, answer=None):
        self.answer = answer
        self.calls = []

    def chat_json(self, messages, schema):
        self.answer = self.answer  # keep the attribute obvious
        self.calls.append({"messages": messages, "schema": schema})
        return self.answer() if callable(self.answer) else self.answer


@pytest.fixture
def model():
    return FakeOllama()


def checker(model):
    return GrammarChecker(model)


class TestCorrections:
    def test_asks_with_a_schema_so_the_answer_cannot_be_prose(self, model):
        checker(model).check_message("hola", [], "Spanish")
        assert model.calls[0]["schema"] == CORRECTION_SCHEMA

    def test_puts_the_learner_message_in_front_of_the_model(self, model):
        checker(model).check_message("yo tengo 20 anos", [], "Spanish")
        user_turn = model.calls[0]["messages"][-1]["content"]
        assert "yo tengo 20 anos" in user_turn

    def test_tells_the_model_the_language_and_the_tone(self, model):
        checker(model).check_message("hola", [], "Spanish", tone="flirty")
        system = model.calls[0]["messages"][0]["content"]
        assert "Spanish" in system
        assert "flirty" in system

    def test_returns_the_correction(self, model):
        model.answer = {
            "has_errors": True,
            "corrected": "Yo tengo 20 años.",
            "explanation": "missing accent on 'anos' should be 'años'",
        }
        result = checker(model).check_message("yo tengo 20 anos", [], "Spanish")

        assert result["has_errors"] is True
        assert result["corrected"] == "Yo tengo 20 años."
        assert "años" in result["explanation"]

    def test_reports_a_clean_sentence_as_clean(self, model):
        model.answer = {"has_errors": False, "corrected": "Hola.", "explanation": ""}
        assert checker(model).check_message("Hola.", [], "Spanish")["has_errors"] is False


class TestTrustingTheTextNotTheFlag:
    """Small models produce a good correction and then misreport whether they
    made one. phi4-mini:3.8b rewrites "yo tengo 20 anos y vivo en madrid" with
    the accent and the capital fixed, and says has_errors: false in the same
    answer. The flag is therefore derived from the text."""

    def test_a_changed_sentence_is_an_error_even_if_the_model_denies_it(self, model):
        model.answer = {
            "has_errors": False,
            "corrected": "yo tengo 20 años y vivo en Madrid",
            "explanation": "The original message is correct. No changes were made.",
        }
        result = checker(model).check_message("yo tengo 20 anos y vivo en madrid", [], "Spanish")
        assert result["has_errors"] is True
        assert result["corrected"] == "yo tengo 20 años y vivo en Madrid"

    def test_an_unchanged_sentence_is_not_an_error_even_if_the_model_claims_it(self, model):
        model.answer = {"has_errors": True, "corrected": "Hola.", "explanation": "Fixed it."}
        assert checker(model).check_message("Hola.", [], "Spanish")["has_errors"] is False

    def test_an_added_accent_counts(self, model):
        model.answer = {"has_errors": False, "corrected": "años", "explanation": ""}
        assert checker(model).check_message("anos", [], "Spanish")["has_errors"] is True

    def test_a_capital_letter_counts(self, model):
        model.answer = {"has_errors": False, "corrected": "vivo en Madrid", "explanation": ""}
        assert checker(model).check_message("vivo en madrid", [], "Spanish")["has_errors"] is True

    def test_reflowed_whitespace_does_not_count(self, model):
        # Models reflow freely; that is not something to teach a learner.
        model.answer = {"has_errors": True, "corrected": "  Hola   amigo ", "explanation": ""}
        assert checker(model).check_message("Hola amigo", [], "Spanish")["has_errors"] is False

    def test_keeps_the_original_when_the_model_offers_no_correction(self, model):
        model.answer = {"has_errors": True, "corrected": "", "explanation": "something"}
        result = checker(model).check_message("mi mensaje", [], "Spanish")
        assert result["corrected"] == "mi mensaje"


class TestTidyingUpIsNotAnError:
    """Every model tried here answers a perfectly good chat message by adding a
    capital and a full stop. Reported as errors, those fill the learning panel
    with remarks about punctuation nobody uses when chatting."""

    def correct(self, model, corrected, original):
        model.answer = {"has_errors": True, "corrected": corrected, "explanation": "e"}
        return checker(model).check_message(original, [], "Spanish")

    @pytest.mark.parametrize(
        "corrected,original",
        [
            ("Ayer fui a la playa.", "ayer fui a la playa"),
            ("Ayer fui a la playa", "ayer fui a la playa"),
            ("ayer fui a la playa.", "ayer fui a la playa"),
            ("¿Cómo estás?", "¿Cómo estás"),
        ],
    )
    def test_a_capital_and_a_full_stop_are_not_worth_a_learning_point(
        self, model, corrected, original
    ):
        assert self.correct(model, corrected, original)["has_errors"] is False

    @pytest.mark.parametrize(
        "corrected,original",
        [
            # A capital inside the sentence is a proper noun, which is the kind
            # of thing this app exists to catch.
            ("vivo en Madrid", "vivo en madrid"),
            ("vivo en Madrid desde hace dos años", "vivo en madrid desde hace dos anos"),
            # A missing accent changes the word, not its presentation.
            ("Tú eres", "Tu eres"),
            # Punctuation that carries meaning rather than tidiness.
            ("¿Dónde vives?", "Dónde vives"),
        ],
    )
    def test_anything_further_in_still_counts(self, model, corrected, original):
        assert self.correct(model, corrected, original)["has_errors"] is True

    def test_a_one_word_message_loses_its_proper_noun_to_this(self, model):
        """The edge of the rule, recorded rather than hidden: in "madrid" the
        capital is both the proper noun and the first letter of the sentence,
        and they cannot be told apart, so nothing is said. The trade is worth
        it — a full stop gets added to almost every good sentence, whereas a
        message that is a bare proper noun and nothing else is rare."""
        assert self.correct(model, "Madrid", "madrid")["has_errors"] is False


class TestCommentaryIsNotACorrection:
    """Asked to correct "vivo en madrid desde hace dos anos", phi4-mini:3.8b
    variously answered "Your message is correct.", "No se realizaron
    correcciones" and "Mi mensaje estaba bien escrito" — commentary in the
    field meant for the sentence. Shown to a learner as their corrected words,
    that is worse than showing nothing."""

    ORIGINAL = "vivo en madrid desde hace dos anos"

    @pytest.mark.parametrize(
        "commentary",
        [
            "Your message is correct.",
            "No se realizaron correcciones",
            "Mi mensaje estaba bien escrito, no se hicieron cambios.",
            "The sentence has no errors.",
        ],
    )
    def test_commentary_is_refused(self, model, commentary):
        model.answer = {"has_errors": True, "corrected": commentary, "explanation": "..."}
        result = checker(model).check_message(self.ORIGINAL, [], "Spanish")

        assert result["has_errors"] is False
        assert result["corrected"] == self.ORIGINAL  # never the commentary

    def test_a_real_correction_still_gets_through(self, model):
        model.answer = {
            "has_errors": False,
            "corrected": "Vivo en Madrid desde hace dos años",
            "explanation": "accents and capitals",
        }
        result = checker(model).check_message(self.ORIGINAL, [], "Spanish")

        assert result["has_errors"] is True
        assert result["corrected"] == "Vivo en Madrid desde hace dos años"

    def test_a_heavy_but_genuine_rewrite_still_gets_through(self, model):
        model.answer = {
            "has_errors": True,
            "corrected": "Hace dos años que vivo en Madrid",
            "explanation": "reordered for naturalness",
        }
        assert checker(model).check_message(self.ORIGINAL, [], "Spanish")["has_errors"] is True

    def test_commentary_that_quotes_the_sentence_back_is_refused(self, model):
        # Shares every word with the original, so vocabulary alone lets it
        # through; it is twice the length, which is what gives it away.
        model.answer = {
            "has_errors": True,
            "corrected": "The message you provided, 'Vivo en Madrid desde hace dos anos', is correct.",
            "explanation": "...",
        }
        result = checker(model).check_message(self.ORIGINAL, [], "Spanish")
        assert result["has_errors"] is False
        assert result["corrected"] == self.ORIGINAL

    def test_a_correction_that_adds_a_missing_word_still_gets_through(self, model):
        model.answer = {"has_errors": True, "corrected": "Yo vivo en Madrid", "explanation": "..."}
        assert checker(model).check_message("vivo en madrid", [], "Spanish")["has_errors"] is True

    def test_an_empty_correction_is_refused(self, model):
        model.answer = {"has_errors": True, "corrected": "   ", "explanation": "..."}
        assert checker(model).check_message(self.ORIGINAL, [], "Spanish")["corrected"] == self.ORIGINAL


class TestWhenTheModelFails:
    """A failed check must never look like a passed one, and must never invent
    a correction — teaching the wrong thing is worse than teaching nothing."""

    def test_a_failed_call_leaves_the_message_alone(self, model):
        model.answer = None
        result = checker(model).check_message("yo tengo 20 anos", [], "Spanish")

        assert result["has_errors"] is False
        assert result["corrected"] == "yo tengo 20 anos"
        assert "Could not check" in result["explanation"]

    def test_a_failed_hints_call_offers_no_hints(self, model):
        model.answer = None
        assert checker(model).get_hints("hola", [], "Spanish") == {
            "has_hints": False,
            "hints": [],
        }


class TestHints:
    def test_asks_with_a_schema(self, model):
        checker(model).get_hints("hola", [], "Spanish")
        assert model.calls[0]["schema"] == HINTS_SCHEMA

    def test_returns_the_hints(self, model):
        model.answer = {"has_hints": True, "hints": ["Try 'qué tal' instead", "Drop the pronoun"]}
        result = checker(model).get_hints("Cómo estás tú", [], "Spanish")
        assert result["hints"] == ["Try 'qué tal' instead", "Drop the pronoun"]
        assert result["has_hints"] is True

    def test_says_nothing_when_the_sentence_already_sounds_right(self, model):
        model.answer = {"has_hints": False, "hints": []}
        assert checker(model).get_hints("¿Qué tal?", [], "Spanish")["has_hints"] is False

    def test_drops_blank_hints(self, model):
        model.answer = {"has_hints": True, "hints": ["Use 'qué tal'", "", "   "]}
        assert checker(model).get_hints("hola", [], "Spanish")["hints"] == ["Use 'qué tal'"]

    def test_claiming_hints_while_offering_none_counts_as_none(self, model):
        # Small models say has_hints: true and then produce an empty list.
        model.answer = {"has_hints": True, "hints": []}
        assert checker(model).get_hints("hola", [], "Spanish")["has_hints"] is False

    def test_copes_with_a_missing_hints_key(self, model):
        model.answer = {"has_hints": True}
        assert checker(model).get_hints("hola", [], "Spanish") == {
            "has_hints": False,
            "hints": [],
        }
