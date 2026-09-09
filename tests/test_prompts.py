"""What each model is actually asked."""

import pytest

from model_profiles import ModelProfile, profile_for
from prompts import (
    conversation_messages,
    correction_messages,
    hints_messages,
    language_name,
    translation_messages,
)

TERSE = ModelProfile(family="small", terse=True, history_messages=6)
DETAILED = ModelProfile(family="big", terse=False, history_messages=10)
NO_SYSTEM = ModelProfile(family="gemma", terse=True, supports_system_role=False)


def text_of(messages):
    return "\n".join(m["content"] for m in messages)


class TestLanguageNames:
    def test_turns_a_code_into_a_name(self):
        assert language_name("es") == "Spanish"
        assert language_name("DE") == "German"

    def test_passes_an_unknown_code_through(self):
        # Adding a language to the UI should not require editing this table first.
        assert language_name("nl") == "nl"

    @pytest.mark.parametrize("builder", [conversation_messages, None])
    def test_prompts_name_the_language_rather_than_its_code(self, builder):
        # The prompts used to interpolate the code: "You are a es native speaker".
        prompt = text_of(conversation_messages(TERSE, "es", "friendly", []))
        assert "Spanish" in prompt
        assert "a es " not in prompt


class TestTheConversationPartner:
    def test_tells_the_model_which_language_and_tone(self):
        prompt = text_of(conversation_messages(DETAILED, "fr", "professional", []))
        assert "French" in prompt
        assert "professional" in prompt

    def test_a_small_model_gets_a_short_prompt(self):
        terse = text_of(conversation_messages(TERSE, "es", "friendly", []))
        detailed = text_of(conversation_messages(DETAILED, "es", "friendly", []))
        assert len(terse) < len(detailed) / 2
        # Both still say the thing that matters most.
        assert "Spanish" in terse and "Spanish" in detailed

    def test_carries_the_transcript(self):
        history = [{"role": "user", "content": "Hola", "timestamp": "t"}]
        assert conversation_messages(TERSE, "es", "friendly", history)[-1] == {
            "role": "user",
            "content": "Hola",
        }


class TestTheCorrector:
    def test_puts_the_message_in_front_of_the_model(self):
        prompt = text_of(correction_messages(TERSE, "es", "friendly", "yo tengo 20 anos"))
        assert "yo tengo 20 anos" in prompt

    def test_says_what_kind_of_change_is_wanted(self):
        prompt = text_of(correction_messages(TERSE, "es", "friendly", "hola"))
        assert "statement" in prompt  # do not turn statements into questions

    def test_a_small_model_gets_a_short_prompt(self):
        terse = text_of(correction_messages(TERSE, "es", "friendly", "hola"))
        detailed = text_of(correction_messages(DETAILED, "es", "friendly", "hola"))
        assert len(terse) < len(detailed) / 2

    def test_folds_the_instruction_in_for_a_family_without_a_system_turn(self):
        messages = correction_messages(NO_SYSTEM, "es", "friendly", "hola")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "tutor" in messages[0]["content"] and "hola" in messages[0]["content"]


class TestGivingTheCriticContext:
    """"Sí, en Madrid" is a fragment on its own and a good answer to a question.
    A critic that cannot see the question misreads it."""

    history = [
        {"role": "assistant", "content": "¿Dónde vives?"},
        {"role": "user", "content": "Sí, en Madrid"},
    ]

    def test_includes_what_came_before(self):
        prompt = text_of(correction_messages(TERSE, "es", "friendly", "Sí, en Madrid", self.history))
        assert "¿Dónde vives?" in prompt

    def test_does_not_repeat_the_message_being_judged(self):
        prompt = text_of(correction_messages(TERSE, "es", "friendly", "Sí, en Madrid", self.history))
        assert prompt.count("Sí, en Madrid") == 1

    def test_works_when_the_caller_did_not_append_the_message_first(self):
        earlier = [{"role": "assistant", "content": "¿Dónde vives?"}]
        prompt = text_of(correction_messages(TERSE, "es", "friendly", "Sí, en Madrid", earlier))
        assert "¿Dónde vives?" in prompt
        assert "Sí, en Madrid" in prompt

    def test_says_nothing_about_context_when_there_is_none(self):
        prompt = text_of(correction_messages(TERSE, "es", "friendly", "hola", []))
        assert "Conversation so far" not in prompt

    def test_carries_less_of_the_transcript_than_the_conversation_does(self):
        # The critic judges one sentence; it does not need the whole exchange.
        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
            for i in range(20)
        ]
        prompt = text_of(correction_messages(DETAILED, "es", "friendly", "hola", long_history))
        included = [t for t in range(20) if f"turn {t}" in prompt]
        assert 0 < len(included) < DETAILED.history_messages

    def test_hints_get_the_same_context(self):
        prompt = text_of(hints_messages(TERSE, "es", "friendly", "Sí, en Madrid", self.history))
        assert "¿Dónde vives?" in prompt


class TestTheCoach:
    def test_asks_for_a_small_number_of_suggestions(self):
        prompt = text_of(hints_messages(TERSE, "es", "friendly", "hola"))
        assert "three" in prompt.lower()

    def test_says_to_stay_quiet_when_there_is_nothing_to_say(self):
        for profile in (TERSE, DETAILED):
            prompt = text_of(hints_messages(profile, "es", "friendly", "hola"))
            assert "natural" in prompt.lower()


class TestTheTranslator:
    def test_translates_into_the_target_language(self):
        prompt = text_of(translation_messages(TERSE, "good morning", "es", "en-to-target"))
        assert "English" in prompt and "Spanish" in prompt
        assert "good morning" in prompt

    def test_translates_back_into_english(self):
        prompt = text_of(translation_messages(TERSE, "buenos días", "es", "target-to-en"))
        assert "buenos días" in prompt
        assert prompt.index("Spanish") < prompt.index("English")  # source before target

    def test_asks_for_the_translation_alone(self):
        prompt = text_of(translation_messages(TERSE, "hola", "es", "target-to-en"))
        assert "no explanation" in prompt.lower()


class TestAgainstTheRealProfiles:
    def test_the_default_model_gets_short_prompts(self):
        # phi4-mini is the shipped default and does better with less.
        prompt = text_of(conversation_messages(profile_for("phi4-mini:3.8b"), "es", "friendly", []))
        assert len(prompt) < 400

    def test_gemma_gets_no_system_message(self):
        messages = conversation_messages(profile_for("gemma2:27b"), "es", "friendly", [])
        assert all(m["role"] != "system" for m in messages)
