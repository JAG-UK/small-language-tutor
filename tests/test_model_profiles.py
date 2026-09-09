"""Which profile a model gets, and how that shapes the messages sent."""

import pytest

from model_profiles import DEFAULT_PROFILE, ModelProfile, compose, profile_for, with_history


class TestRecognisingAFamily:
    @pytest.mark.parametrize(
        "model,family",
        [
            ("phi4-mini:3.8b", "phi"),
            ("gemma2:27b", "gemma"),
            ("gemma4:31b", "gemma4+"),
            ("translategemma:12b", "translategemma"),
            ("qwen3:4b", "reasoning"),
            ("deepseek-r1:70b", "reasoning"),
            ("llama3.1:8b", "general"),
            ("mistral-small", "general"),
        ],
    )
    def test_places_a_model_in_its_family(self, model, family):
        assert profile_for(model).family == family

    def test_gemma4_is_matched_before_gemma(self):
        # Order matters: the general pattern would otherwise swallow the
        # specific one, and Gemma 4 is the generation that gained a system turn.
        assert profile_for("gemma4:26b").supports_system_role is True
        assert profile_for("gemma2:27b").supports_system_role is False

    def test_translategemma_is_a_translator_first_and_a_gemma_second(self):
        profile = profile_for("translategemma:12b")
        assert profile.translation_only is True
        assert profile.supports_system_role is False  # still Gemma underneath

    def test_a_reasoning_model_is_flagged_as_such(self):
        assert profile_for("qwen3.5:9b").reasoning is True
        assert profile_for("phi4-mini-reasoning:3.8b").reasoning is True

    def test_plain_phi_is_not_mistaken_for_its_reasoning_sibling(self):
        assert profile_for("phi4-mini:3.8b").reasoning is False

    def test_an_unknown_model_gets_the_cautious_default(self):
        profile = profile_for("some-new-model:7b")
        assert profile is DEFAULT_PROFILE
        assert profile.terse is True  # short prompts are the safe end
        assert profile.supports_system_role is True

    @pytest.mark.parametrize("name", ["", None])
    def test_a_missing_name_does_not_raise(self, name):
        assert profile_for(name) is DEFAULT_PROFILE

    def test_matching_ignores_case(self):
        assert profile_for("Phi4-Mini:3.8B").family == "phi"


SYSTEM = "You are a tutor."
USER = "Correct this: hola"


class TestComposingOneExchange:
    def test_a_family_with_a_system_turn_gets_two_messages(self):
        profile = ModelProfile(family="x", supports_system_role=True)
        assert compose(profile, SYSTEM, USER) == [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ]

    def test_a_family_without_one_gets_the_instruction_folded_in(self):
        # Passing a system message to Gemma 1-3 leaves it technically present
        # and largely ignored, which is worse than not having sent it.
        profile = ModelProfile(family="x", supports_system_role=False)
        assert compose(profile, SYSTEM, USER) == [
            {"role": "user", "content": f"{SYSTEM}\n\n{USER}"}
        ]


def transcript(n):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}", "timestamp": "t"}
        for i in range(n)
    ]


class TestCarryingTheConversation:
    def test_sends_the_instruction_then_the_transcript(self):
        profile = ModelProfile(family="x", history_messages=10)
        messages = with_history(profile, SYSTEM, transcript(3))
        assert messages[0] == {"role": "system", "content": SYSTEM}
        assert [m["content"] for m in messages[1:]] == ["turn 0", "turn 1", "turn 2"]

    def test_keeps_only_as_much_history_as_the_family_should_carry(self):
        profile = ModelProfile(family="x", history_messages=4)
        messages = with_history(profile, SYSTEM, transcript(10))
        assert [m["content"] for m in messages[1:]] == ["turn 6", "turn 7", "turn 8", "turn 9"]

    def test_strips_anything_ollama_would_reject(self):
        # Stored messages carry a timestamp; the API takes role and content.
        messages = with_history(ModelProfile(family="x"), SYSTEM, transcript(2))
        assert all(set(m) == {"role", "content"} for m in messages)

    def test_an_empty_conversation_still_sends_the_instruction(self):
        assert with_history(ModelProfile(family="x"), SYSTEM, []) == [
            {"role": "system", "content": SYSTEM}
        ]

    def test_without_a_system_turn_the_instruction_rides_on_the_first_user_message(self):
        profile = ModelProfile(family="x", supports_system_role=False, history_messages=10)
        messages = with_history(profile, SYSTEM, transcript(3))

        assert all(m["role"] != "system" for m in messages)
        assert messages[0]["content"] == f"{SYSTEM}\n\nturn 0"
        assert [m["content"] for m in messages[1:]] == ["turn 1", "turn 2"]

    def test_without_a_system_turn_and_no_history_it_becomes_the_message(self):
        profile = ModelProfile(family="x", supports_system_role=False)
        assert with_history(profile, SYSTEM, []) == [{"role": "user", "content": SYSTEM}]

    def test_without_a_system_turn_and_no_user_message_it_leads(self):
        # An opening from the assistant: there is no user turn to fold into.
        profile = ModelProfile(family="x", supports_system_role=False)
        history = [{"role": "assistant", "content": "¡Hola!"}]
        messages = with_history(profile, SYSTEM, history)
        assert messages[0] == {"role": "user", "content": SYSTEM}
        assert messages[1]["content"] == "¡Hola!"
