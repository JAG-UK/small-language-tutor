"""Feedback on what the learner wrote: corrections, and hints on naturalness.

Both ask the model for a fixed shape and get it, because Ollama constrains
generation to a JSON schema. That replaces several hundred lines of regex that
existed to salvage half-formed JSON out of prose — a small model asked politely
for JSON will still wrap it in a code fence, trail off mid-object, or explain
itself first. Asked with a schema, it cannot.

Getting the shape right is not the same as getting the answer right. Small
models are markedly better at producing a corrected sentence than at judging
whether they produced one: phi4-mini:3.8b will rewrite "yo tengo 20 anos y vivo
en madrid" as "yo tengo 20 años y vivo en Madrid" and report has_errors: false
in the same breath. So the flag is derived from whether the text actually
changed, and the model's own answer is only a hint.
"""

import re

CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "has_errors": {"type": "boolean"},
        "corrected": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["has_errors", "corrected", "explanation"],
}

HINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "has_hints": {"type": "boolean"},
        "hints": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["has_hints", "hints"],
}


def _differs(corrected, original):
    """Whether a correction actually changed anything a learner would notice.

    Whitespace is normalised because models reflow freely; accents and case are
    emphatically not, since those are usually the whole correction.
    """
    tidy = lambda text: re.sub(r"\s+", " ", (text or "")).strip()
    return tidy(corrected) != tidy(original)


class GrammarChecker:
    def __init__(self, ollama_client):
        self.ollama = ollama_client

    def check_message(self, user_message, conversation_context, target_language, tone="friendly"):
        """Check the learner's message for errors and suggest a correction."""
        tone_instructions = {
            "friendly": "casual, informal, warm",
            "professional": "formal, polite, business-like",
            "flirty": "playful, charming, slightly suggestive",
        }
        tone_desc = tone_instructions.get(tone, "casual, informal")

        system_prompt = f"""You are a precise language tutor. Analyze the user's message for grammar, spelling, naturalness, and tone appropriateness in {target_language}. The conversation tone is {tone} ({tone_desc}). PRESERVE THE USER'S INTENT: Do not change statements to questions, questions to statements, or alter the sentence structure. Only fix grammar, spelling, punctuation, and word errors.

CRITICAL INSTRUCTIONS:
1. First, create the corrected version of the message
2. Then, CAREFULLY compare the ORIGINAL and CORRECTED versions side-by-side
3. In your explanation, ONLY describe the ACTUAL differences you made between original and corrected
4. Do NOT convert statements to questions or vice versa unless there is a clear grammatical error requiring it. For example:
    - If the user writes 'Estoy buscando...' (statement), do NOT change it to '¿Estás buscando...?' (question) - preserve the statement form.
5. Only correct actual errors. Do not try to 'improve' or restructure the message.
6. Be precise and accurate when describing the differences. For example:
   - If the change was a simple typo, say "typo in '...' should be '...'" NOT "should be '...'"
   - If the change was a missing accent, say "missing accent on '...' should be '...'" NOT "should be '...'"
   - If the change was a mis-placed or extra accent, say "mis-placed accent on '...' should be '...'" NOT "should be '...'"
7. Only mention changes that actually exist - verify each point by comparing original vs corrected
8. Be specific: name the exact letters, accents, punctuation marks, or words that changed
9. Check if the tone matches the conversation style ({tone}): if the user uses overly formal language in a flirty conversation, or vice versa, suggest appropriate tone adjustments

Respond as JSON with keys has_errors, corrected and explanation.
The explanation should list ONLY the actual differences, numbered if multiple.
If no errors, return has_errors: false and the message unchanged."""

        result = self.ollama.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Correct this message: {user_message}"},
            ],
            CORRECTION_SCHEMA,
        )
        if not result:
            # The model or the server failed. Saying "no errors" is the safe
            # answer: inventing a correction would teach the wrong thing.
            return {
                "has_errors": False,
                "corrected": user_message,
                "explanation": "Could not check this message.",
            }

        corrected = (result.get("corrected") or "").strip() or user_message
        # Derived, not trusted: a model that says "no errors" while handing back
        # a different sentence has still found one, and the learner should see
        # it. A model that flags an error but changes nothing has not.
        has_errors = _differs(corrected, user_message)
        return {
            "has_errors": has_errors,
            "corrected": corrected,
            "explanation": result.get("explanation") or "",
        }

    def get_hints(self, user_message, conversation_context, target_language, tone="friendly"):
        """Suggest ways to sound more natural, even when nothing is wrong."""
        tone_instructions = {
            "friendly": "casual, informal, warm",
            "professional": "formal, polite, business-like",
            "flirty": "playful, charming, slightly suggestive",
        }
        tone_desc = tone_instructions.get(tone, "casual, informal")

        system_prompt = f"""You are a helpful language tutor providing tips to make {target_language} more natural and idiomatic. You are watching a conversation the user is having with a native {target_language} speaker and providing real-time feedback to help the user improve next time. The conversation tone is {tone} ({tone_desc}).

Look for:
1. Phrases that are grammatically correct but sound unnatural or textbook-like
2. More idiomatic or colloquial alternatives a native speaker would use
3. Ways to better match the {tone} tone of the conversation
4. Common expressions that would fit better

Respond as JSON with keys has_hints and hints, where hints is a list of short suggestions.
Give at most three hints. If the message already sounds natural, return has_hints: false and an empty list."""

        result = self.ollama.chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Provide hints for this message: {user_message}"},
            ],
            HINTS_SCHEMA,
        )
        if not result:
            return {"has_hints": False, "hints": []}

        hints = [str(h) for h in (result.get("hints") or []) if str(h).strip()]
        return {"has_hints": bool(result.get("has_hints")) and bool(hints), "hints": hints}
