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
import unicodedata

from model_profiles import profile_for
from prompts import correction_messages, hints_messages

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


def _tidy(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def _differs(corrected, original):
    """Whether a correction actually changed anything a learner would notice.

    Whitespace is normalised because models reflow freely; accents and case are
    emphatically not, since those are usually the whole correction.
    """
    return _tidy(corrected) != _tidy(original)


def _words(text):
    """Lower-cased, accent-folded words. Accents are folded *here only* — a
    correction that adds one must still count as the same word, or fixing
    "anos" to "años" would look like a different sentence entirely."""
    folded = unicodedata.normalize("NFD", _tidy(text).lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return set(re.findall(r"\w+", folded))


def _is_a_correction(corrected, original):
    """Whether this is a corrected version of the sentence, or something else.

    A small model asked to correct "vivo en madrid desde hace dos anos" will
    sometimes answer "Your message is correct." or, in Spanish, "no se
    realizaron correcciones" — putting its commentary in the field meant for
    the sentence. Shown to a learner as their own corrected words, that is
    worse than showing nothing at all.

    A correction keeps the sentence's words: accents, agreement, punctuation
    and word order may all move, but a rewrite that shares almost no vocabulary
    with the original is not a correction of it. Word overlap rather than
    character similarity, because reordering — "hace dos años que vivo en
    Madrid" — is a legitimate correction that a sequence comparison scores
    close to zero.
    """
    if not _tidy(corrected):
        return False
    original_words = _words(original)
    if not original_words:
        return True

    shared = len(original_words & _words(corrected)) / len(original_words)
    if shared < 0.5:
        return False

    # Length as well as vocabulary, because the commentary sometimes quotes the
    # sentence back: "The message you provided, 'Vivo en Madrid...', is
    # correct" shares every word and is twice the length. A correction may gain
    # a missing article or two; it does not gain a sentence of English about
    # itself.
    n = len(re.findall(r"\w+", _tidy(original)))
    return len(re.findall(r"\w+", _tidy(corrected))) <= max(1.6 * n, n + 3)


class GrammarChecker:
    def __init__(self, ollama_client, profile=None):
        self.ollama = ollama_client
        # The profile decides prompt length, whether a system turn is heeded,
        # and how much conversation to carry. Derived from the model unless a
        # caller wants the critique done by a different one.
        self.profile = profile or profile_for(getattr(ollama_client, "model", ""))

    def check_message(self, user_message, conversation_context, target_language, tone="friendly"):
        """Check the learner's message for errors and suggest a correction."""
        messages = correction_messages(
            self.profile, target_language, tone, user_message, conversation_context
        )
        result = self.ollama.chat_json(messages, CORRECTION_SCHEMA)
        if not result:
            # The model or the server failed. Saying "no errors" is the safe
            # answer: inventing a correction would teach the wrong thing.
            return {
                "has_errors": False,
                "corrected": user_message,
                "explanation": "Could not check this message.",
            }

        corrected = (result.get("corrected") or "").strip() or user_message
        if not _is_a_correction(corrected, user_message):
            # The model answered with commentary rather than a sentence. Better
            # to show the learner nothing than to show them that.
            return {
                "has_errors": False,
                "corrected": user_message,
                "explanation": "Could not check this message.",
            }
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
        messages = hints_messages(
            self.profile, target_language, tone, user_message, conversation_context
        )
        result = self.ollama.chat_json(messages, HINTS_SCHEMA)
        if not result:
            return {"has_hints": False, "hints": []}

        hints = [str(h) for h in (result.get("hints") or []) if str(h).strip()]
        return {"has_hints": bool(result.get("has_hints")) and bool(hints), "hints": hints}
