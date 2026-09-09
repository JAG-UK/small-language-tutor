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

# A hint is two things in two languages, so the schema asks for them separately.
# As one free string the model wrote whichever it felt like — sometimes an
# English rewrite of a Spanish sentence, sometimes a bare phrase with no reason
# attached, sometimes a reply to the learner rather than advice for them. Split
# in two, generation is constrained to produce both parts.
HINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "has_hints": {"type": "boolean"},
        "hints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    #: How to say it, in the language being learned.
                    "suggestion": {"type": "string"},
                    #: Why that is better, in English, so the learner can follow.
                    "why": {"type": "string"},
                },
                "required": ["suggestion", "why"],
            },
        },
    },
    "required": ["has_hints", "hints"],
}


def _tidy(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def _unmarkdown(text):
    """Strip emphasis markers from a sentence meant to be read as plain text.

    Models reach for markdown unprompted — translategemma:12b italicises book
    titles as *Don Quijote*, phi4-mini quotes words as `ameliorar`, and both
    have been seen to bold a whole correction. The
    panel shows the sentence as-is, so the asterisks arrive as asterisks and the
    learner is quietly taught to type them.
    """
    # The markers only count at the edges of a word: "una nota_al_pie" is a word
    # with underscores in it, not emphasis, and unwrapping it loses letters.
    return re.sub(
        r"(?<![\w*_`])(\*{1,3}|_{1,3}|`{1,3})(\S(?:.*?\S)?)\1(?![\w*_`])", r"\2", text or ""
    )


def _cosmetic(text):
    """The sentence with the two things chat does not observe removed: a capital
    on the first word, and a full stop at the end."""
    stripped = _tidy(text).rstrip(".!?")
    return stripped[:1].lower() + stripped[1:]


def _differs(corrected, original):
    """Whether a correction changed anything worth telling a learner about.

    Whitespace is normalised because models reflow freely; accents and case
    within the sentence are emphatically not, since those are usually the whole
    correction — "madrid" to "Madrid" is exactly the kind of thing to catch.

    But every model tried here answers a perfectly good chat message by adding a
    capital and a full stop, and reporting that as an error fills the panel with
    remarks about punctuation nobody uses when chatting. So a change confined to
    the opening capital and the closing stop does not count as one.
    """
    if _tidy(corrected) == _tidy(original):
        return False
    return _cosmetic(corrected) != _cosmetic(original)


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


#: A word the explanation puts in quotes, of either kind.
_QUOTED = r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{1,30}?)['\"\u2018\u2019\u201c\u201d]"
_REMOVED = re.compile(rf"(?:removed|deleted|dropped)\s+(?:the\s+\w+\s+)?{_QUOTED}", re.I)
_WAS_REMOVED = re.compile(
    rf"{_QUOTED}\s+(?:was|were|is|has been)\s+(?:removed|deleted|dropped)", re.I
)
_CHANGED_TO = re.compile(rf"{_QUOTED}\s*(?:to|->|\u2192)\s*{_QUOTED}", re.I)
#: A whole explanation that amounts to "nothing was wrong". Matched against the
#: entire text, so a real explanation that happens to end on a reassuring note
#: keeps the part that teaches something.
_DENIAL = re.compile(
    r"^(?:there\s+(?:are|were)\s+)?no\s+(?:changes?|corrections?|errors?|mistakes?)"
    r"(?:\s+(?:needed|required|found|were\s+made|made))?[.!]?$",
    re.I,
)


def _mentions(word, sentence):
    """Case-insensitively, because the sentence may capitalise it — but never
    accent-insensitively: "Si" and "Sí" are different words, and that difference
    is usually the entire correction."""
    return re.search(rf"(?<!\w){re.escape(word.strip())}(?!\w)", sentence, re.I) is not None


def _contradicts(corrected, explanation):
    """Whether the explanation claims a change the correction plainly did not make.

    Asked to correct "Si, gracias. ¿Tal vez algo internacional?", phi4-mini
    fixed the accent on "Si" and then said "the word 'tal' was removed because
    it's unnecessary" — with "tal" still sitting in the sentence it had just
    written. It also produced "Corrected 'true' to 'true', 'horrors' to
    'errors'" for a sentence containing none of those words.

    A learner cannot tell that from a real correction, and this one teaches them
    to stop using a word that was fine. So the claim is checked against the two
    sentences, and only claims contradicted by them count: a word said to be
    removed that is still there, or a word said to have been changed into
    itself. Nothing here judges whether an explanation is *good* — an
    explanation naming no words at all, or reaching for "ser" to explain a fix
    to "soy", is vague rather than false, and is left alone.
    """
    for pattern in (_REMOVED, _WAS_REMOVED):
        for match in pattern.finditer(explanation or ""):
            if _mentions(match.group(1), corrected):
                return True
    for match in _CHANGED_TO.finditer(explanation or ""):
        # Exactly equal, not merely equal ignoring case: "changed 'madrid' to
        # 'Madrid'" is the correction working, not a contradiction.
        if match.group(1).strip() == match.group(2).strip():
            return True
    # "No changes needed." printed beside a visible change. The model makes the
    # correction and then denies making it, the same way it misreports
    # has_errors. Saying nothing says as much, without the confusion.
    return bool(_DENIAL.match((explanation or "").strip()))


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

        corrected = _unmarkdown(result.get("corrected") or "").strip() or user_message
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
        explanation = _unmarkdown(result.get("explanation") or "")
        if _contradicts(corrected, explanation):
            # The correction itself still stands — it is the prose about it that
            # is wrong, and no explanation beats a false one.
            explanation = ""

        return {
            "has_errors": has_errors,
            "corrected": corrected,
            "explanation": explanation,
        }

    def get_hints(self, user_message, conversation_context, target_language, tone="friendly"):
        """Suggest ways to sound more natural, even when nothing is wrong."""
        messages = hints_messages(
            self.profile, target_language, tone, user_message, conversation_context
        )
        result = self.ollama.chat_json(messages, HINTS_SCHEMA)
        if not result:
            return {"has_hints": False, "hints": []}

        hints = []
        for hint in result.get("hints") or []:
            if not isinstance(hint, dict):
                continue  # a bare string is the old shape, and has no "why"
            suggestion = _tidy(_unmarkdown(hint.get("suggestion")))
            why = _tidy(_unmarkdown(hint.get("why")))
            if suggestion and why:
                hints.append({"suggestion": suggestion, "why": why})

        return {"has_hints": bool(result.get("has_hints")) and bool(hints), "hints": hints}
