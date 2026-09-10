"""What the mistakes add up to.

A single correction tells a learner what they got wrong once. Forty of them,
side by side, say what they keep getting wrong — which is the thing worth
practising, and the thing neither the learner nor any one conversation can see.

The split here is the same one that runs through the rest of this app: work out
what can be worked out, and ask the model only for what cannot. That you wrote
"anos" for "años" four times is arithmetic, and arithmetic does not hallucinate.
That those four sit alongside "espanol" and "manana" and amount to *not typing
accents* is a judgement, and that is what the model is for.
"""

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from prompts import language_name, report_messages

#: Below this there is no pattern to see, only a short list of mistakes.
MINIMUM_CORRECTIONS = 4

#: Themes to ask for. Enough to be useful, few enough to act on.
MAX_THEMES = 4

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    #: What to call it: "Gender agreement", "Missing accents".
                    "area": {"type": "string"},
                    #: What the learner is doing, in English.
                    "what_happens": {"type": "string"},
                    #: Their own words. Checked against the real mistakes below.
                    #: At least one is required: a model that runs long in the
                    #: prose fields will otherwise close this empty, and a theme
                    #: with no example behind it cannot be shown.
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    #: One concrete thing to do about it.
                    "practise": {"type": "string"},
                },
                "required": ["area", "what_happens", "examples", "practise"],
            },
        },
    },
    "required": ["themes"],
}


def _words(text):
    return re.findall(r"\w+", text or "")


def _fold(word):
    """Lower-cased and stripped of accents, for comparing loosely."""
    return "".join(
        c for c in unicodedata.normalize("NFD", (word or "").lower())
        if not unicodedata.combining(c)
    )


def word_changes(original, corrected):
    """The words that were swapped, one for one, as (wrote, should_be) pairs.

    Only one-for-one swaps. A correction that rewrites half the sentence says
    something about that sentence and nothing countable about a habit, and
    counting it would bury the swaps that do repeat.
    """
    before, after = _words(original), _words(corrected)
    pairs = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            pairs.extend(
                (wrote, correct)
                for wrote, correct in zip(before[i1:i2], after[j1:j2])
                if wrote != correct
            )
    return pairs


def repeated_slips(corrections, minimum=2):
    """Words corrected the same way more than once, commonest first.

    The most useful line in the whole report, and the one no model is needed
    for: you wrote "anos" for "años" four times.
    """
    counts = Counter()
    for correction in corrections:
        for wrote, correct in word_changes(correction["message"], correction["corrected"]):
            counts[(wrote, correct)] += 1

    return [
        {"wrote": wrote, "should_be": correct, "times": times}
        for (wrote, correct), times in counts.most_common()
        if times >= minimum
    ]


def _mentioned(example, corrections):
    """Whether an example the model quoted is really something they wrote.

    Loosely: accents and capitals are exactly what tends to differ between the
    quote and the original, and holding it to the letter would throw away
    perfectly good examples.
    """
    wanted = _fold(example).strip()
    if not wanted:
        return False
    for correction in corrections:
        haystack = _fold(correction["message"]) + " " + _fold(correction["corrected"])
        if wanted in haystack:
            return True
    return False


def _grounded(theme, corrections):
    """A theme keeps only the examples the learner actually wrote.

    Asked to find patterns, a model will illustrate one with a plausible mistake
    that nobody made. A report card that invents your mistakes is worse than no
    report card, so an example that cannot be found is dropped, and a theme left
    with none goes with it.
    """
    examples = [
        example for example in (theme.get("examples") or [])
        if isinstance(example, str) and _mentioned(example, corrections)
    ]
    if not examples:
        return None
    return {
        "area": (theme.get("area") or "").strip(),
        "what_happens": (theme.get("what_happens") or "").strip(),
        "examples": examples,
        "practise": (theme.get("practise") or "").strip(),
    }


def build(ollama, profile, corrections, language):
    """A report card, or a reason there isn't one.

    Always returns the countable part, whatever the model does — the repeats and
    the tally are true regardless of whether anything is answering.
    """
    report = {
        "language": language_name(language),
        "total": len(corrections),
        "conversations": len({c["conversation_id"] for c in corrections}),
        "repeats": repeated_slips(corrections),
        "themes": [],
        "note": "",
    }

    if len(corrections) < MINIMUM_CORRECTIONS:
        report["note"] = (
            f"Only {len(corrections)} corrections so far. "
            f"Keep talking — patterns need a few more before they mean anything."
        )
        return report

    answer = ollama.chat_json(
        report_messages(profile, language, corrections, MAX_THEMES), REPORT_SCHEMA
    )
    if not answer:
        report["note"] = "Could not look over these just now."
        return report

    themes = []
    for theme in (answer.get("themes") or [])[:MAX_THEMES]:
        if not isinstance(theme, dict):
            continue
        grounded = _grounded(theme, corrections)
        if grounded and grounded["area"]:
            themes.append(grounded)

    report["themes"] = themes
    if not themes:
        report["note"] = "Nothing stood out across these beyond the repeats above."
    return report
