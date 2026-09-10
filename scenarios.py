"""Something to talk about, chosen for what you keep getting wrong.

Two problems solved at once. Asked to simply start talking, a small model says
"¡Hola! ¿Cómo estás?" and says it again tomorrow — a concrete role gets a
concrete opening, and a deck of them gets variety without hoping for it. And a
learner who has to invent the subject every time practises whatever they can
already say, which is the opposite of the point.

So the deck is tagged with what each situation drags out of you, the recent
corrections are read for what you keep dropping, and the two are matched. What
you need to practise is counted from your own mistakes rather than asked of a
model: a scenario chosen for the wrong reason is a wasted conversation, and
there is no way to check a model's guess about that.
"""

import random
import re
import unicodedata

#: What a mistake tells us, and what a situation asks for. Deliberately few:
#: every one of these can be spotted in a correction without guessing, and four
#: reliable signals beat eight shaky ones.
ACCENTS = "accents"
CAPITALS = "capitals"
SER_ESTAR = "ser/estar"
ARTICLES = "articles"
AGREEMENT = "agreement"

#: Situations. Described in English because that is the instruction; the model
#: plays them in the language being learned.
DECK = [
    {
        "label": "At the pharmacy",
        "setting": 'You are a pharmacist behind the counter. The learner has just walked in looking unwell. Greet them and ask what the matter is.',
        "elicits": {SER_ESTAR},
    },
    {
        "label": "A neighbour you have not seen for a while",
        "setting": "You are the learner's neighbour and have not seen them in months. You have just bumped into them in the street. Say hello and ask how they have been.",
        "elicits": {SER_ESTAR},
    },
    {
        "label": "Describing your home",
        "setting": "You are a friend who has never been to the learner's home. Say you are curious about it and ask them to describe it.",
        "elicits": {ARTICLES, AGREEMENT},
    },
    {
        "label": "At the market",
        "setting": 'You run a fruit and vegetable stall and the learner is your customer. Greet them and tell them what is good today.',
        "elicits": {ARTICLES, AGREEMENT},
    },
    {
        "label": "Ordering in a restaurant",
        "setting": 'You are a waiter and the learner is your customer, just seated. Welcome them and ask what they would like.',
        "elicits": {ARTICLES, AGREEMENT},
    },
    {
        "label": "Describing a friend",
        "setting": "You are a friend of the learner's. Ask them to tell you about someone close to them and what that person is like.",
        "elicits": {SER_ESTAR, AGREEMENT},
    },
    {
        "label": "How is everyone today",
        "setting": "You are the learner's colleague. The office is unusually busy. Say so, and ask how they are holding up.",
        "elicits": {SER_ESTAR, AGREEMENT},
    },
    {
        "label": "Lost in the city",
        "setting": 'You are a passer-by in the street and the learner looks lost. Ask them whether they need help finding somewhere.',
        "elicits": {ARTICLES},
    },
    {
        "label": "Booking a room",
        "setting": 'You are a hotel receptionist and the learner has come to the desk. Welcome them and ask what sort of room they need.',
        "elicits": {ARTICLES, AGREEMENT},
    },
    {
        "label": "Names and places",
        "setting": 'You are meeting the learner for the first time. Introduce yourself and ask their name and where they are from.',
        "elicits": {CAPITALS, ACCENTS},
    },
    {
        "label": "How old is everything",
        "setting": 'You are an antiques dealer and the learner is browsing your shop. Ask them about the oldest thing they own.',
        "elicits": {ACCENTS},
    },
    {
        "label": "Weekend plans",
        "setting": 'You are a friend making plans for the weekend. Suggest something and ask whether the learner wants to come.',
        "elicits": set(),
    },
    {
        "label": "Something you cooked",
        "setting": 'You are a friend who loves food. Ask the learner what the last thing they cooked was.',
        "elicits": {ARTICLES},
    },
    {
        "label": "A film you both saw",
        "setting": "You are a friend of the learner's. You have both just seen the same film and you liked it less than they did. Say what you thought and ask what they made of it.",
        "elicits": {AGREEMENT},
    },
]


def _fold(word):
    return "".join(
        c for c in unicodedata.normalize("NFD", (word or "").lower())
        if not unicodedata.combining(c)
    )


#: Swaps that say something specific, in Spanish. Everything above this line
#: works in any language the app offers; this table does not, so a language it
#: does not cover simply contributes the accent and capital signals.
_SER_ESTAR = {
    ("soy", "estoy"), ("estoy", "soy"), ("es", "está"), ("está", "es"),
    ("son", "están"), ("están", "son"), ("eres", "estás"), ("estás", "eres"),
    ("era", "estaba"), ("estaba", "era"),
}
_ARTICLES = {
    ("el", "la"), ("la", "el"), ("un", "una"), ("una", "un"),
    ("los", "las"), ("las", "los"), ("unos", "unas"), ("unas", "unos"),
}


def what_it_shows(wrote, should_be):
    """What one corrected word says the learner needs work on, or None.

    Only the confident cases. A swap this cannot explain contributes nothing
    rather than a guess — the deck falls back to variety, which is no worse than
    where it started.
    """
    if wrote == should_be:
        return None

    pair = (wrote.lower(), should_be.lower())
    if pair in _SER_ESTAR:
        return SER_ESTAR
    if pair in _ARTICLES:
        return ARTICLES

    # Same letters, different marks: an accent or a tilde, whatever the language.
    if _fold(wrote) == _fold(should_be):
        return CAPITALS if wrote.lower() == should_be.lower() else ACCENTS

    # Gender and number live in the ending, so a word that changed only there
    # is an agreement slip: alto/alta, gusta/gustan, español/española.
    #
    # Tightly, though. A looser "same stem" rule called comer/comí agreement,
    # which is a tense mistake wearing a similar shape, and a scenario chosen
    # for the wrong reason is a wasted conversation.
    before, after = _fold(wrote), _fold(should_be)
    if len(before) == len(after) and before[:-1] == after[:-1]:
        # o <-> a, and only that. Compared unfolded, so an accented ending is
        # excluded: hablo/hablé is a tense, and folding turned the é into an e
        # that looked like one more gender ending.
        if {wrote[-1:].lower(), should_be[-1:].lower()} == {"o", "a"}:
            return AGREEMENT
    elif after in (before + suffix for suffix in ("s", "n", "a", "es", "as", "os")):
        return AGREEMENT

    return None


def weak_areas(corrections, minimum=2):
    """What the learner keeps getting wrong, commonest first.

    A single slip is a slip. Twice is worth a conversation built around it.
    """
    from report import word_changes

    counts = {}
    for correction in corrections:
        for wrote, should_be in word_changes(correction["message"], correction["corrected"]):
            area = what_it_shows(wrote, should_be)
            if area:
                counts[area] = counts.get(area, 0) + 1

    return [area for area, times in sorted(counts.items(), key=lambda kv: -kv[1])
            if times >= minimum]


def choose(corrections=(), avoid=(), rng=random):
    """A situation to talk about, picked for what needs practice.

    Among the scenarios that exercise the learner's worst area, at random — so
    the choice is targeted without being the same conversation every time.
    Falls back to the whole deck when there is nothing to go on, which is the
    first few conversations and any language the table above does not cover.
    """
    avoid = set(avoid)
    available = [s for s in DECK if s["label"] not in avoid] or DECK

    for area in weak_areas(corrections):
        matching = [s for s in available if area in s["elicits"]]
        if matching:
            return dict(rng.choice(matching), chosen_for=area)

    return dict(rng.choice(available), chosen_for=None)
