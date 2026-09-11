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
import unicodedata

#: What a mistake tells us, and what a situation asks for. Deliberately few:
#: every one of these can be spotted in a correction without guessing, and four
#: reliable signals beat eight shaky ones.
ACCENTS = "accents"
CAPITALS = "capitals"
SER_ESTAR = "ser/estar"
ARTICLES = "articles"
AGREEMENT = "agreement"

#: Situations, and what each one drags out of you.
#:
#: Two fields rather than one, because they are used at different moments.
#: `setting` is who the other person is and what they want, and it travels with
#: the conversation for the whole of it. `opening` is only their first move.
#: Folding the two together — "Greet them and ask what the matter is" — had the
#: partner greeting the learner again on every turn, once the setting started
#: being carried past the opening.
#:
#: Described in English because that is the instruction; the model plays them in
#: the language being learned. They are set where the learner lives, which is
#: the difference between practising a language and practising a phrasebook:
#: nobody who lives here needs to check into a hotel.
#:
#: Each one gives the other person something they want — an opinion, a
#: complaint, a thing to sell, a suspicion. A partner with nothing at stake asks
#: "¿y tú?" until the conversation dies of it.
DECK = [
    {
        "label": "The portera and the building work",
        "setting": "You are the portera of the learner's building, mopping the "
        "entrance. The lift has been out for a week, the builders keep propping "
        "the street door open, and you have opinions about both.",
        "opening": "Catch them on their way in and tell them what the builders "
        "have done today.",
        "elicits": {SER_ESTAR, AGREEMENT},
    },
    {
        "label": "The gestoría and the wrong document",
        "setting": "You work at a gestoría. The learner has come in about their "
        "empadronamiento and has brought the wrong piece of paper. You are not "
        "unkind about it, but you cannot do anything with what they have.",
        "opening": "Ask what they have come in for, then tell them the document "
        "is not the right one.",
        "elicits": {ARTICLES, CAPITALS},
    },
    {
        "label": "The landlord and the deposit",
        "setting": "You are the learner's landlord. They are moving out and you "
        "intend to keep part of the fianza for a mark on the kitchen wall, which "
        "you suspect was there when they arrived but would rather not discuss.",
        "opening": "Tell them you have been round the flat and there is "
        "something you want to talk about.",
        "elicits": {SER_ESTAR, ARTICLES},
    },
    {
        "label": "The bar downstairs",
        "setting": "You own the bar de barrio the learner drinks in. You have "
        "known them by sight for a year, you have wondered about them for most "
        "of it, and today you have decided to find out.",
        "opening": "Put their coffee down and ask them the thing you have been "
        "wondering.",
        "elicits": {SER_ESTAR, AGREEMENT},
    },
    {
        "label": "The building's group chat",
        "setting": "You are a neighbour in the learner's building, writing in "
        "the building's group chat. Somebody has been leaving bags beside the "
        "bins instead of in them. You are fairly sure it is not the learner, but "
        "you want them on your side before the next junta.",
        "opening": "Write the message that starts the row, politely.",
        "elicits": {SER_ESTAR, AGREEMENT},
    },
    {
        "label": "At the CAP",
        "setting": "You are a doctor at the learner's CAP. They have come in "
        "with something that has been going on for a while, and they are "
        "underplaying it.",
        "opening": "Greet them, ask what has brought them in, and ask how long "
        "it has been like that.",
        "elicits": {SER_ESTAR},
    },
    {
        "label": "The stall at the mercat",
        "setting": "You run a stall at the neighbourhood mercat and the learner "
        "is a regular. What they always buy is poor this week and you would "
        "rather sell them something better.",
        "opening": "Greet them, and steer them off their usual order before "
        "they ask for it.",
        "elicits": {ARTICLES, AGREEMENT},
    },
    {
        "label": "Sant Jordi",
        "setting": "You are a friend of the learner's, at a book stall on Sant "
        "Jordi. You are buying for someone you want to impress and you cannot "
        "decide between two books.",
        "opening": "Hold both of them up and ask which one they would give.",
        "elicits": {CAPITALS, ACCENTS},
    },
    {
        "label": "A tourist asks you",
        "setting": "You are a visitor to the city, lost, with a phone that has "
        "run out of data. The learner lives here — you stopped them because they "
        "looked local.",
        "opening": "Apologise for stopping them, and ask how to get where you "
        "are going.",
        "elicits": {ARTICLES},
    },
    {
        "label": "The internet has been down three days",
        "setting": "You are on the phone from the learner's internet provider. "
        "Their connection has been down since Monday, you do not believe it is "
        "your company's fault, and you would like to establish that before "
        "sending anybody out.",
        "opening": "Introduce yourself and ask them to describe exactly what is "
        "happening.",
        "elicits": {ARTICLES, AGREEMENT},
    },
    {
        "label": "The parcel that was delivered to nobody",
        "setting": "You work at the Correos office. The learner's parcel is "
        "marked as delivered and signed for, they never received it, and the "
        "signature is not a name either of you recognises.",
        "opening": "Look it up, and tell them what the system says.",
        "elicits": {ARTICLES, CAPITALS},
    },
    {
        "label": "The party upstairs",
        "setting": "You are the learner's upstairs neighbour. They have knocked "
        "at one in the morning about the noise. You are not especially sorry, "
        "and it is somebody's birthday.",
        "opening": "Open the door, over the music, and ask what they want.",
        "elicits": {SER_ESTAR, AGREEMENT},
    },
    {
        "label": "Helping with the festa major",
        "setting": "You are a neighbour organising the barrio's festa major. You "
        "are short of people for Saturday and the learner has walked past at the "
        "wrong moment.",
        "opening": "Stop them, and ask them to take on one specific job.",
        "elicits": {CAPITALS, ACCENTS},
    },
    {
        "label": "After the match",
        "setting": "You are a regular at the bar and have just watched the match "
        "beside the learner. You thought the team were dreadful and you suspect "
        "they are about to defend them.",
        "opening": "Say what you thought of it, plainly, and ask whether they "
        "saw the same game.",
        "elicits": {AGREEMENT},
    },
    {
        "label": "In the peluquería",
        "setting": "You are cutting the learner's hair, with half an hour and "
        "nothing else to do. You want to know where they are from, what they are "
        "doing here, and whether they are staying.",
        "opening": "Ask what they want done, then start asking about them.",
        "elicits": {CAPITALS, ACCENTS, SER_ESTAR},
    },
    {
        "label": "Viewing a flat in Gràcia",
        "setting": "You are showing the learner a flat you are letting. It is "
        "smaller and darker than the advert suggested, you know it, and you are "
        "hoping they will not ask about the courtyard.",
        "opening": "Let them in, and start selling it to them.",
        "elicits": {ARTICLES, AGREEMENT},
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
