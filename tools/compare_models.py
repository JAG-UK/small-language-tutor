"""Measure a model at each of the three jobs this app asks of one.

The recommendations in the README came out of this script. They were measured
on one laptop against Spanish, and both of those matter — a different machine
or a different target language may well order the models differently, so the
script is here to be re-run rather than the numbers taken on faith.

    python tools/compare_models.py critic    phi4-mini:3.8b translategemma:12b
    python tools/compare_models.py chat      phi4-mini:3.8b gemma2:27b
    python tools/compare_models.py translate translategemma:4b phi4-mini:3.8b

Nothing here is deterministic, so each case is run TRIALS times and the totals
reported. A single run genuinely does flip results between models.
"""

import re
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from grammar_checker import GrammarChecker  # noqa: E402
from model_profiles import profile_for  # noqa: E402
from ollama_client import OllamaClient  # noqa: E402
from prompts import conversation_messages, translation_messages  # noqa: E402

TRIALS = 3
LANGUAGE = "es"

# What was being talked about, so the critic is judged the way the app uses it.
CONTEXT = [
    {"role": "assistant", "content": "¡Hola! ¿Cómo estás hoy?"},
    {"role": "user", "content": "Estoy bien gracias"},
    {"role": "assistant", "content": "Me alegro. ¿Dónde vives?"},
]

# (sentence, words the correction must contain — or None for "must stay quiet").
# The quiet cases matter as much as the others: a critic that rewrites correct
# Spanish teaches the learner nothing and buries the corrections that count.
CORRECTION_CASES = [
    ("vivo en madrid desde hace dos anos", ["Madrid", "años"]),  # capital, accent
    ("me gusta mucho el comida espanol", ["la comida"]),         # gender
    ("yo soy muy cansado hoy", ["estoy"]),                       # ser vs estar
    ("ayer yo comer una paella grande", ["comí"]),               # tense
    ("tengo veinte anos y soy estudiante", ["años"]),            # accent
    ("mi hermana es muy alto", ["alta"]),                        # agreement
    ("no me gusta las verduras", ["gustan"]),                    # number agreement
    ("hace mucho tiempo que no te veo", None),                   # correct
    ("Sí, en Madrid", None),                                     # a fine short answer
    ("¿Cómo se dice 'book' en español?", None),                  # correct
    ("Me encanta la música española.", None),                    # correct
]

CHAT_TURNS = [
    "hola, me llamo Jonathan",
    "vivo en madrid desde hace dos anos",
    "me gusta cocinar y leer",
]

TRANSLATIONS = [
    ("good morning, how are you?", "en-to-target", ["buenos días"]),
    ("me gustaría reservar una mesa", "target-to-en", ["table", "reserv", "book"]),
]

# Drifting into English is the failure a language-practice partner must not have.
ENGLISH = re.compile(r"\b(the|you|your|and|is|are|that|with|hello|how|what)\b", re.I)
SPANISH = re.compile(r"[¿¡áéíóúñ]|\b(que|los|las|una|muy|está|bien|hola|gracias)\b", re.I)


def _report(model, headline, misses, times):
    print(f"{model:22} {headline}   median {statistics.median(times):5.1f}s")
    for line, n in Counter(misses).most_common():
        print(f"{line}   [{n}/{TRIALS}]")
    if not misses:
        print("    (nothing missed in any trial)")
    print(flush=True)


def critic(model):
    """Does it catch real mistakes, and does it leave good sentences alone?"""
    checker = GrammarChecker(OllamaClient(model=model, timeout=600))
    times, caught, quiet, misses = [], [0, 0], [0, 0], []

    for _ in range(TRIALS):
        for text, wanted in CORRECTION_CASES:
            history = CONTEXT + [{"role": "user", "content": text}]
            start = time.perf_counter()
            result = checker.check_message(text, history, LANGUAGE, "friendly")
            times.append(time.perf_counter() - start)
            corrected = result["corrected"]

            if wanted is not None:
                caught[1] += 1
                if result["has_errors"] and all(w.lower() in corrected.lower() for w in wanted):
                    caught[0] += 1
                else:
                    misses.append(f"    missed  {text[:34]!r} -> {corrected[:52]!r}")
            else:
                quiet[1] += 1
                if result["has_errors"]:
                    misses.append(f"    noisy   {text[:34]!r} -> {corrected[:52]!r}")
                else:
                    quiet[0] += 1

    _report(model, f"caught {caught[0]}/{caught[1]}   left alone {quiet[0]}/{quiet[1]}",
            misses, times)


def chat(model):
    """Does it answer quickly, in the target language, and briefly?"""
    client, profile = OllamaClient(model=model, timeout=600), profile_for(model)
    times, drifted, words, misses = [], 0, [], []

    for _ in range(TRIALS):
        history = []
        for message in CHAT_TURNS:
            history.append({"role": "user", "content": message})
            start = time.perf_counter()
            reply = client.chat(conversation_messages(profile, LANGUAGE, "friendly", history),
                                LANGUAGE)
            times.append(time.perf_counter() - start)
            history.append({"role": "assistant", "content": reply})
            words.append(len(reply.split()))
            if len(ENGLISH.findall(reply)) > len(SPANISH.findall(reply)):
                drifted += 1
                misses.append(f"    english {reply[:60]!r}")

    total = TRIALS * len(CHAT_TURNS)
    _report(model, f"stayed in Spanish {total - drifted}/{total}   "
                   f"{statistics.median(words):.0f} words/reply", misses, times)


def translate(model):
    """Does it give the phrase back in the other language, and nothing else?"""
    client, profile = OllamaClient(model=model, timeout=600), profile_for(model)
    times, ok, misses = [], [0, 0], []

    for _ in range(TRIALS):
        for phrase, direction, wanted in TRANSLATIONS:
            start = time.perf_counter()
            out = client.chat(translation_messages(profile, phrase, LANGUAGE, direction), LANGUAGE)
            times.append(time.perf_counter() - start)
            ok[1] += 1
            if any(w.lower() in out.lower() for w in wanted):
                ok[0] += 1
            else:
                misses.append(f"    odd     {phrase[:30]!r} -> {out.strip()[:52]!r}")

    _report(model, f"sensible {ok[0]}/{ok[1]}", misses, times)


JOBS = {"critic": critic, "chat": chat, "translate": translate}

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in JOBS:
        sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(JOBS)}}} MODEL [MODEL ...]")
    job = JOBS[sys.argv[1]]
    print(f"job: {sys.argv[1]}   language: {LANGUAGE}   trials: {TRIALS}\n")
    for name in sys.argv[2:]:
        try:
            job(name)
        except Exception as exc:  # a missing model should not stop the comparison
            print(f"{name:22} FAILED: {exc}\n", flush=True)
