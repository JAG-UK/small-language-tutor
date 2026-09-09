"""Everything the models are asked, in one place.

Each builder returns a ready-to-send message list rather than a string, because
how the instruction is delivered is part of the prompt: a family without a
system turn needs it folded into the user message, and how much transcript to
carry is a property of the model, not of the caller. See model_profiles.

Prompts come in two lengths. The detailed ones are the tuned originals; the
terse ones say the same thing in a few imperative lines, because a 3B model
follows those better than nine numbered rules — past a point more instruction
makes a small model worse. Which one is used is the profile's call.
"""

from model_profiles import ModelProfile, compose, with_history

# The UI sends codes; prompts need names. Interpolating the code produced
# "You are a es native speaker", which is not a sentence any model benefits from.
LANGUAGE_NAMES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "en": "English",
}


def language_name(code: str) -> str:
    """"es" -> "Spanish". Unknown codes pass through, so adding a language to
    the UI does not require editing this file first."""
    return LANGUAGE_NAMES.get((code or "").lower(), code)


# Two registers for each tone: the conversational partner needs to be told how
# to behave, the critic only needs to know what it is judging against.
TONES = {
    "friendly": {
        "behave": "warm, casual, and approachable. Use informal language, "
        "friendly expressions, and show interest in the conversation.",
        "judge": "casual, informal, warm",
    },
    "professional": {
        "behave": "polite, formal, and business-like. Use formal language, "
        "proper titles if appropriate, and maintain a respectful distance.",
        "judge": "formal, polite, business-like",
    },
    "flirty": {
        "behave": "playful, charming, and slightly suggestive. Use teasing "
        "language, compliments, and create a light romantic or playful atmosphere.",
        "judge": "playful, charming, slightly suggestive",
    },
}


def tone_description(tone: str, register: str = "behave") -> str:
    return TONES.get(tone, TONES["friendly"])[register]


# --- the conversation partner ---------------------------------------------


def _conversation_system(profile: ModelProfile, language: str, tone: str) -> str:
    lang = language_name(language)
    how = tone_description(tone, "behave")

    if profile.terse:
        return (
            f"You are a native {lang} speaker chatting with someone learning {lang}.\n"
            f"Be {how}\n"
            f"Reply only in {lang}, never in English. Keep it to 2-3 sentences.\n"
            f"If their message does not make sense, ask them to explain."
        )

    return f"""You are a {lang} native speaker having a natural conversation with the user. The tone of this conversation is {tone}: be {how}

CRITICAL RULES:
1. You MUST respond ENTIRELY in {lang}. Do NOT use English or any other language.
2. Every word, phrase, and sentence must be in {lang} only.
3. Maintain the {tone} tone throughout your responses - match the user's tone and style.
4. Keep responses natural, conversational, and appropriate for a language learner.
5. Keep responses brief (2-3 sentences max).
6. If you need to explain something, explain it in {lang}, not in English.
7. If you don't understand what the user is saying, or it doesn't make sense in the context of the rest of the conversation, ask them for clarification.

Remember: This is a language practice conversation in a {tone} tone. The entire conversation must be in {lang}."""


def conversation_messages(profile: ModelProfile, language: str, tone: str, history: list[dict]):
    """The partner's turn: instruction plus as much transcript as it should carry."""
    return with_history(profile, _conversation_system(profile, language, tone), history)


# --- the corrector ---------------------------------------------------------


def _correction_system(profile: ModelProfile, language: str, tone: str) -> str:
    lang = language_name(language)
    judged = tone_description(tone, "judge")

    if profile.terse:
        return (
            f"You are a {lang} tutor. Correct the user's {lang} message.\n"
            f"Fix only grammar, spelling, accents and punctuation. Keep their "
            f"meaning and sentence type: a statement stays a statement, a "
            f"question stays a question.\n"
            f"Say exactly what you changed and why, naming the words involved: "
            f"one short sentence per change, no introduction and no summary.\n"
            f"If nothing is wrong, return the message unchanged.\n"
            f"The conversation is {tone} ({judged})."
        )

    return f"""You are a precise language tutor. Analyze the user's message for grammar, spelling, naturalness, and tone appropriateness in {lang}. The conversation tone is {tone} ({judged}). PRESERVE THE USER'S INTENT: Do not change statements to questions, questions to statements, or alter the sentence structure. Only fix grammar, spelling, punctuation, and word errors.

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

Report has_errors, the corrected message, and an explanation listing ONLY the
actual differences, numbered if there are several. If nothing is wrong, return
the message unchanged.

Keep the explanation to one short sentence per change, with no introduction and
no closing summary. Length is not thoroughness here: a long explanation is cut
off mid-sentence before the learner reaches the end of it."""


def correction_messages(
    profile: ModelProfile, language: str, tone: str, message: str, history=()
):
    """Ask for a correction of one message, optionally in the light of what was
    being talked about."""
    return _judging(
        profile,
        _correction_system(profile, language, tone),
        "correct this message",
        message,
        history,
        reminder="Explain in English, in one short sentence per change.",
    )


# --- the coach -------------------------------------------------------------


def _hints_system(profile: ModelProfile, language: str, tone: str) -> str:
    lang = language_name(language)
    judged = tone_description(tone, "judge")

    if profile.terse:
        return (
            f"You are a {lang} tutor watching a {tone} conversation ({judged}).\n"
            f"Suggest up to three short ways the user's message could sound more "
            f"like a native speaker — more idiomatic, or a better fit for the tone.\n"
            f"For each one give a suggestion — how to say it, written in {lang} "
            f"— and a why, one short sentence of English explaining what is "
            f"better about it. Never suggest an English phrase.\n"
            f"You are advising the learner, not replying to them: do not answer "
            f"the message or continue the conversation.\n"
            f"Only suggest something if it would genuinely be an improvement. "
            f"If the message already sounds natural, return no hints."
        )

    return f"""You are a helpful language tutor providing tips to make {lang} more natural and idiomatic. You are watching a conversation the user is having with a native {lang} speaker and providing real-time feedback to help the user improve next time. The conversation tone is {tone} ({judged}).

CRITICAL RULES:
1. Your explanations must be in English, so the learner understands them, BUT
2. every phrase, alternative and example you suggest MUST be in {lang}
3. NEVER offer an English phrase as an alternative — a hint like "try 'Not yet'"
   is useless to someone learning {lang}
4. You are advising the learner, not talking to them. Do not answer their
   message, and do not carry the conversation on

Look for:
1. Phrases that are grammatically correct but sound unnatural or textbook-like
2. More idiomatic or colloquial alternatives a native speaker would use
3. Ways to better match the {tone} tone of the conversation
4. Common expressions that would fit better

Each hint is a suggestion — the phrase itself, in {lang} — and a why, one short
sentence of English saying what is better about it.

Right: suggestion "Aún no", why "More natural than 'No todavía' in speech."
Wrong: suggestion "Not yet"  (the suggestion must never be English)
Wrong: suggestion "¿Has pensado en otro libro?"  (that is a reply, not a hint)

Give at most three short hints. If the message already sounds natural, return
no hints rather than inventing one."""


def hints_messages(profile: ModelProfile, language: str, tone: str, message: str, history=()):
    lang = language_name(language)
    return _judging(
        profile,
        _hints_system(profile, language, tone),
        "suggest improvements to this message",
        message,
        history,
        reminder=f"Write each suggestion in {lang}, and each why in English.",
    )


# --- the translator --------------------------------------------------------


def translation_messages(profile: ModelProfile, phrase: str, language: str, direction: str):
    """One phrase, either way. Direction is "en-to-target" or the reverse."""
    lang = language_name(language)
    source, target = ("English", lang) if direction == "en-to-target" else (lang, "English")

    system = (
        f"You are a translator. Translate the {source} phrase into {target}. "
        f"Give the natural, conversational wording a native speaker would use. "
        f"Reply with the translation alone: no explanation, no quotation marks."
    )
    return compose(profile, system, f"Translate this {source} phrase into {target}: {phrase}")


# --- shared ----------------------------------------------------------------


#: Added whenever context is sent, because without it a small model treats the
#: transcript as more of the thing to fix.
CONTEXT_ONLY = (
    "You will see the recent conversation, then one message to judge. "
    "The conversation is background only: never correct it, never comment on "
    "it, and never mention it in your answer."
)


def _judging(
    profile: ModelProfile,
    system: str,
    instruction: str,
    message: str,
    history,
    reminder: str = "",
) -> list[dict]:
    """Messages that put one turn in front of the model along with the
    conversation it came from — without the model mistaking one for the other.

    Context is needed because a reply misreads without it: "Sí, en Madrid" is a
    fragment on its own and a perfectly good answer to a question. But pasting
    the transcript into the instruction ("Conversation so far: ...") invites the
    model to correct the transcript instead, which it does: asked to correct
    "¡Bueno! Lo comprará. Gracias." it would explain what was wrong with a
    message three turns earlier, or answer the conversation rather than mark it.

    Sending the transcript as actual conversation turns keeps the two apart.
    Measured over three trials of five cases on translategemma:12b, that took
    corrections caught from 7/9 to 9/9 and sentences correctly left alone from
    3/6 to 5/6; on phi4-mini:3.8b, from 2/9 to 5/9 with the stray commentary
    gone. See tools/compare_models.py.

    A reminder rides on the final turn rather than only in the system prompt,
    because the transcript sits between the two and a small model answers in the
    language it has just been reading: told once at the top to explain in
    English, phi4-mini then explained in Spanish on 3 of 6 mid-conversation
    hints, and none once told again at the end.
    """
    turns = [m for m in (history or []) if (m.get("content") or "").strip()]
    # The message being judged is usually the last thing in the transcript;
    # drop it rather than showing it twice. Checked rather than assumed, so a
    # caller that passes history without it still works.
    if turns and (turns[-1].get("content") or "").strip() == (message or "").strip():
        turns = turns[:-1]

    tail = f"\n\n{reminder}" if reminder else ""

    recent = turns[-_context_turns(profile) :]
    if not recent:
        return compose(profile, system, f"{instruction}: {message}{tail}")

    opening = compose(profile, f"{system}\n\n{CONTEXT_ONLY}", "Here is the conversation so far.")
    return [
        *opening,
        *({"role": m["role"], "content": m["content"]} for m in recent),
        {
            "role": "user",
            "content": f"Now, {instruction}, and only this message: {message}{tail}",
        },
    ]


def _context_turns(profile: ModelProfile) -> int:
    """Fewer turns than the conversation itself carries: the critic needs the
    thread of the exchange, not the whole of it."""
    return max(2, profile.history_messages // 3)
