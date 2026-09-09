"""What a model family needs said to it, and how much it can keep in its head.

Swapping the model should not mean rewriting prompts. Families differ in ways
that matter to this app:

* **The system role.** Gemma 1-3 were trained with only `user` and `model`
  turns — Ollama will accept a system message and the model will under-weight
  it. Gemma 4 added a real system turn. Everything else here supports one.
* **How much instruction helps.** A 3B model follows three imperative lines
  better than nine numbered rules with sub-clauses; a 9B model uses the detail.
  Past a point, more instruction makes a small model worse, not better.
* **How much history to carry.** Conversation context costs tokens on every
  turn, and a small model with a modest window spends its attention on the
  transcript instead of the reply.
* **Whether it thinks first.** Reasoning models emit a chain of thought before
  answering, which is wasted on a chat reply and costs a lot of wall-clock —
  qwen3.5:9b took 70s on a sentence phi4-mini answered in 1s.

Add a family by adding a row to FAMILIES. The default is deliberately
conservative: assume a system role works and keep the prompt short, which is
the safe end for a model nobody has characterised.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ModelProfile:
    """How to talk to one family of models."""

    family: str
    #: False when the family has no system turn, so it must be folded into the
    #: first user message rather than silently under-weighted.
    supports_system_role: bool = True
    #: Prefer short imperative prompts over long numbered rules.
    terse: bool = True
    #: How many past messages to send as conversation context.
    history_messages: int = 10
    #: Emits a chain of thought before answering; slow for interactive use.
    reasoning: bool = False
    #: A translation specialist rather than a conversationalist.
    translation_only: bool = False
    #: Shown in the UI and the README when explaining a choice.
    note: str = ""


# Matched in order against the model name, so put the specific before the
# general — "gemma4" has to win before "gemma".
FAMILIES: list[tuple[str, ModelProfile]] = [
    (
        r"translategemma",
        ModelProfile(
            family="translategemma",
            supports_system_role=False,  # Gemma 3 underneath
            terse=True,
            history_messages=4,
            translation_only=True,
            note="Purpose-built translator. Excellent for the translate box, "
            "poor as a conversation partner or a grammar tutor.",
        ),
    ),
    (
        r"gemma-?[4-9]",
        ModelProfile(
            family="gemma4+",
            supports_system_role=True,
            terse=False,
            history_messages=10,
            note="Gemma 4 added a real system turn, unlike its predecessors.",
        ),
    ),
    (
        r"gemma",
        ModelProfile(
            family="gemma",
            supports_system_role=False,
            terse=False,
            history_messages=10,
            note="Gemma 1-3 know only user and model turns; the system prompt "
            "is folded into the first user message.",
        ),
    ),
    (
        r"phi\d*-mini-reasoning|deepseek-r1|^qwen3:|qwen3\.5",
        ModelProfile(
            family="reasoning",
            supports_system_role=True,
            terse=True,
            history_messages=6,
            reasoning=True,
            note="Thinks before answering. Accurate, but too slow for live "
            "conversation on a laptop.",
        ),
    ),
    (
        r"phi",
        ModelProfile(
            family="phi",
            supports_system_role=True,
            terse=True,
            history_messages=8,
            note="Small and quick. Follows short instructions well; long "
            "numbered rule lists make it worse, not better.",
        ),
    ),
    (
        r"llama|mistral|qwen",
        ModelProfile(
            family="general",
            supports_system_role=True,
            terse=False,
            history_messages=10,
        ),
    ),
]

DEFAULT_PROFILE = ModelProfile(
    family="unknown",
    supports_system_role=True,
    terse=True,  # the safe end for a model nobody has characterised
    history_messages=8,
    note="Not a family this app knows; assuming short prompts and a system role.",
)


def profile_for(model_name: str) -> ModelProfile:
    """The profile for a model name such as "phi4-mini:3.8b"."""
    name = (model_name or "").lower()
    for pattern, profile in FAMILIES:
        if re.search(pattern, name):
            return profile
    return DEFAULT_PROFILE


def compose(profile: ModelProfile, system: str, user: str) -> list[dict]:
    """Turn a system instruction and a user turn into messages this family
    will actually heed.

    Where there is no system role the instruction is folded into the user turn,
    which is what Gemma's own prompt guidance recommends — passing it as a
    system message instead leaves it technically present and largely ignored.
    """
    if profile.supports_system_role:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    return [{"role": "user", "content": f"{system}\n\n{user}"}]


def with_history(profile: ModelProfile, system: str, history: list[dict]) -> list[dict]:
    """Messages for a conversational turn: the instruction, then as much of the
    transcript as this family should carry."""
    recent = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])[-profile.history_messages :]
    ]
    if profile.supports_system_role:
        return [{"role": "system", "content": system}, *recent]

    if not recent:
        return [{"role": "user", "content": system}]
    # Fold the instruction into the earliest user turn we are sending, so it
    # stays in front of the model rather than being dropped with the system.
    folded = list(recent)
    for i, message in enumerate(folded):
        if message["role"] == "user":
            folded[i] = {"role": "user", "content": f"{system}\n\n{message['content']}"}
            return folded
    return [{"role": "user", "content": system}, *folded]
