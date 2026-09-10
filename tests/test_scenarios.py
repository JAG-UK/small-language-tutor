"""Choosing something to talk about, and choosing it for a reason."""

import random

import pytest

from scenarios import (
    ACCENTS,
    AGREEMENT,
    ARTICLES,
    CAPITALS,
    DECK,
    SER_ESTAR,
    choose,
    weak_areas,
    what_it_shows,
)


def corrections(*pairs):
    return [{"message": wrote, "corrected": right} for wrote, right in pairs]


class TestWhatOneMistakeShows:
    @pytest.mark.parametrize(
        "wrote,should_be,area",
        [
            ("soy", "estoy", SER_ESTAR),
            ("está", "es", SER_ESTAR),
            ("el", "la", ARTICLES),
            ("una", "un", ARTICLES),
            ("anos", "años", ACCENTS),
            ("aqui", "aquí", ACCENTS),
            ("madrid", "Madrid", CAPITALS),
            ("alto", "alta", AGREEMENT),
            ("bonito", "bonita", AGREEMENT),
            ("gusta", "gustan", AGREEMENT),
            ("espanol", "española", AGREEMENT),
        ],
    )
    def test_the_cases_it_is_sure_about(self, wrote, should_be, area):
        assert what_it_shows(wrote, should_be) == area

    @pytest.mark.parametrize(
        "wrote,should_be",
        [
            # Tense, not agreement. A looser rule called both of these agreement
            # — comer/comí on a shared stem, hablo/hablé because folding away
            # the accent left an ending that looked like a gender one.
            ("comer", "comí"),
            ("hablo", "hablé"),
            ("tengo", "tenía"),
            ("vivo", "vive"),
            # A different word entirely.
            ("es", "fue"),
            ("hola", "adiós"),
            # Nothing changed.
            ("casa", "casa"),
        ],
    )
    def test_says_nothing_rather_than_guessing(self, wrote, should_be):
        """A scenario chosen for the wrong reason is a wasted conversation, and
        there is nothing to check a guess against."""
        assert what_it_shows(wrote, should_be) is None


class TestWhatKeepsHappening:
    def test_needs_more_than_one_to_count(self):
        once = corrections(("yo soy cansado", "yo estoy cansado"))
        assert weak_areas(once) == []

    def test_twice_is_a_habit(self):
        twice = corrections(("yo soy cansado", "yo estoy cansado"),
                            ("ella es enferma", "ella está enferma"))
        assert weak_areas(twice) == [SER_ESTAR]

    def test_commonest_first(self):
        mixed = corrections(
            ("yo soy cansado", "yo estoy cansado"),
            ("ella es enferma", "ella está enferma"),
            ("el es alto", "él está alto"),
            ("tengo dos anos", "tengo dos años"),
            ("vivo aqui", "vivo aquí"),
        )
        areas = weak_areas(mixed)
        assert areas[0] == SER_ESTAR
        assert ACCENTS in areas

    def test_nothing_to_go_on(self):
        assert weak_areas([]) == []


class TestChoosingASituation:
    def test_picks_one_that_exercises_the_weak_area(self):
        twice = corrections(("yo soy cansado", "yo estoy cansado"),
                            ("ella es enferma", "ella está enferma"))
        for _ in range(10):
            chosen = choose(twice)
            assert SER_ESTAR in chosen["elicits"]
            assert chosen["chosen_for"] == SER_ESTAR

    def test_says_nothing_was_targeted_when_nothing_stands_out(self):
        assert choose([])["chosen_for"] is None

    def test_still_picks_something_with_no_history(self):
        assert choose([])["label"] in [s["label"] for s in DECK]

    def test_does_not_repeat_the_same_one_every_time(self):
        # Targeted, but not the same conversation twice.
        picks = {choose([], rng=random.Random(seed))["label"] for seed in range(20)}
        assert len(picks) > 3

    def test_can_be_told_what_to_avoid(self):
        everything_but = [s["label"] for s in DECK[:-1]]
        assert choose([], avoid=everything_but)["label"] == DECK[-1]["label"]

    def test_avoiding_everything_still_returns_something(self):
        assert choose([], avoid=[s["label"] for s in DECK])["label"]


class TestTheDeckItself:
    def test_every_situation_has_what_it_needs(self):
        for scenario in DECK:
            assert scenario["label"] and scenario["setting"]
            assert isinstance(scenario["elicits"], set)

    def test_labels_are_distinct(self):
        assert len({s["label"] for s in DECK}) == len(DECK)

    def test_every_detectable_area_has_somewhere_to_practise_it(self):
        """A weak area with no matching situation would be detected and then
        quietly ignored."""
        covered = set().union(*(s["elicits"] for s in DECK))
        assert {ACCENTS, CAPITALS, SER_ESTAR, ARTICLES, AGREEMENT} <= covered

    def test_the_situation_is_described_to_the_model_not_to_the_learner(self):
        # It is an instruction: second person, addressed to the model.
        assert all(s["setting"].startswith("You ") for s in DECK)
