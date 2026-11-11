import numpy as np
import pandas as pd
import pytest

from speech_hallucinations.heuristic import (
    Thresholds,
    is_hallucination,
    random_hypothesis_pool,
    wer_fraction,
)
from speech_hallucinations.human_eval import filter_candidates, plant_shuffled_hypotheses

PAPER = Thresholds(wer=30, cosine=0.2, perplexity=200)


def test_paper_thresholds_with_percent_scale():
    assert is_hallucination(0.45, 50.0, 0.1, PAPER, "percent")
    assert not is_hallucination(0.30, 50.0, 0.1, PAPER, "percent")
    assert not is_hallucination(0.45, 200.0, 0.1, PAPER, "percent")
    assert not is_hallucination(0.45, 50.0, 0.2, PAPER, "percent")


def test_preserved_fraction_scale_never_fires_for_realistic_wer():
    # jiwer WER is a fraction; against a threshold of 30 it needs 30x more
    # errors than reference words, which is why the paper's heuristic flagged nothing.
    for wer in (0.0, 0.6, 1.0, 5.0):
        assert not is_hallucination(wer, 1.0, 0.0, PAPER, "fraction")
    assert wer_fraction("the cat sat", "a dog ran") == pytest.approx(1.0)


def test_unknown_wer_scale_is_rejected():
    with pytest.raises(ValueError):
        is_hallucination(0.5, 1.0, 0.0, PAPER, "ratio")


def test_random_hypotheses_keep_references_and_draw_from_the_pool():
    frame = pd.DataFrame({"reference": list("abcdef"), "hypothesis": list("uvwxyz")})
    pool = random_hypothesis_pool(frame, 4, seed=42)
    assert len(pool) == 4
    assert set(pool["hypothesis"]) <= set("uvwxyz")
    assert pool.equals(random_hypothesis_pool(frame, 4, seed=42))


def test_human_eval_filters_and_plants_shuffled_hypotheses():
    pool = pd.DataFrame(
        {
            "reference": ["one two", "one two", "single", "one two", "one two"],
            "hypothesis": ["a b", "a b", "a b", "word", " ".join(["w"] * 100)],
            "WER": [80.0, 50.0, 90.0, 90.0, 90.0],
        }
    )
    assert filter_candidates(pool).index.tolist() == [0]

    np.random.seed(42)
    sample = pd.DataFrame({"hypothesis": [f"h{i}" for i in range(10)]})
    planted = plant_shuffled_hypotheses(sample, 4)
    flagged = planted["flag"] == "hallucination"
    assert flagged.sum() == 4
    assert sorted(planted.loc[flagged, "hypothesis"]) == sorted(sample.loc[flagged, "hypothesis"])
    assert (planted.loc[~flagged, "hypothesis"] == sample.loc[~flagged, "hypothesis"]).all()
