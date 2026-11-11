import math

import pandas as pd
import pytest

from speech_hallucinations.labels import COARSE_LABELS, FINE_LABELS, parse_label
from speech_hallucinations.metrics import hallucination_error_rate, judge_column, summarize
from speech_hallucinations.wer import add_metrics, normalize, utterance_metrics


def test_normalizer_drops_case_punctuation_and_bracketed_words():
    assert normalize("Hello, World! [noise] (laughs)").split() == ["hello", "world"]
    assert normalize("Mr. Smith's") == "mr smith s"


def test_wer_is_computed_after_normalization_and_reported_in_percent():
    assert utterance_metrics("The cat sat.", "the CAT sat") == {
        "WER": 0.0,
        "CER": 0.0,
        "nErrors": 0,
        "RefLength": 3,
    }
    metrics = utterance_metrics("the cat sat", "the cat sat on the mat")
    assert metrics["WER"] == 100.0
    assert metrics["nErrors"] == 3


def test_rows_missing_text_keep_empty_metrics():
    frame = pd.DataFrame({"reference": ["a b", None], "hypothesis": ["a c", "x"]})
    result = add_metrics(frame)
    assert result.loc[0, "WER"] == 50.0
    assert pd.isna(result.loc[1, "WER"])


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Hallucination Error", "Hallucination"),
        ("  **Non-Hallucination Error**\n", "Non-Hallucination"),
        ("Output: No Error.", "No Error"),
        ("APIError", None),
        ("The output contains a Hallucination Error", None),
        (float("nan"), None),
    ],
)
def test_coarse_labels(response, expected):
    assert parse_label(response, "coarse") == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Phonetic Error", "Phonetic"),
        ('"Oscillation Error"', "Oscillation"),
        ("language error", "Language"),
        ("Hallucination Error", "Hallucination"),
        ("No Error", "No Error"),
        ("Non-Hallucination Error", None),
    ],
)
def test_fine_labels(response, expected):
    assert parse_label(response, "fine") == expected


def test_every_canonical_label_round_trips_from_prompt_wording():
    for label in COARSE_LABELS:
        wording = label if label == "No Error" else f"{label} Error"
        assert parse_label(wording, "coarse") == label
    for label in FINE_LABELS:
        wording = label if label == "No Error" else f"{label} Error"
        assert parse_label(wording, "fine") == label


def test_her_counts_unparseable_rows_in_the_denominator():
    responses = ["Hallucination Error", "No Error", "APIError", None]
    assert hallucination_error_rate(responses, "coarse") == 25.0
    assert math.isnan(hallucination_error_rate([], "coarse"))


def test_summary_reads_both_judge_columns():
    frame = pd.DataFrame(
        {
            "WER": [10.0, 30.0],
            "CER": [5.0, 5.0],
            judge_column("gpt-4o-mini", "coarse"): ["Hallucination Error", "No Error"],
            judge_column("gpt-4o-mini", "fine"): ["Phonetic Error", "Phonetic Error"],
        }
    )
    summary = summarize(frame, "gpt-4o-mini")
    assert summary == {"n": 2, "WER": 20.0, "CER": 5.0, "HER_coarse": 50.0, "HER_fine": 0.0}
