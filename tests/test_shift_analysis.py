import math

import numpy as np
import pandas as pd
import pytest

from speech_hallucinations.correlation import alpha_table, correlation, degradation_table
from speech_hallucinations.divergences import cmd, coral
from speech_hallucinations.embeddings import domain_key, parse_embeddings
from speech_hallucinations.shifts import domain_shift, match_sizes


def test_cmd_is_zero_for_identical_samples_and_positive_under_shift():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(200, 8))
    assert cmd(source, source) == pytest.approx(0.0)
    assert coral(source, source) == pytest.approx(0.0)
    assert cmd(source, source + 1.0) > 0.5
    assert coral(source, source * 3.0) > 0


def test_cmd_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(300, 4))
    target = rng.normal(size=(300, 4))
    assert cmd(source, target + 0.5) < cmd(source, target + 2.0)


def test_mean_variant_reduces_to_distance_between_means():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(50, 6))
    target = rng.normal(loc=1.0, size=(80, 6))
    shift = domain_shift(source, target, np.random.default_rng(0))
    expected = np.linalg.norm(source.mean(axis=0) - target.mean(axis=0))
    assert shift["CMD"]["mean"] == pytest.approx(expected)
    assert shift["Coral"]["mean"] == pytest.approx(0.0)


def test_subsampling_matches_sizes_and_is_seeded():
    source, target = np.arange(20).reshape(10, 2), np.arange(8).reshape(4, 2)
    first = match_sizes(source, target, np.random.default_rng(3))
    second = match_sizes(source, target, np.random.default_rng(3))
    assert first[0].shape == first[1].shape == (4, 2)
    np.testing.assert_array_equal(first[0], second[0])


def test_embedding_files_parse_and_map_to_result_keys(tmp_path):
    column = pd.Series(["[[0.1, 0.2, 0.3]]", "not a vector", "[[0.4, 0.5, 0.6]]"])
    assert parse_embeddings(column).shape == (2, 3)
    assert domain_key(tmp_path / "kresnik_librispeech_asr_test.csv") == "librispeech_asr_test"
    assert domain_key(tmp_path / "macabdul9_salt-multispeaker-eng.csv") == "salt-multispeaker-eng"


def test_alpha_is_pearson_by_default():
    shift = [0.1, 0.2, 0.3, 0.4]
    assert correlation(shift, [1, 2, 3, 4]) == pytest.approx(1.0)
    assert correlation(shift, [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert correlation(shift, [1, 4, 9, 16]) < 1.0
    assert correlation(shift, [1, 4, 9, 16], method="spearman") == pytest.approx(1.0)
    assert math.isnan(correlation(shift, [2, 2, 2, 2]))


def _judged(wer, hallucinations, n=4):
    labels = ["Hallucination Error"] * hallucinations + ["No Error"] * (n - hallucinations)
    return pd.DataFrame({"WER": [wer] * n, "gpt-4o-mini_coarsegrained": labels})


def test_degradation_is_target_minus_source():
    results = {
        ("m", "librispeech_asr_test"): _judged(5.0, 0),
        ("m", "ami"): _judged(25.0, 1),
        ("m", "GLOBE"): _judged(15.0, 2),
        ("m", "slue"): _judged(35.0, 3),
    }
    shifts = {"ami": 2.0, "GLOBE": 1.0, "slue": 3.0}
    table = degradation_table(
        results, shifts, "librispeech_asr_test", ["ami", "GLOBE", "slue"], "gpt-4o-mini", "coarse"
    )
    assert table.set_index("dataset")["WERD"].to_dict() == {
        "ami": 20.0,
        "GLOBE": 10.0,
        "slue": 30.0,
    }
    assert table.set_index("dataset")["HERD"].to_dict() == {
        "ami": 25.0,
        "GLOBE": 50.0,
        "slue": 75.0,
    }
    alphas = alpha_table(table, "pearson").set_index("model")
    assert alphas.loc["m", "alpha_WERD"] == pytest.approx(1.0)
