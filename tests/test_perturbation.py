import random

import numpy as np
import pytest

from speech_hallucinations.perturbation import fix_length, perturb

SAMPLING_RATE = 16000


@pytest.fixture
def audio():
    t = np.linspace(0, 1, SAMPLING_RATE, endpoint=False)
    return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_fix_length_trims_and_zero_pads():
    np.testing.assert_array_equal(fix_length(np.arange(5), 3), [0, 1, 2])
    np.testing.assert_array_equal(fix_length(np.arange(3), 5), [0, 1, 2, 0, 0])


def test_clean_and_noisy_audio_keep_shape(audio):
    param_rng, noise_rng = random.Random(42), np.random.default_rng(42)
    clean = perturb(audio, SAMPLING_RATE, "none", param_rng, noise_rng)
    noisy = perturb(audio, SAMPLING_RATE, "white_noise", param_rng, noise_rng)
    np.testing.assert_array_equal(clean, audio)
    assert noisy.shape == audio.shape
    assert 0.005 < np.std(noisy - audio) < 0.02


def test_unknown_perturbation_is_rejected(audio):
    with pytest.raises(ValueError):
        perturb(audio, SAMPLING_RATE, "echo", random.Random(0), np.random.default_rng(0))


@pytest.mark.parametrize("kind", ["time_stretch", "pitch_shift"])
def test_librosa_perturbations_restore_length(audio, kind):
    pytest.importorskip("librosa")
    out = perturb(audio, SAMPLING_RATE, kind, random.Random(42), np.random.default_rng(42))
    assert out.shape == audio.shape
