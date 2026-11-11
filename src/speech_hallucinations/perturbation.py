"""Transcribe clean and perturbed audio (white noise, time stretch, pitch shift).

Draws ``--num-samples`` clips per configured dataset (seeded shuffle), pools
them, and keeps ``--num-samples`` clips of the pooled set after another seeded
shuffle. Each clip is transcribed once per perturbation. Writes
``<output-dir>/<model>/perturbation_output.csv`` with WER/CER, ``reference``,
``hypothesis``, and ``perturbation``; judge it with ``judge`` afterwards.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from speech_hallucinations.asr_models import MODEL_TYPES, load_model, transcribe
from speech_hallucinations.data import (
    DEFAULT_DATASETS_CONFIG,
    load_audio_dataset,
    load_dataset_specs,
    model_short_name,
    select_samples,
)
from speech_hallucinations.wer import utterance_metrics

logger = logging.getLogger(__name__)

PERTURBATIONS = ("none", "white_noise", "time_stretch", "pitch_shift")
NOISE_STD = 0.01
STRETCH_RANGE = (0.9, 1.1)
PITCH_STEPS = (-2, 2)


def fix_length(audio: np.ndarray, size: int) -> np.ndarray:
    """Trim or zero-pad at the end, like ``librosa.util.fix_length``."""
    if len(audio) >= size:
        return audio[:size]
    return np.pad(audio, (0, size - len(audio)))


def add_white_noise(audio: np.ndarray, noise_rng: np.random.Generator) -> np.ndarray:
    return audio + noise_rng.normal(0, NOISE_STD, audio.shape)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    import librosa

    return librosa.effects.time_stretch(y=audio, rate=rate)


def pitch_shift(audio: np.ndarray, sampling_rate: int, n_steps: int) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(y=audio, sr=sampling_rate, n_steps=n_steps)


def perturb(
    audio,
    sampling_rate: int,
    kind: str,
    param_rng: random.Random,
    noise_rng: np.random.Generator,
) -> np.ndarray:
    """Apply one perturbation and restore the original length.

    Parameters are drawn from `param_rng` in the same order as the original
    script's global ``random`` calls, so a fixed seed reproduces them.
    """
    audio = np.asarray(audio)
    if kind == "none":
        modified = audio
    elif kind == "white_noise":
        modified = add_white_noise(audio, noise_rng)
    elif kind == "time_stretch":
        modified = time_stretch(audio, param_rng.uniform(*STRETCH_RANGE))
    elif kind == "pitch_shift":
        modified = pitch_shift(audio, sampling_rate, param_rng.randint(*PITCH_STEPS))
    else:
        raise ValueError(f"Unknown perturbation: {kind}")
    return fix_length(modified, len(audio))


def load_pooled_dataset(args: argparse.Namespace):
    from datasets import concatenate_datasets

    pooled = []
    for spec in load_dataset_specs(args.datasets_config, args.datasets):
        dataset = load_audio_dataset(spec, args.resample_to)
        dataset = select_samples(dataset, args.num_samples, "shuffle", args.seed)
        dataset = dataset.select_columns([spec.audio_column, spec.text_column])
        renames = {spec.audio_column: "audio", spec.text_column: "reference"}
        dataset = dataset.rename_columns({old: new for old, new in renames.items() if old != new})
        # Concatenation needs identical features; the first dataset's audio
        # feature (sampling rate included) is applied to the rest.
        if pooled:
            dataset = dataset.cast(pooled[0].features)
        pooled.append(dataset)
    return select_samples(concatenate_datasets(pooled), args.num_samples, "shuffle", args.seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--perturbations", nargs="+", choices=PERTURBATIONS, default=PERTURBATIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample-to", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/perturbation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    param_rng = random.Random(args.seed)
    # The original noise draws were unseeded; seeding them makes runs repeatable.
    noise_rng = np.random.default_rng(args.seed)
    dataset = load_pooled_dataset(args)
    asr = load_model(args.model_type, args.model, args.device)

    rows = []
    for example in tqdm(dataset, desc="clips"):
        audio, sampling_rate = example["audio"]["array"], example["audio"]["sampling_rate"]
        for kind in args.perturbations:
            hypothesis = transcribe(asr, perturb(audio, sampling_rate, kind, param_rng, noise_rng))
            rows.append(
                {
                    **utterance_metrics(example["reference"], hypothesis),
                    "reference": example["reference"],
                    "hypothesis": hypothesis,
                    "perturbation": kind,
                }
            )

    output_dir = args.output_dir / model_short_name(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "perturbation_output.csv", index=False)


if __name__ == "__main__":
    main()
