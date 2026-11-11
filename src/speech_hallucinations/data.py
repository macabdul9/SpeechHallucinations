from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASETS_CONFIG = REPO_ROOT / "configs" / "datasets.json"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    config: str | None
    split: str
    text_column: str
    audio_column: str = "audio"

    @property
    def short_name(self) -> str:
        """Last component of the Hub ID; used for result file names and table keys."""
        return self.name.split("/")[-1]


def load_dataset_specs(path: Path, names: list[str] | None = None) -> list[DatasetSpec]:
    entries = json.loads(Path(path).read_text())
    specs = [DatasetSpec(name=name, **info) for name, info in entries.items()]
    if names is None:
        return specs
    by_name = {spec.name: spec for spec in specs} | {spec.short_name: spec for spec in specs}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise KeyError(f"Datasets not in {path}: {', '.join(missing)}")
    return [by_name[name] for name in names]


def select_samples(dataset, num_samples: int, sampling: str, seed: int):
    """Take `num_samples` rows, either the first rows or after a seeded shuffle.

    The original transcription runs used the first rows of every configured
    dataset; embedding extraction and perturbation shuffled with seed 42.
    """
    if sampling == "shuffle":
        dataset = dataset.shuffle(seed=seed)
    elif sampling != "first":
        raise ValueError(f"Unknown sampling mode: {sampling}")
    return dataset.select(range(min(num_samples, len(dataset))))


def load_audio_dataset(spec: DatasetSpec, resample_to: int | None = None):
    from datasets import Audio, load_dataset

    dataset = load_dataset(spec.name, spec.config, split=spec.split)
    if resample_to is not None:
        dataset = dataset.cast_column(spec.audio_column, Audio(sampling_rate=resample_to))
    return dataset


def model_short_name(model_name_or_path: str) -> str:
    return model_name_or_path.rstrip("/").split("/")[-1]
