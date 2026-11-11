"""Extract SONAR speech and text embeddings for each evaluation dataset.

Writes ``<output-dir>/<org>_<name>.csv`` with ``id``, ``split``,
``speech_embedding``, and ``text_embedding`` (JSON lists, shape 1 x 1024).
Text is embedded as given in the dataset, without normalization.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from speech_hallucinations.data import (
    DEFAULT_DATASETS_CONFIG,
    DatasetSpec,
    load_audio_dataset,
    load_dataset_specs,
    select_samples,
)

logger = logging.getLogger(__name__)


def embedding_file_name(spec: DatasetSpec) -> str:
    return f"{spec.name.replace('/', '_')}.csv"


def domain_key(path: Path) -> str:
    """kresnik_librispeech_asr_test.csv -> librispeech_asr_test, the result-file key."""
    return path.stem.split("_", 1)[1]


def parse_embeddings(column: pd.Series) -> np.ndarray:
    """Stack JSON-list embeddings into (N, D), skipping unparseable rows."""
    vectors = []
    for value in column:
        try:
            vectors.append(np.asarray(json.loads(value), dtype=np.float64).reshape(-1))
        except (TypeError, ValueError):
            continue
    return np.stack(vectors)


def load_sonar(device: str):
    import torch
    from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    device = torch.device(device)
    speech = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng", device=device)
    text = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device
    )
    return speech, text


def embed_dataset(spec: DatasetSpec, dataset, speech_pipeline, text_pipeline) -> pd.DataFrame:
    import torch

    rows = []
    for index in tqdm(range(len(dataset)), desc=spec.short_name):
        sample = dataset[index]
        # Very long or silent clips can fail inside SONAR; the original run skipped them.
        try:
            audio = torch.tensor(sample[spec.audio_column]["array"])
            speech = speech_pipeline.predict([audio.unsqueeze(0)])
            text = text_pipeline.predict([sample[spec.text_column]], source_lang="eng_Latn")
        except Exception as error:  # noqa: BLE001
            logger.warning("Skipping %s[%d]: %s", spec.short_name, index, error)
            continue
        rows.append(
            {
                "id": sample.get("id", index),
                "split": spec.split,
                "speech_embedding": json.dumps(speech.tolist()),
                "text_embedding": json.dumps(text.tolist()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--sampling", choices=["first", "shuffle"], default="shuffle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample-to", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/embeddings"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    speech_pipeline, text_pipeline = load_sonar(args.device)
    for spec in load_dataset_specs(args.datasets_config, args.datasets):
        output_file = args.output_dir / embedding_file_name(spec)
        if output_file.exists():
            logger.info("Skipping %s: %s exists", spec.name, output_file)
            continue
        dataset = load_audio_dataset(spec, args.resample_to)
        dataset = select_samples(dataset, args.num_samples, args.sampling, args.seed)
        embed_dataset(spec, dataset, speech_pipeline, text_pipeline).to_csv(
            output_file, index=False
        )


if __name__ == "__main__":
    main()
