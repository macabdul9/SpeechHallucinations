"""Transcribe evaluation datasets with one ASR model.

Writes ``<output-dir>/<model>/<dataset>.csv`` with ``audio_id``, ``reference``,
and ``hypothesis`` columns. Existing result files are skipped, and partial
runs resume from ``<dataset>_checkpoint.csv``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from speech_hallucinations.asr_models import MODEL_TYPES, load_model, transcribe
from speech_hallucinations.data import (
    DEFAULT_DATASETS_CONFIG,
    DatasetSpec,
    load_audio_dataset,
    load_dataset_specs,
    model_short_name,
    select_samples,
)

logger = logging.getLogger(__name__)


def transcribe_dataset(asr, spec: DatasetSpec, dataset, output_file: Path, checkpoint_every: int):
    checkpoint_file = output_file.with_name(f"{output_file.stem}_checkpoint.csv")
    rows = pd.read_csv(checkpoint_file).to_dict("records") if checkpoint_file.exists() else []
    start = int(rows[-1]["audio_id"].rsplit("_", 1)[1]) + 1 if rows else 0
    skipped = 0
    for index in tqdm(range(start, len(dataset)), desc=spec.short_name):
        example = dataset[index]
        # Some long or malformed clips fail inside model code; the original runs
        # skipped them, so result files can hold fewer rows than requested.
        try:
            hypothesis = transcribe(asr, example[spec.audio_column]["array"])
        except Exception as error:  # noqa: BLE001
            logger.warning("Skipping %s[%d]: %s", spec.short_name, index, error)
            skipped += 1
            continue
        rows.append(
            {
                "audio_id": f"{spec.short_name}_{index}",
                "reference": example[spec.text_column],
                "hypothesis": hypothesis,
            }
        )
        if (index + 1) % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(checkpoint_file, index=False)

    if rows:
        pd.DataFrame(rows).to_csv(output_file, index=False)
    checkpoint_file.unlink(missing_ok=True)
    logger.info("%s: %d transcribed, %d skipped", spec.short_name, len(rows), skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--model", required=True, help="Hub ID or local checkpoint")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASETS_CONFIG)
    parser.add_argument(
        "--datasets", nargs="+", help="Subset of the config, by Hub ID or short name"
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument(
        "--sampling",
        choices=["first", "shuffle"],
        default="first",
        help="'first' takes the first rows unshuffled, as in the paper's runs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for --sampling shuffle")
    parser.add_argument(
        "--resample-to",
        type=int,
        default=None,
        help="Resample audio (e.g. 16000). Off by default to match the original runs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/asr"))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    specs = load_dataset_specs(args.datasets_config, args.datasets)
    model_dir = args.output_dir / model_short_name(args.model)
    model_dir.mkdir(parents=True, exist_ok=True)
    asr = load_model(args.model_type, args.model, args.device)

    for spec in specs:
        output_file = model_dir / f"{spec.short_name}.csv"
        if output_file.exists():
            logger.info("Skipping %s: %s exists", spec.name, output_file)
            continue
        dataset = load_audio_dataset(spec, args.resample_to)
        dataset = select_samples(dataset, args.num_samples, args.sampling, args.seed)
        transcribe_dataset(asr, spec, dataset, output_file, args.checkpoint_every)


if __name__ == "__main__":
    main()
