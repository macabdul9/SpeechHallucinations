"""Per-utterance WER and CER after Whisper's BasicTextNormalizer.

Scores are percentages per row. Dataset-level WER in the paper is the mean of
these per-row values, not a corpus-level WER.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import jiwer
import pandas as pd

logger = logging.getLogger(__name__)


def normalize(text: str) -> str:
    """Whisper's ``BasicTextNormalizer()`` with default arguments.

    Copied from openai/whisper (MIT) so WER does not require transformers; the
    experiments imported the identical class from transformers.
    """
    text = str(text).lower()
    text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
    text = re.sub(r"\(([^)]+?)\)", "", text)
    text = "".join(
        " " if unicodedata.category(char)[0] in "MSP" else char
        for char in unicodedata.normalize("NFKC", text)
    ).lower()
    return re.sub(r"\s+", " ", text)


def utterance_metrics(reference: str, hypothesis: str) -> dict:
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    words = jiwer.process_words(
        reference=reference,
        hypothesis=hypothesis,
        reference_transform=jiwer.wer_contiguous,
        hypothesis_transform=jiwer.wer_contiguous,
    )
    return {
        "WER": round(words.wer * 100, 2),
        "CER": round(jiwer.cer(reference=reference, hypothesis=hypothesis) * 100, 2),
        "nErrors": words.substitutions + words.insertions + words.deletions,
        "RefLength": len(words.references[0]),
    }


def add_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add WER/CER/nErrors/RefLength columns; rows missing either text stay empty."""
    frame = frame.copy()
    rows = {}
    for index, row in frame.iterrows():
        if pd.isna(row["reference"]) or pd.isna(row["hypothesis"]):
            continue
        rows[index] = utterance_metrics(row["reference"], row["hypothesis"])
    metrics = pd.DataFrame.from_dict(rows, orient="index")
    for column in ("WER", "CER", "nErrors", "RefLength"):
        frame[column] = metrics[column] if column in metrics else None
    return frame


def iter_csv_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        files.extend(sorted(path.rglob("*.csv")) if path.is_dir() else [path])
    return [file for file in files if not file.stem.endswith("_checkpoint")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Add per-row WER/CER to transcription CSVs")
    parser.add_argument("paths", nargs="+", type=Path, help="CSV files or directories")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for path in iter_csv_files(args.paths):
        frame = pd.read_csv(path)
        if not {"reference", "hypothesis"} <= set(frame.columns):
            logger.info("Skipping %s: no reference/hypothesis columns", path)
            continue
        add_metrics(frame).to_csv(path, index=False)
        logger.info("Updated %s", path)


if __name__ == "__main__":
    main()
