"""HER and WER tables per model and dataset from judged transcription CSVs.

Expects the layout written by ``transcribe``: ``<results-dir>/<model>/<dataset>.csv``
with ``WER``/``CER`` columns from ``wer`` and judge columns from ``judge``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from speech_hallucinations.labels import HALLUCINATION, parse_label

logger = logging.getLogger(__name__)

GRANULARITY_PROMPTS = {"coarse": "coarsegrained", "fine": "finegrained"}


def judge_column(judge_model: str, granularity: str) -> str:
    return f"{judge_model}_{GRANULARITY_PROMPTS[granularity]}"


def hallucination_error_rate(responses: Iterable, granularity: str) -> float:
    """Percentage of rows labelled Hallucination.

    The denominator is every row, including API errors and unparseable answers,
    matching the paper's "hallucination errors / total examples".
    """
    responses = list(responses)
    if not responses:
        return float("nan")
    hallucinations = sum(parse_label(r, granularity) == HALLUCINATION for r in responses)
    return 100 * hallucinations / len(responses)


def summarize(frame: pd.DataFrame, judge_model: str) -> dict:
    summary = {"n": len(frame)}
    for metric in ("WER", "CER"):
        if metric in frame:
            summary[metric] = pd.to_numeric(frame[metric], errors="coerce").mean()
    for granularity in GRANULARITY_PROMPTS:
        column = judge_column(judge_model, granularity)
        if column in frame:
            summary[f"HER_{granularity}"] = hallucination_error_rate(frame[column], granularity)
    return summary


def load_results(results_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    results = {}
    for path in sorted(results_dir.glob("*/*.csv")):
        if path.stem.endswith("_checkpoint"):
            continue
        results[(path.parent.name, path.stem)] = pd.read_csv(path)
    if not results:
        raise FileNotFoundError(f"No <model>/<dataset>.csv files under {results_dir}")
    return results


def results_table(results_dir: Path, judge_model: str) -> pd.DataFrame:
    rows = [
        {"model": model, "dataset": dataset, **summarize(frame, judge_model)}
        for (model, dataset), frame in load_results(results_dir).items()
    ]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate WER/CER/HER per model and dataset")
    parser.add_argument("--results-dir", type=Path, default=Path("runs/asr"))
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Prefix of judge columns")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tables"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    table = results_table(args.results_dir, args.judge_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "results.csv", index=False)
    for metric in ("WER", "HER_coarse", "HER_fine"):
        if metric in table:
            wide = table.pivot(index="model", columns="dataset", values=metric).round(2)
            wide.to_csv(args.output_dir / f"{metric}.csv")
    logger.info("Wrote %d rows to %s", len(table), args.output_dir)


if __name__ == "__main__":
    main()
