"""Heuristic hallucination baseline after Frieske and Shi.

A hypothesis is flagged when WER > threshold, Flan-T5 perplexity of the
hypothesis < threshold, and the cosine similarity of mean-pooled Flan-T5
encoder states of reference and hypothesis < threshold.

Scale caveat: the experiment script compared jiwer's WER, a fraction in [0, 1]
(and rarely above 1), against thresholds of 20-80 meant as percentages, so the
WER condition almost never held and the heuristic flagged nothing. That is the
behavior behind the paper's 0.00 Human-Heuristic agreement, and it remains the
default (``--wer-scale fraction``). ``--wer-scale percent`` compares WER x 100.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
from tqdm import tqdm

from speech_hallucinations.wer import iter_csv_files, normalize

logger = logging.getLogger(__name__)

SWEEP_WER_THRESHOLDS = tuple(range(20, 81, 10))
SWEEP_COSINE_THRESHOLDS = tuple(np.round(np.arange(0.1, 0.6, 0.1), 1))


@dataclass(frozen=True, slots=True)
class Thresholds:
    wer: float = 30
    cosine: float = 0.2
    perplexity: float = 200


def wer_fraction(reference: str, hypothesis: str) -> float:
    return jiwer.wer(normalize(reference), normalize(hypothesis))


def is_hallucination(
    wer: float, perplexity: float, cosine: float, thresholds: Thresholds, wer_scale: str
) -> bool:
    # "fraction" compares jiwer's 0-1 WER with a percentage threshold, exactly as the
    # experiment script did; it reproduces the paper's heuristic numbers (see module docstring).
    if wer_scale == "percent":
        wer = wer * 100
    elif wer_scale != "fraction":
        raise ValueError(f"Unknown WER scale: {wer_scale}")
    return (
        wer > thresholds.wer and perplexity < thresholds.perplexity and cosine < thresholds.cosine
    )


class FlanT5Scorer:
    def __init__(self, model_name: str, device: str):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()

    def _forward(self, text: str):
        import torch

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, labels=input_ids)
        return output.loss, output.encoder_last_hidden_state.mean(dim=1)

    def score(self, reference: str, hypothesis: str) -> tuple[float, float]:
        """Return (perplexity of the hypothesis, reference-hypothesis cosine)."""
        import torch

        loss, hypothesis_state = self._forward(hypothesis)
        _, reference_state = self._forward(reference)
        cosine = torch.nn.functional.cosine_similarity(hypothesis_state, reference_state, dim=1)
        return torch.exp(loss).item(), cosine.item()


def random_hypothesis_pool(frame: pd.DataFrame, num_samples: int, seed: int) -> pd.DataFrame:
    """Pair references with hypotheses drawn from other rows, as in the benchmark.

    Every pair is treated as a hallucination, so the flagged fraction is recall.
    """
    frame = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled = frame.iloc[:num_samples].copy()
    sampled["hypothesis"] = (
        frame["hypothesis"].sample(n=len(sampled), replace=False, random_state=seed).values
    )
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="CSVs with reference/hypothesis")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/heuristic"))
    parser.add_argument("--wer-threshold", type=float, default=30)
    parser.add_argument("--cosine-threshold", type=float, default=0.2)
    parser.add_argument("--perplexity-threshold", type=float, default=200)
    parser.add_argument("--wer-scale", choices=["fraction", "percent"], default="fraction")
    parser.add_argument(
        "--sweep", action="store_true", help="Also report the original WER x cosine grid"
    )
    parser.add_argument(
        "--random-hypotheses",
        type=int,
        metavar="N",
        help="Benchmark mode: N references paired with shuffled hypotheses",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scorer-model", default="google/flan-t5-small")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    frame = pd.concat(
        [pd.read_csv(path) for path in iter_csv_files(args.inputs)], ignore_index=True
    )
    frame = frame.dropna(subset=["reference", "hypothesis"])
    if args.random_hypotheses:
        frame = random_hypothesis_pool(frame, args.random_hypotheses, args.seed)

    scorer = FlanT5Scorer(args.scorer_model, args.device)
    features = []
    for reference, hypothesis in tqdm(
        zip(frame["reference"], frame["hypothesis"], strict=True), total=len(frame)
    ):
        perplexity, cosine = scorer.score(str(reference), str(hypothesis))
        features.append((wer_fraction(str(reference), str(hypothesis)), perplexity, cosine))
    frame[["heuristic_wer", "flan_t5_perplexity", "flan_t5_cosine"]] = features

    def flag(thresholds: Thresholds) -> pd.Series:
        return pd.Series(
            [is_hallucination(*row, thresholds, args.wer_scale) for row in features],
            index=frame.index,
        )

    main_thresholds = Thresholds(
        args.wer_threshold, args.cosine_threshold, args.perplexity_threshold
    )
    frame["heuristic_label"] = flag(main_thresholds).map(
        {True: "Hallucination", False: "Non-Hallucination"}
    )
    summary = {
        "wer_scale": args.wer_scale,
        "thresholds": asdict(main_thresholds),
        "flagged_fraction": float((frame["heuristic_label"] == "Hallucination").mean()),
    }
    if args.sweep:
        summary["sweep"] = [
            {
                "wer": wer_threshold,
                "cosine": float(cosine_threshold),
                "flagged_fraction": float(
                    flag(
                        Thresholds(wer_threshold, cosine_threshold, args.perplexity_threshold)
                    ).mean()
                ),
            }
            for wer_threshold in SWEEP_WER_THRESHOLDS
            for cosine_threshold in SWEEP_COSINE_THRESHOLDS
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "heuristic_labels.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Flagged %.4f of %d rows", summary["flagged_fraction"], len(frame))


if __name__ == "__main__":
    main()
