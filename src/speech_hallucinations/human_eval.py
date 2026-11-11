"""Sample transcriptions for human annotation.

Pools every ``<model>/<dataset>.csv`` under the results directory, keeps rows
with WER > 60 whose reference and hypothesis have more than one word and whose
hypothesis has fewer than 100 words, samples ``--num-samples`` rows, then
replaces the hypotheses of ``--num-shuffled`` rows with a permutation of each
other to plant known hallucinations (``flag`` = "hallucination").
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from speech_hallucinations.metrics import load_results

logger = logging.getLogger(__name__)


def pool_results(results_dir: Path) -> pd.DataFrame:
    frames = [
        frame.assign(model_name=model, Dataset=dataset)
        for (model, dataset), frame in load_results(results_dir).items()
    ]
    return pd.concat(frames, ignore_index=True)


def filter_candidates(pool: pd.DataFrame, min_wer: float = 60) -> pd.DataFrame:
    pool = pool.dropna(subset=["reference", "hypothesis", "WER"])
    hypothesis_words = pool["hypothesis"].astype(str).str.split().str.len()
    reference_words = pool["reference"].astype(str).str.split().str.len()
    keep = (
        (pd.to_numeric(pool["WER"]) > min_wer)
        & (hypothesis_words > 1)
        & (reference_words > 1)
        & (hypothesis_words < 100)
    )
    return pool[keep]


def plant_shuffled_hypotheses(sample: pd.DataFrame, num_shuffled: int) -> pd.DataFrame:
    """Permute hypotheses among `num_shuffled` random rows; uses NumPy's global RNG."""
    sample = sample.copy()
    chosen = np.random.choice(sample.index, size=min(num_shuffled, len(sample)), replace=False)
    sample.loc[chosen, "hypothesis"] = sample.loc[chosen, "hypothesis"].sample(frac=1).values
    sample["flag"] = "unknown"
    sample.loc[chosen, "flag"] = "hallucination"
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=Path("runs/asr"))
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-shuffled", type=int, default=50)
    parser.add_argument("--min-wer", type=float, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/human_eval/samples.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # The original script seeded NumPy's global RNG with 42, which also drives the
    # unseeded pandas shuffle inside plant_shuffled_hypotheses.
    np.random.seed(args.seed)
    candidates = filter_candidates(pool_results(args.results_dir), args.min_wer)
    if len(candidates) < args.num_samples:
        raise ValueError(f"Only {len(candidates)} candidates; need {args.num_samples}")
    sample = candidates.sample(n=args.num_samples, random_state=args.seed)
    sample = plant_shuffled_hypotheses(sample, args.num_shuffled)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    logger.info("Wrote %d samples (%d shuffled) to %s", len(sample), args.num_shuffled, args.output)


if __name__ == "__main__":
    main()
