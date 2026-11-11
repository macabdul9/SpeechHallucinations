"""Distribution shift of each dataset from the LibriSpeech source domain.

For every embedding CSV, reports CMD and CORAL in two variants:

``mean``  between the source and target mean embeddings (1 x D each). Central
          moments of a single vector vanish, so CMD reduces to the L2 distance
          between the means and CORAL is 0. The paper's figures and alpha use
          this variant.
``all``   between equally sized sets; the larger domain is subsampled
          without replacement using ``--seed``.

Output JSON: ``{domain: {"CMD": {"mean": .., "all": ..}, "Coral": {...}}}``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from speech_hallucinations.divergences import cmd, coral
from speech_hallucinations.embeddings import domain_key, parse_embeddings

logger = logging.getLogger(__name__)

DIVERGENCES = {"CMD": cmd, "Coral": coral}


def match_sizes(
    source: np.ndarray, target: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    size = min(len(source), len(target))
    if len(source) > size:
        source = source[rng.choice(len(source), size=size, replace=False)]
    if len(target) > size:
        target = target[rng.choice(len(target), size=size, replace=False)]
    return source, target


def domain_shift(source: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> dict:
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_sampled, target_sampled = match_sizes(source, target, rng)
    return {
        name: {
            "mean": divergence(source_mean, target_mean),
            "all": divergence(source_sampled, target_sampled),
        }
        for name, divergence in DIVERGENCES.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--embeddings-dir", type=Path, default=Path("runs/embeddings"))
    parser.add_argument(
        "--source", default="librispeech_asr_test", help="Domain key of the source dataset"
    )
    parser.add_argument(
        "--embedding", choices=["speech", "text"], default="speech", help="SONAR modality"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Subsampling seed for the 'all' variant (the original run was unseeded)",
    )
    parser.add_argument("--output", type=Path, help="Defaults to runs/shifts/<embedding>.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    column = f"{args.embedding}_embedding"
    files = {domain_key(path): path for path in sorted(args.embeddings_dir.glob("*.csv"))}
    if args.source not in files:
        raise FileNotFoundError(
            f"No embedding file for source {args.source!r} in {args.embeddings_dir}"
        )
    source = parse_embeddings(pd.read_csv(files[args.source])[column])

    rng = np.random.default_rng(args.seed)
    results = {}
    for domain, path in files.items():
        target = parse_embeddings(pd.read_csv(path)[column])
        results[domain] = domain_shift(source, target, rng)
        logger.info("%s: CMD(mean)=%.4f", domain, results[domain]["CMD"]["mean"])

    output = args.output or Path("runs/shifts") / f"{args.embedding}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=4) + "\n")


if __name__ == "__main__":
    main()
