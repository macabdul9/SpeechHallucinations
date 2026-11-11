"""Error-rate degradation under distribution shift and its correlation (alpha) with CMD.

For each model and target dataset:
  WERD = mean WER(target) - mean WER(source)
  HERD = HER(target) - HER(source), in percentage points
alpha is the correlation across target datasets between the shift and WERD or
HERD, computed per model. The paper calls alpha "the correlation coefficient";
Pearson is the default here (see README).

Writes ``degradation.csv`` (model, dataset, shift, WERD, HERD) and
``alpha.csv`` (model, alpha_WERD, alpha_HERD) to the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from speech_hallucinations.metrics import load_results, summarize

logger = logging.getLogger(__name__)

# Target datasets in Figure 2.
FIGURE_DOMAINS = ("LibriSpeech_test_noise", "GLOBE", "slue", "Primock57", "ami", "atcosim_corpus")


def correlation(x, y, method: str = "pearson") -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    statistic = pearsonr if method == "pearson" else spearmanr
    return float(statistic(x, y)[0])


def degradation_table(
    results: dict[tuple[str, str], pd.DataFrame],
    shifts: dict[str, float],
    source: str,
    domains: list[str],
    judge_model: str,
    granularity: str,
) -> pd.DataFrame:
    her_key = f"HER_{granularity}"
    rows = []
    for model in sorted({model for model, _ in results}):
        source_summary = summarize(results[(model, source)], judge_model)
        for domain in domains:
            if (model, domain) not in results:
                logger.warning("Missing %s/%s.csv; skipping", model, domain)
                continue
            target_summary = summarize(results[(model, domain)], judge_model)
            rows.append(
                {
                    "model": model,
                    "dataset": domain,
                    "shift": shifts[domain],
                    "WERD": target_summary["WER"] - source_summary["WER"],
                    "HERD": target_summary[her_key] - source_summary[her_key],
                }
            )
    return pd.DataFrame(rows)


def alpha_table(degradation: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for model, group in degradation.groupby("model"):
        rows.append(
            {
                "model": model,
                "alpha_WERD": correlation(group["shift"], group["WERD"], method),
                "alpha_HERD": correlation(group["shift"], group["HERD"], method),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=Path("runs/asr"))
    parser.add_argument("--shifts", type=Path, default=Path("runs/shifts/speech.json"))
    parser.add_argument("--divergence", default="CMD")
    parser.add_argument("--variant", choices=["mean", "all"], default="mean")
    parser.add_argument("--source", default="librispeech_asr_test")
    parser.add_argument("--domains", nargs="+", default=list(FIGURE_DOMAINS))
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--granularity", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/correlation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    shift_json = json.loads(args.shifts.read_text())
    shifts = {
        domain: values[args.divergence][args.variant] for domain, values in shift_json.items()
    }
    degradation = degradation_table(
        load_results(args.results_dir),
        shifts,
        args.source,
        args.domains,
        args.judge_model,
        args.granularity,
    )
    alphas = alpha_table(degradation, args.method)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    degradation.to_csv(args.output_dir / "degradation.csv", index=False)
    alphas.to_csv(args.output_dir / "alpha.csv", index=False)
    print(alphas.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
