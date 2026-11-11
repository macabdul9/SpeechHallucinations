"""Figures from result tables.

shift         WERD and HERD against CMD for each model (Figure 2)
model-size    average WERD and HERD across Whisper sizes (Figure 4)
perturbation  WER and HER per perturbation for each model (Figure 3)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from speech_hallucinations.data import REPO_ROOT
from speech_hallucinations.metrics import hallucination_error_rate

WER_COLOR = "#0072B2"
HER_COLOR = "#D55E00"
CONFIGS = REPO_ROOT / "configs"


def _style(*axes) -> None:
    for ax in axes:
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
    axes[0].yaxis.grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.6, zorder=0)


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _dual_axis_lines(x, werd, herd, werd_label: str, herd_label: str):
    fig, ax_left = plt.subplots(figsize=(5, 3.5))
    ax_right = ax_left.twinx()
    line_style = {"linewidth": 2.0, "markersize": 6, "markeredgecolor": "black", "zorder": 3}
    ax_left.plot(x, werd, marker="o", color=WER_COLOR, label=werd_label, **line_style)
    ax_right.plot(x, herd, marker="s", color=HER_COLOR, label=herd_label, **line_style)
    ax_left.set_ylabel(werd_label, fontsize=13, color=WER_COLOR)
    ax_right.set_ylabel(herd_label, fontsize=13, color=HER_COLOR)
    ax_left.tick_params(axis="y", labelcolor=WER_COLOR, labelsize=11)
    ax_right.tick_params(axis="y", labelcolor=HER_COLOR, labelsize=11)
    _style(ax_left, ax_right)
    return fig, ax_left


def plot_shift(args: argparse.Namespace) -> None:
    degradation = pd.read_csv(args.correlation_dir / "degradation.csv")
    alphas = pd.read_csv(args.correlation_dir / "alpha.csv").set_index("model")
    display_names = json.loads((CONFIGS / "dataset_display_names.json").read_text())
    models = args.models or sorted(degradation["model"].unique())
    for model in models:
        group = degradation[degradation["model"] == model].sort_values("shift")
        fig, ax = _dual_axis_lines(group["shift"], group["WERD"], group["HERD"], "WERD", "HERD")
        for shift, werd, dataset in zip(
            group["shift"], group["WERD"], group["dataset"], strict=True
        ):
            ax.annotate(
                display_names.get(dataset, dataset),
                (shift, werd),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax.set_xlabel("CMD", fontsize=12)
        alpha = alphas.loc[model]
        ax.set_title(
            f"{model} (α WERD={alpha['alpha_WERD']:.2f}, α HERD={alpha['alpha_HERD']:.2f})",
            fontsize=11,
            pad=28,
        )
        _save(fig, args.output_dir / f"shift_{model}.pdf")


def plot_model_size(args: argparse.Namespace) -> None:
    degradation = pd.read_csv(args.correlation_dir / "degradation.csv")
    sizes = json.loads(args.model_sizes.read_text())
    averages = degradation.groupby("model")[["WERD", "HERD"]].mean()
    models = sorted((m for m in sizes if m in averages.index), key=lambda m: (sizes[m], m))
    x = np.arange(len(models))
    fig, ax = _dual_axis_lines(
        x,
        averages.loc[models, "WERD"],
        averages.loc[models, "HERD"],
        "Average WERD",
        "Average HERD",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m} ({sizes[m]}M)" for m in models], rotation=45, ha="right", fontsize=9)
    _save(fig, args.output_dir / "model_size.pdf")


def plot_perturbation(args: argparse.Namespace) -> None:
    for path in args.inputs:
        frame = pd.read_csv(path)
        grouped = frame.groupby("perturbation", sort=False).agg(
            WER=("WER", "mean"),
            HER=(args.judge_column, lambda s: hallucination_error_rate(s, args.granularity)),
        )
        x = np.arange(len(grouped))
        width = 0.35
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bar_style = {"width": width, "edgecolor": "black", "zorder": 3}
        ax.bar(
            x - width / 2, grouped["WER"], color=WER_COLOR, hatch="///", label="WER", **bar_style
        )
        ax.bar(
            x + width / 2, grouped["HER"], color=HER_COLOR, hatch="...", label="HER", **bar_style
        )
        ax.set_ylabel("Error rate (%)", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        _style(ax)
        _save(fig, args.output_dir / f"perturbation_{path.parent.name}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    shift = commands.add_parser("shift", help="Figure 2")
    shift.add_argument("--correlation-dir", type=Path, default=Path("runs/correlation"))
    shift.add_argument("--models", nargs="+")
    shift.set_defaults(handler=plot_shift)

    size = commands.add_parser("model-size", help="Figure 4")
    size.add_argument("--correlation-dir", type=Path, default=Path("runs/correlation"))
    size.add_argument("--model-sizes", type=Path, default=CONFIGS / "whisper_model_sizes.json")
    size.set_defaults(handler=plot_model_size)

    perturbation = commands.add_parser("perturbation", help="Figure 3")
    perturbation.add_argument("inputs", nargs="+", type=Path, help="perturbation_output.csv files")
    perturbation.add_argument("--judge-column", default="gpt-4o-mini_finegrained")
    perturbation.add_argument("--granularity", choices=["coarse", "fine"], default="fine")
    perturbation.set_defaults(handler=plot_perturbation)

    for subparser in (shift, size, perturbation):
        subparser.add_argument("--output-dir", type=Path, default=Path("runs/figures"))
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
