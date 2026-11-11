"""t-SNE of SONAR speech and text embeddings across datasets (Figure 1).

Writes ``tsne_data.csv`` and one PDF per modality to the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from speech_hallucinations.data import REPO_ROOT
from speech_hallucinations.embeddings import domain_key, parse_embeddings

MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "X", "d"]


def plot_projection(projection: np.ndarray, labels: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = plt.cm.tab20.colors
    for index, label in enumerate(dict.fromkeys(labels)):
        mask = labels == label
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=8,
            color=colors[index % len(colors)],
            marker=MARKERS[index % len(MARKERS)],
            label=label,
            zorder=3,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        frameon=False,
        fontsize=9,
        markerscale=1.5,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embeddings-dir", type=Path, default=Path("runs/embeddings"))
    parser.add_argument(
        "--display-names", type=Path, default=REPO_ROOT / "configs" / "dataset_display_names.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tsne"))
    args = parser.parse_args()

    display_names = json.loads(args.display_names.read_text())
    vectors = {"speech": [], "text": []}
    labels = []
    for path in sorted(args.embeddings_dir.glob("*.csv")):
        frame = pd.read_csv(path).dropna(subset=["speech_embedding", "text_embedding"])
        for modality in vectors:
            vectors[modality].append(parse_embeddings(frame[f"{modality}_embedding"]))
        if len(vectors["speech"][-1]) != len(vectors["text"][-1]):
            raise ValueError(f"{path}: speech and text embedding counts differ")
        name = display_names.get(domain_key(path), domain_key(path))
        labels.extend([name] * len(vectors["speech"][-1]))

    table = {}
    for modality, arrays in vectors.items():
        projection = TSNE(n_components=2, random_state=args.seed).fit_transform(np.vstack(arrays))
        table[f"{modality}_c0"], table[f"{modality}_c1"] = projection[:, 0], projection[:, 1]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsne_data = pd.DataFrame(table).assign(dataset_labels=labels)
    tsne_data.to_csv(args.output_dir / "tsne_data.csv", index=False)
    for modality in ("speech", "text"):
        projection = tsne_data[[f"{modality}_c0", f"{modality}_c1"]].to_numpy()
        plot_projection(
            projection, tsne_data["dataset_labels"].to_numpy(), args.output_dir / f"{modality}.pdf"
        )


if __name__ == "__main__":
    main()
