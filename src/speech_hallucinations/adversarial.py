"""Build the adversarial evaluation set from RaphaelOlivier/librispeech_asr_adversarial.

Combines the ``adv_0.04``, ``adv_0.015``, and ``adv_0.015_RIR`` splits of the
``adv`` config, tags each row with ``split_source``, and sorts by ``id``. The
paper evaluates the published copy of this set, ``HaninZ/adversarial``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

SOURCE_DATASET = "RaphaelOlivier/librispeech_asr_adversarial"
SPLITS = ("adv_0.04", "adv_0.015", "adv_0.015_RIR")


def build_adversarial_set():
    from datasets import concatenate_datasets, load_dataset

    dataset = load_dataset(SOURCE_DATASET, "adv")
    parts = [
        dataset[split].add_column("split_source", [split] * len(dataset[split])) for split in SPLITS
    ]
    return concatenate_datasets(parts).sort("id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("data/adversarial"))
    args = parser.parse_args()

    combined = build_adversarial_set()
    combined.save_to_disk(str(args.output_dir))
    print(f"{len(combined)} rows saved to {args.output_dir}")
    for split, count in Counter(combined["split_source"]).items():
        print(f"{split}: {count}")


if __name__ == "__main__":
    main()
