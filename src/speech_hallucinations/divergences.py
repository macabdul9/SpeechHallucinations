"""Domain divergences between two sets of embeddings with shape (N, D).

NumPy ports of the PyTorch implementations used in the experiments; see
THIRD_PARTY_NOTICES.md for their origin.
"""

from __future__ import annotations

import numpy as np


def cmd(source: np.ndarray, target: np.ndarray, n_moments: int = 5) -> float:
    """Central Moment Discrepancy (Zellinger et al., 2017), as implemented in the experiments.

    Sums the L2 distance between means and between central moments 2..n_moments.
    Unlike the original paper's definition there is no 1/|b-a|^k range scaling,
    and the distances are not squared.
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centered_source, centered_target = source - source_mean, target - target_mean
    total = np.linalg.norm(source_mean - target_mean)
    for order in range(2, n_moments + 1):
        total += np.linalg.norm(
            (centered_source**order).mean(axis=0) - (centered_target**order).mean(axis=0)
        )
    return float(total)


def coral(source: np.ndarray, target: np.ndarray) -> float:
    """CORAL loss: mean squared difference of (unnormalized) covariances over 4 D^2."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    dim = source.shape[1]
    centered_source = source.mean(axis=0, keepdims=True) - source
    centered_target = target.mean(axis=0, keepdims=True) - target
    difference = centered_source.T @ centered_source - centered_target.T @ centered_target
    return float(np.mean(difference**2) / (4 * dim * dim))
