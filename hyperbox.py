"""
Hyper-box Granulation and Feature Distillation Module.

Provides classes and functions for:
  - Representing data granules as axis-aligned hyper-boxes.
  - Computing point-to-box distances and box granularity.
  - Estimating an adaptive granularity threshold (delta) from nearest-neighbor distances.
  - Performing gamma-scale adaptive hyper-box granulation on a dataset.
  - Distilling a subset of features via hyper-box-based attribute clustering
    and diversity completion (Algorithm 2 in the accompanying paper).

All public functions preserve their original signatures and parameter defaults.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from scipy.spatial import cKDTree

EPS = 1e-8


@dataclass
class HyperBox:
    """An axis-aligned hyper-box defined by lower/upper bounds and the indices of contained samples."""
    lower: np.ndarray
    upper: np.ndarray
    indices: List[int]

    def copy(self) -> "HyperBox":
        return HyperBox(self.lower.copy(), self.upper.copy(), self.indices.copy())


def granularity(box: HyperBox) -> float:
    """Granularity G(H) = ||upper - lower||_2 (Euclidean length of the main diagonal)."""
    return float(np.linalg.norm(box.upper - box.lower))


def distances_to_box_batch(X_batch: np.ndarray,
                           lower: np.ndarray,
                           upper: np.ndarray) -> np.ndarray:
    """
    Euclidean distance from each point in X_batch to the hyper-box [lower, upper].
    dist(x, H) = || x - clip(x, lower, upper) ||_2.
    """
    proj = np.clip(X_batch, lower, upper)
    diff = X_batch - proj
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def estimate_delta_knn(
        X: np.ndarray,
        gamma_box: float = 0.5,
        sample_size: int = 200,
        q: float = 0.9,
) -> float:
    """
    Estimate adaptive granularity threshold delta = gamma * s.
    s is the q-th quantile of the nearest-neighbor distances of a random subsample of size sample_size.
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    if N <= 1:
        return 1.0

    m = min(sample_size, N)
    idx = np.random.choice(N, size=m, replace=False)
    X_sub = X[idx]

    tree = cKDTree(X_sub)
    dists, _ = tree.query(X_sub, k=2)
    nn_dist = dists[:, 1]

    s0 = np.quantile(nn_dist, q)
    delta = gamma_box * s0
    return float(max(delta, EPS))


def hyperbox_granulation(
        X: np.ndarray,
        gamma_box: float = 0.5,
        delta: Optional[float] = None,
        sample_size: int = 200,
        q: float = 0.9,
        random_state: Optional[int] = None,
) -> List[HyperBox]:
    """
    Gamma-scale adaptive hyper-box granulation (Algorithm 1).
    Builds a list of hyper-boxes by iteratively merging the closest sample
    until the box granularity exceeds the threshold delta.
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    rng = np.random.RandomState(random_state)

    if N == 0:
        return []

    if delta is None or delta <= 0:
        delta = estimate_delta_knn(X, gamma_box=gamma_box,
                                   sample_size=sample_size, q=q)

    mask = np.ones(N, dtype=bool)
    start_idx = rng.choice(N)
    mask[start_idx] = False

    H_list = []
    cur_lower = X[start_idx].copy()
    cur_upper = X[start_idx].copy()
    cur_indices = [int(start_idx)]

    while np.any(mask):
        proj = np.clip(X, cur_lower, cur_upper)
        diff = X - proj
        dists = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        dists[~mask] = np.inf

        best_idx = int(np.argmin(dists))
        x_m = X[best_idx]

        new_lower = np.minimum(cur_lower, x_m)
        new_upper = np.maximum(cur_upper, x_m)

        if np.linalg.norm(new_upper - new_lower) <= delta:
            cur_lower = new_lower
            cur_upper = new_upper
            cur_indices.append(best_idx)
        else:
            H_list.append(HyperBox(cur_lower.copy(),
                                   cur_upper.copy(),
                                   cur_indices.copy()))
            cur_lower = x_m.copy()
            cur_upper = x_m.copy()
            cur_indices = [best_idx]

        mask[best_idx] = False

    H_list.append(HyperBox(cur_lower.copy(),
                           cur_upper.copy(),
                           cur_indices.copy()))
    return H_list


def degenerate_hyperboxes_from_samples(X: np.ndarray) -> List[HyperBox]:
    """Create degenerate (point) hyper-boxes, one per sample (used in ablation)."""
    X = np.asarray(X, dtype=float)
    return [HyperBox(lower=xi.copy(), upper=xi.copy(), indices=[i])
            for i, xi in enumerate(X)]


def hyperbox_feature_distillation_image_algo_fast(
        X: np.ndarray,
        rho: float = 0.5,
        random_state: int = None,
        gamma_box: float = 0.5,
        q_thresh: float = 0.9,
) -> (np.ndarray, List[int]):
    """
    Hyper-box-driven attribute distillation (Algorithm 2).

    Steps:
      1. Min-Max normalize each attribute vector to [0,1].
      2. Transpose: treat attributes as samples.
      3. Perform hyper-box granulation in the attribute space.
      4. Sort boxes by cardinality and select the attribute closest to each box centre.
      5. If the target dimension is not reached, apply Max-Min diversity completion.
    Returns the data matrix restricted to the selected attributes and their indices.
    """
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    target_T = max(1, int(np.round(d * rho)))
    rng = np.random.RandomState(random_state)

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + EPS)

    X_calc = X_norm
    X_feats = X_calc.T  # shape (d, N)

    delta_auto = estimate_delta_knn(
        X_feats, gamma_box=gamma_box,
        sample_size=min(d, 200), q=q_thresh
    )
    feature_boxes = hyperbox_granulation(
        X_feats, delta=delta_auto, random_state=random_state,
        q=q_thresh
    )

    feature_boxes.sort(key=lambda box: len(box.indices), reverse=True)

    selected_indices = []
    for box in feature_boxes:
        if len(selected_indices) >= target_T:
            break
        center = (box.lower + box.upper) * 0.5
        feats_in_box = X_feats[np.array(box.indices)]
        diff = feats_in_box - center
        dists = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        best_global_idx = int(box.indices[np.argmin(dists)])
        selected_indices.append(best_global_idx)

    if len(selected_indices) < target_T:
        remaining = list(set(range(d)) - set(selected_indices))
        selected_feats = X_feats[selected_indices]
        while len(selected_indices) < target_T and remaining:
            min_dists = np.array([
                np.min(np.linalg.norm(selected_feats - X_feats[r], axis=1))
                for r in remaining
            ])
            best_local = np.argmax(min_dists)
            sel = remaining.pop(best_local)
            selected_indices.append(sel)
            selected_feats = X_feats[selected_indices]

    final_indices = np.array(sorted(selected_indices))
    return X[:, final_indices], final_indices
