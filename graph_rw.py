"""
Graph-based Random Walk and Scoring Module.

Provides functions for:
  - Computing hyper-box centres.
  - Building a kNN weighted graph from a list of hyper-boxes.
  - Generating first-order and second-order biased random walks.
  - Computing co-occurrence matrices, structural similarity, and
    hyper-box anomaly memberships.
  - Estimating sigma_R via median distance and computing Gaussian
    membership of samples to hyper-boxes.
  - Propagating anomaly scores from hyper-boxes to samples via
    fuzzy rough lower/upper approximations.
"""

import numpy as np
from typing import List, Optional
from numba import njit

from hyperbox import HyperBox, EPS


# ---------- 1. Centre computation ----------
def compute_box_centers(H_list: List[HyperBox]) -> np.ndarray:
    """Return the centre coordinates of each hyper-box."""
    M = len(H_list)
    if M == 0:
        return np.zeros((0, 0), dtype=float)
    d = H_list[0].lower.shape[0]
    centers = np.zeros((M, d), dtype=float)
    for i, box in enumerate(H_list):
        centers[i] = 0.5 * (box.lower + box.upper)
    return centers


# ---------- 2. kNN graph construction (vectorised RBF weights) ----------
def build_knn_graph(
        H_list: List[HyperBox],
        k: int = 10,
        sigma_dist: float = 1.0,
) -> np.ndarray:
    """Build a kNN weighted graph on hyper-box centres using Gaussian kernel weights."""
    M = len(H_list)
    if M == 0:
        return np.zeros((0, 0), dtype=float)
    centers = compute_box_centers(H_list)
    G = np.sum(centers ** 2, axis=1, keepdims=True)
    D2 = G + G.T - 2.0 * (centers @ centers.T)
    D2[D2 < 0] = 0.0
    D = np.sqrt(D2)

    W = np.zeros((M, M), dtype=float)
    if M == 1:
        return W
    actual_k = min(k, M - 1)
    # binary kNN adjacency
    for i in range(M):
        dist_i = D[i].copy()
        dist_i[i] = np.inf
        nn_idx = np.argpartition(dist_i, actual_k)[:actual_k]
        W[i, nn_idx] = 1.0
        W[nn_idx, i] = 1.0

    # vectorised RBF weights
    mask = W > 0
    W[mask] = np.exp(-D[mask] ** 2 / (2.0 * sigma_dist ** 2 + EPS))
    np.fill_diagonal(W, 0.0)
    return W


# ---------- 3. Second-order biased random walk ----------
def generate_second_order_random_walks(
        W: np.ndarray,
        num_walks_per_node: Optional[int] = None,
        p: float = 2,
        q: float = 0.5,
        min_walk_length: int = 10,
        max_walk_length: int = 80,
        random_state: Optional[int] = None,
) -> List[List[int]]:
    """Generate second-order biased random walks (Node2Vec style) on the weighted graph W."""
    if W.shape[0] == 0:
        return []
    if random_state is None:
        random_state = np.random.randint(0, 2 ** 31)
    rng = np.random.RandomState(random_state)
    seed = rng.randint(0, 2 ** 31)

    M = W.shape[0]
    if num_walks_per_node is None:
        num_walks_per_node = 10 if M <= 2000 else 3
    L = int(np.floor(0.3 * M))
    L = max(min_walk_length, L)
    L = min(max_walk_length, L)

    walks_np = _second_order_walks_numba(W, num_walks_per_node, p, q, L, seed)
    return [list(walk) for walk in walks_np]


@njit(nogil=True)
def _second_order_walks_numba(W, n_walks, p, q, L, seed):
    """Numba kernel for second-order random walks."""
    np.random.seed(seed)
    M = W.shape[0]
    total = M * n_walks
    walks = np.zeros((total, L), dtype=np.int64)

    # pre-store neighbours and weights to avoid advanced indexing issues
    max_deg = 0
    for i in range(M):
        deg = 0
        for j in range(M):
            if W[i, j] > 0:
                deg += 1
        if deg > max_deg:
            max_deg = deg

    neigh_list = np.full((M, max_deg), -1, dtype=np.int64)
    weight_list = np.zeros((M, max_deg), dtype=np.float64)
    for i in range(M):
        deg = 0
        for j in range(M):
            if W[i, j] > 0:
                if deg < max_deg:
                    neigh_list[i, deg] = j
                    weight_list[i, deg] = W[i, j]
                    deg += 1

    idx = 0
    for start in range(M):
        for _ in range(n_walks):
            walk = walks[idx]
            walk[0] = start
            curr = start
            deg0 = 0
            while deg0 < max_deg and neigh_list[curr, deg0] >= 0:
                deg0 += 1
            if deg0 == 0:
                for step in range(1, L):
                    walk[step] = curr
                idx += 1
                continue

            # first step: weighted first-order choice
            nbrs = neigh_list[curr, :deg0]
            wts = weight_list[curr, :deg0]
            s0 = wts.sum()
            probs = wts / s0 if s0 > 0 else np.ones(deg0) / deg0
            nxt = _choice_numba(nbrs, probs)
            walk[1] = nxt
            prev = curr
            curr = nxt

            # subsequent steps: second-order bias
            for step in range(2, L):
                deg = 0
                while deg < max_deg and neigh_list[curr, deg] >= 0:
                    deg += 1
                if deg == 0:
                    walk[step] = curr
                    prev = curr
                    continue
                nbrs_curr = neigh_list[curr, :deg]
                wts_curr = weight_list[curr, :deg]
                unnorm = np.zeros(deg)
                for k in range(deg):
                    dst = nbrs_curr[k]
                    if dst == prev:
                        bias = 1.0 / p
                    else:
                        is_neighbor = False
                        deg_prev = 0
                        while deg_prev < max_deg and neigh_list[prev, deg_prev] >= 0:
                            if neigh_list[prev, deg_prev] == dst:
                                is_neighbor = True
                                break
                            deg_prev += 1
                        bias = 1.0 if is_neighbor else 1.0 / q
                    unnorm[k] = bias * wts_curr[k]
                s = unnorm.sum()
                probs = unnorm / s if s > 0 else np.ones(deg) / deg
                nxt = _choice_numba(nbrs_curr, probs)
                walk[step] = nxt
                prev = curr
                curr = nxt
            idx += 1
    return walks


@njit(nogil=True)
def _choice_numba(options, probs):
    """Weighted random choice (Numba-compatible)."""
    cs = probs.cumsum()
    r = np.random.random()
    for i in range(len(cs)):
        if r < cs[i]:
            return options[i]
    return options[-1]


# ---------- 4. Co-occurrence matrix  ----------
def compute_cooccurrence_matrix(
        walks: List[List[int]],
        num_nodes: int,
        window_size: int = 5,
) -> np.ndarray:
    """Build co-occurrence matrix from a list of walk sequences."""
    if num_nodes == 0 or len(walks) == 0:
        return np.zeros((num_nodes, num_nodes), dtype=np.float32)
    walk_arr = np.array(walks, dtype=np.int64)
    C = _cooccurrence_numba(walk_arr, num_nodes, window_size)
    return C.astype(np.float32)


@njit(nogil=True)
def _cooccurrence_numba(walks_arr, num_nodes, window_size):
    """Numba kernel for co-occurrence matrix computation."""
    n_walks, L = walks_arr.shape
    C = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for w in range(n_walks):
        walk = walks_arr[w]
        for pos in range(L):
            u = walk[pos]
            for offset in range(1, window_size + 1):
                if pos - offset >= 0:
                    v = walk[pos - offset]
                    C[u, v] += 1.0
                if pos + offset < L:
                    v = walk[pos + offset]
                    C[u, v] += 1.0
    return C


# ---------- 5. Structural similarity ----------
def compute_structural_similarity_cosine_exact(C: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix from co-occurrence profiles."""
    C = np.asarray(C, dtype=np.float32)
    M = C.shape[0]
    if M == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    norms[norms < EPS] = 1.0
    C_norm = C / norms
    S = C_norm @ C_norm.T
    np.clip(S, 0.0, 1.0, out=S)
    return S


# ---------- 6. Structural anomaly membership ----------
def compute_structural_anomaly_membership(
        S: np.ndarray,
        tau: float = 2.0,
) -> np.ndarray:
    """Convert structural similarity matrix to anomaly membership via sigmoid normalisation."""
    S = np.asarray(S, dtype=float)
    M = S.shape[0]
    if M == 0:
        return np.zeros(0, dtype=float)
    avg_sim = S.mean(axis=1)
    raw = 1.0 - avg_sim
    m = np.median(raw)
    mad = np.median(np.abs(raw - m))
    z = (raw - m) / (tau * mad + EPS)
    return 1.0 / (1.0 + np.exp(-z))


# ---------- 7. Size prior ----------
def compute_box_size_prior(H_list: List[HyperBox]) -> np.ndarray:
    """Min-max normalised density penalty: smaller boxes receive higher prior weight."""
    M = len(H_list)
    if M == 0:
        return np.zeros(0, dtype=float)
    sizes = np.array([len(box.indices) for box in H_list], dtype=float)
    min_size = sizes.min()
    max_size = sizes.max()
    if max_size - min_size < EPS:
        return np.zeros(M, dtype=float)
    normalized = (sizes - min_size) / (max_size - min_size)
    return 1.0 - normalized


# ---------- 8. Hyper-box anomaly membership ----------
def compute_box_anomaly_membership(
        H_list: List[HyperBox],
        S: np.ndarray,
        tau: float = 2.0,
) -> np.ndarray:
    """Combine structural anomaly membership with size prior."""
    A_struct = compute_structural_anomaly_membership(S, tau=tau)
    A_prior = compute_box_size_prior(H_list)
    return A_struct * A_prior


# ---------- 9. sigma_R estimation ----------
def estimate_sigma_R_median(
        X: np.ndarray,
        H_list: List[HyperBox],
        sample_points: int = 500,
        sample_boxes: Optional[int] = None,
) -> float:
    """Estimate sigma_R as the median distance from samples to hyper-box centres."""
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    M = len(H_list)
    if N == 0 or M == 0:
        return 1.0
    m_points = min(sample_points, N)
    idx_points = np.random.choice(N, size=m_points, replace=False)
    if sample_boxes is None:
        m_boxes = M
        idx_boxes = np.arange(M)
    else:
        m_boxes = min(sample_boxes, M)
        idx_boxes = np.random.choice(M, size=m_boxes, replace=False)
    X_sub = X[idx_points]
    centers = np.zeros((m_boxes, d), dtype=float)
    for j, box_idx in enumerate(idx_boxes):
        box = H_list[box_idx]
        centers[j] = 0.5 * (box.lower + box.upper)
    Gx = np.sum(X_sub ** 2, axis=1, keepdims=True)
    Gc = np.sum(centers ** 2, axis=1, keepdims=True).T
    D2 = Gx + Gc - 2.0 * (X_sub @ centers.T)
    D2[D2 < 0] = 0.0
    D = np.sqrt(D2)
    all_d = D.ravel()
    all_d = all_d[all_d > 0]
    if all_d.size == 0:
        return 1.0
    sigma_R = float(np.median(all_d))
    return sigma_R if sigma_R > EPS else 1.0


# ---------- 10. Gaussian membership ----------
def gaussian_membership_X_H(
        X: np.ndarray,
        H_list: List[HyperBox],
        sigma_R: Optional[float] = None,
        sample_points: int = 500,
        sample_boxes: Optional[int] = None,
        batch_size: int = 512,
) -> np.ndarray:
    """Compute Gaussian membership matrix of samples to hyper-boxes."""
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    M = len(H_list)
    if M == 0:
        return np.zeros((N, 0), dtype=float)
    if sigma_R is None or sigma_R <= 0:
        sigma_R = estimate_sigma_R_median(X, H_list, sample_points, sample_boxes)
    centers = np.zeros((M, d), dtype=float)
    for i, box in enumerate(H_list):
        centers[i] = 0.5 * (box.lower + box.upper)
    R = np.zeros((N, M), dtype=float)
    start = 0
    while start < N:
        end = min(start + batch_size, N)
        X_batch = X[start:end]
        Gx = np.sum(X_batch ** 2, axis=1, keepdims=True)
        Gc = np.sum(centers ** 2, axis=1, keepdims=True).T
        D2_batch = Gx + Gc - 2.0 * (X_batch @ centers.T)
        D2_batch[D2_batch < 0] = 0.0
        R_batch = np.exp(-D2_batch / (2.0 * sigma_R ** 2 + EPS))
        R[start:end] = R_batch
        start = end
    return R


# ---------- 11. Fuzzy rough approximations ----------
def compute_fuzzy_rough_approximation(
        R: np.ndarray,
        A_box: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray):
    """Compute fuzzy lower/upper approximations and approximation consistency."""
    R = np.asarray(R, dtype=float)
    A = np.asarray(A_box, dtype=float)
    N, M = R.shape
    if M == 0:
        return (np.zeros(N), np.zeros(N), np.zeros(N))
    A_row = A.reshape(1, M)
    L = np.maximum(1.0 - R, A_row).min(axis=1)
    U = np.minimum(R, A_row).max(axis=1)
    AAA = 1.0 - np.abs(U - L)
    AAA = np.clip(AAA, 0.0, 1.0)
    return L, U, AAA


# ---------- 12. Fuzzy rough anomaly score ----------
def compute_fuzzy_rough_anomaly(
        R: np.ndarray,
        A_box: np.ndarray,
        use_enhanced: bool = True,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Compute the final sample-level anomaly score via fuzzy rough propagation."""
    L, U, AAA = compute_fuzzy_rough_approximation(R, A_box)
    DOF_FR = 0.5 * (L + U)
    if use_enhanced:
        DOF_FR = AAA * DOF_FR
    return L, U, AAA, DOF_FR
