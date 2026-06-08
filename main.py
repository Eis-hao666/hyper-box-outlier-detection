"""
Main entry point for HSRWOD anomaly detection.

Provides functions for:
  - Loading and normalising a dataset from a .txt file.
  - Running the full HSRWOD pipeline: attribute distillation, hyper-box
    granulation, kNN graph construction, second-order biased random walks,
    structural scoring, and fuzzy rough propagation.
  - Grid searching over rho and gamma for hyper-parameter tuning.
  - Evaluating AUC and AP using the custom evaluation module.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from hyperbox import hyperbox_granulation, hyperbox_feature_distillation_image_algo_fast
from graph_rw import (
    build_knn_graph,
    generate_second_order_random_walks,
    compute_cooccurrence_matrix,
    compute_structural_similarity_cosine_exact,
    compute_box_anomaly_membership,
    gaussian_membership_X_H,
    compute_fuzzy_rough_anomaly,
)


def run_anomaly_detection_with_hyperboxes(
        data_path: str = "musk.txt",
        rho: float = 0.5,
        gamma: float = 1.0,
        random_state: int = 1,
        p: float = 2.0,
        q: float = 0.5,
        gamma_q: float = 0.9,  # quantile
        return_scores: bool = False,
):
    """
    Run the full HSRWOD anomaly detection pipeline on a single dataset.

    Parameters
    ----------
    data_path : str
        Path to the .txt dataset file (space-separated, last column is label).
    rho : float
        Attribute retention ratio for feature distillation.
    gamma : float
        Hyper-box scale factor gamma (granularity control).
    random_state : int
        Random seed for reproducibility.
    p : float
        Second-order walk return parameter.
    q : float
        Second-order walk in-out parameter.
    gamma_q : float
        Quantile used for the adaptive threshold delta.
    return_scores : bool
        If True, also return the raw outlier scores.

    Returns
    -------
    auc : float
        Area Under the ROC Curve.
    ap : float
        Average Precision.
    DOF_FR : np.ndarray (only if return_scores=True)
        Final sample-level outlier scores.
    """
    np.random.seed(random_state)

    # 1. Load data
    try:
        df = pd.read_csv(data_path, sep=r'\s+', header=None, skiprows=1)
        X_raw = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
    except Exception:
        print(f"Warning: {data_path} not found. Using synthetic data.")
        X_raw = np.random.rand(500, 20)
        y_raw = np.zeros(500, dtype=int)
        y_raw[:50] = 1

    values, counts = np.unique(y_raw, return_counts=True)
    normal_label = values[np.argmax(counts)]
    y_binary = (y_raw != normal_label).astype(int)

    # 2. Min-Max normalisation
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Attribute distillation
    X, selected_idx = hyperbox_feature_distillation_image_algo_fast(
        X_scaled, rho=rho,
        random_state=random_state,
        gamma_box=gamma, q_thresh=gamma_q,
    )
    # 4. Hyper-box granulation
    H_list = hyperbox_granulation(
        X, gamma_box=gamma, random_state=random_state, q=gamma_q,
    )
    if len(H_list) < 2:
        if return_scores:
            return 0.0, 0.0, np.zeros(X.shape[0])
        return 0.0, 0.0
    # 5. Graph construction and second-order random walks
    W = build_knn_graph(H_list, k=10)
    walks = generate_second_order_random_walks(
        W, p=p, q=q, random_state=random_state, max_walk_length=80,
    )
    C = compute_cooccurrence_matrix(walks, num_nodes=len(H_list))
    S = compute_structural_similarity_cosine_exact(C)

    # 6. Hyper-box anomaly membership and Gaussian membership
    A_box = compute_box_anomaly_membership(H_list, S)
    R = gaussian_membership_X_H(X, H_list)

    # 7. Fuzzy rough anomaly scoring
    _, _, _, DOF_FR = compute_fuzzy_rough_anomaly(R, A_box)

    return DOF_FR


def main():
    # ========== Configuration ==========
    DATASET_PATH = "Dataset/arrhythmia_variant1.txt"
    RANDOM_STATE = 2
    r = 0.2
    b = 0.3

    scores = run_anomaly_detection_with_hyperboxes(
        data_path=DATASET_PATH,
        rho=r,
        gamma=b,
        random_state=RANDOM_STATE,
        p=0.5,
        q=2,
        gamma_q=0.9,
    )
    print(scores)


if __name__ == "__main__":
    main()
