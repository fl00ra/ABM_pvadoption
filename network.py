import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def build_multiple_networks(
    agent_array,
    intra_ratios=(0.1, 0.3, 0.5, 0.7, 0.85, 0.95),
    max_degree_per_node: int = 25,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
):
    """
    Build multiple homophilic networks for a given agent population.

    Nodes are grouped by dwelling type and WOZ high/low, then
    connected with a target intra-group link fraction.
    """
    n = agent_array.n_agents
    features = agent_array.get_feature_matrix()
    similarity = cosine_similarity(features)

    if rng is None:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    np.fill_diagonal(similarity, 0.0)

    def woz_label(woz: float, threshold: float = 400000.0) -> str:
        return "high" if woz >= threshold else "low"

    woning_types = agent_array.woning_type
    woz_values = agent_array.woz_value
    group_labels = [
        f"{woning}_{woz_label(woz)}"
        for woning, woz in zip(woning_types, woz_values)
    ]
    unique_groups = sorted(set(group_labels))
    group_labels_arr = np.array(group_labels)

    combined_groups = {
        g: np.where(group_labels_arr == g)[0].tolist()
        for g in unique_groups
    }
    group_ids = np.array([unique_groups.index(lbl) for lbl in group_labels])

    def total_deg() -> int:
        """Sample a total degree from a capped log-normal distribution."""
        deg = int(rng.lognormal(mean=np.log(8), sigma=0.4))
        return max(3, min(deg, max_degree_per_node))

    def select_top_k(sim_row: np.ndarray, candidates: list[int], k: int) -> list[int]:
        """Select up to k candidates with highest similarity."""
        if not candidates or k == 0:
            return []
        sim_scores = sim_row[candidates]
        if len(sim_scores) > k:
            top_k_idx = np.argpartition(-sim_scores, k - 1)[:k]
        else:
            top_k_idx = np.arange(len(sim_scores))
        return [candidates[idx] for idx in top_k_idx]

    def compute_network_homophily(adjacency_matrix: csr_matrix, groups: np.ndarray) -> float:
        """Fraction of edges connecting same-group nodes."""
        A = adjacency_matrix.tocoo()
        same_group_links = 0
        total_links = A.nnz
        for i, j in zip(A.row, A.col):
            if groups[i] == groups[j]:
                same_group_links += 1
        return same_group_links / total_links if total_links > 0 else 0.0

    networks: dict[float, csr_matrix] = {}
    meta_info: dict[float, dict] = {}

    for intra_ratio in intra_ratios:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        degree_count = np.zeros(n, dtype=int)

        for i in range(n):
            group = group_labels[i]
            same_group = [j for j in combined_groups[group] if j != i]
            other_group = [
                j
                for g in combined_groups
                if g != group
                for j in combined_groups[g]
            ]

            k_total = total_deg()
            k_in = int(k_total * intra_ratio)
            k_out = k_total - k_in

            selected_in = select_top_k(similarity[i], same_group, k_in)
            if other_group and k_out > 0:
                selected_out = rng.choice(
                    other_group,
                    size=min(k_out, len(other_group)),
                    replace=False,
                ).tolist()
            else:
                selected_out = []

            for j in selected_in + selected_out:
                if degree_count[i] < max_degree_per_node and degree_count[j] < max_degree_per_node:
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)
                    degree_count[i] += 1
                    degree_count[j] += 1

        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        A = A.maximum(A.transpose()).tolil()
        A.setdiag(0.0)
        A = A.tocsr()
        A.eliminate_zeros()

        adjacency_matrix = A
        degrees = adjacency_matrix.sum(axis=1).A1
        empirical_homophily = compute_network_homophily(
            adjacency_matrix,
            group_labels_arr,
        )

        meta_info[float(intra_ratio)] = {
            "avg_degree": float(degrees.mean()) if degrees.size > 0 else 0.0,
            "min_degree": int(degrees.min()) if degrees.size > 0 else 0,
            "max_degree": int(degrees.max()) if degrees.size > 0 else 0,
            "empirical_homophily": float(empirical_homophily),
        }

        if verbose and degrees.size > 0:
            print(
                f"[Homophily {intra_ratio:.2f}] "
                f"Avg deg: {degrees.mean():.1f}, Homophily: {empirical_homophily:.3f}"
            )

        networks[float(intra_ratio)] = adjacency_matrix

    return networks, meta_info, group_ids


def compute_social_weights(adjacency_matrix: csr_matrix, verbose: bool = False) -> np.ndarray:
    """
    Compute social influence weights from network structure using PageRank.
    """
    G = nx.from_scipy_sparse_array(adjacency_matrix)

    try:
        pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        if verbose:
            print("PageRank did not converge in 100 iterations, retrying with relaxed settings.")
        pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-4)

    scores = np.array(
        [pagerank_scores[i] for i in range(adjacency_matrix.shape[0])],
        dtype=float,
    )
    min_score = float(scores.min())
    max_score = float(scores.max())

    if max_score > min_score:
        normalized = (scores - min_score) / (max_score - min_score)
    else:
        normalized = np.full_like(scores, 0.5)

    scaled_scores = 0.5 + 1.5 * normalized

    if verbose:
        print(
            f"PageRank weights: "
            f"Mean={scaled_scores.mean():.3f}, "
            f"Std={scaled_scores.std():.3f}, "
            f"Min={scaled_scores.min():.3f}, "
            f"Max={scaled_scores.max():.3f}"
        )

    return scaled_scores.astype(np.float32)
