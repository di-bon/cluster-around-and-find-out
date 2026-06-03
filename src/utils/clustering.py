import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import normalize

from data.document_registry import DocumentRegistry
from utils.reduce_dimensions import reduce_dimensions


def run_clustering(embeddings: np.ndarray, config: dict) -> np.ndarray:
    algo = config["algorithm"]

    # Pick the right reduction strategy per algorithm
    if algo == "hdbscan":
        reduced = reduce_dimensions(embeddings, method="umap", n_components=50)
    elif algo in ("kmeans", "agglomerative"):
        reduced = reduce_dimensions(embeddings, method="pca", n_components=100)
    else:
        reduced = embeddings

    normed = normalize(reduced, norm="l2")
    params = config["params"]

    if algo == "kmeans":
        model = KMeans(**params)
        labels = model.fit_predict(normed)
    elif algo == "hdbscan":
        # After UMAP, euclidean works well and is faster than cosine
        params = {**params, "metric": "euclidean"}
        model = hdbscan.HDBSCAN(**params)
        labels = model.fit_predict(normed)
    elif algo == "agglomerative":
        model = AgglomerativeClustering(**params)
        labels = model.fit_predict(normed)
    else:
        raise ValueError(f"Unknown algorithm: '{algo}'.")

    n_clusters_found = len(set(labels) - {-1})
    print(f"\n✅ Clustering complete. Clusters found: {n_clusters_found}")
    if -1 in labels:
        print(f"   ⚠️  HDBSCAN marked {np.sum(labels == -1)} document(s) as noise.")

    return labels


def build_cluster_index(
    documents: list[str], labels: np.ndarray
) -> dict[int, list[str]]:
    """
    Returns a dict mapping cluster_id → list of documents in that cluster.
    Noise points (label == -1) are grouped under the key -1.
    """
    index: dict[int, list[str]] = {}
    for doc, label in zip(documents, labels):
        index.setdefault(int(label), []).append(doc)
    return index


def merge_clusters(
    registry: DocumentRegistry, cluster_ids: list[int]
) -> DocumentRegistry:
    """Reassigns all records in cluster_ids to the lowest id among them."""
    target = min(cluster_ids)
    for record in registry._records:
        if record.cluster in cluster_ids:
            record.cluster = target
    print(f"✅ Merged clusters {cluster_ids} → cluster {target}.")
    return registry

