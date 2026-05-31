import secrets

import numpy as np
from loguru import logger
from sklearn.decomposition import PCA
from umap import UMAP


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",  # "umap" | "pca" | "none"
    n_components: int = 50,
) -> np.ndarray:
    """
    Reduces embedding dimensionality before clustering.
    Recommended for all algorithms, essential for HDBSCAN.
    """
    if method == "none":
        return embeddings

    n_samples, n_features = embeddings.shape

    # n_components must be strictly less than both axes
    safe_n = min(n_components, n_samples - 1, n_features - 1)

    if safe_n < n_components:
        print(
            f"   ⚠️  Requested {n_components} components but only {n_samples} samples — "
            f"reducing to {safe_n}."
        )

    if safe_n < 2:
        print(f"   ⚠️  Too few samples ({n_samples}) to reduce — skipping.")
        return embeddings

    print(f"📉 Reducing dimensions: {n_features}D → {safe_n}D via {method.upper()}...")

    random_state = secrets.randbits(32)
    logger.info(f"Using random_state: {random_state}")

    if method == "pca":
        reducer = PCA(n_components=safe_n, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        explained = reducer.explained_variance_ratio_.sum()
        print(f"   Explained variance retained: {explained:.1%}")

    elif method == "umap":
        n_neighbors = min(15, n_samples - 1)
        logger.info(f"In umap, using n_neighbors: {n_neighbors}")
        reducer = UMAP(
            n_components=safe_n,
            n_neighbors=n_neighbors,  # controls local vs global structure
            min_dist=0.0,  # 0.0 keeps points tightly packed — better for clustering
            metric="cosine",  # matches NV-Embed-v1's similarity space
            random_state=random_state,
        )
        reduced = reducer.fit_transform(embeddings)

    else:
        raise ValueError(
            f"Unknown method: '{method}'. Expected 'umap', 'pca', or 'none'."
        )

    print(f"   Done. Output shape: {reduced.shape}")
    return reduced

