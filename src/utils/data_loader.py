import os
import random
import urllib.request

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def download_dataset(dataset_dir: str, url_path_pairs: list[tuple[str, str]]) -> None:
    os.makedirs(dataset_dir, exist_ok=True)
    for url, path in url_path_pairs:
        if not os.path.exists(path):
            print(f"⬇️  Downloading {os.path.basename(path)}…")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"✅ Already cached: {os.path.basename(path)}")


def load_dataset(
    titles_path: str, sample_size: int | None = None, seed: int = 42
) -> list[str]:
    with open(titles_path, encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]

    if sample_size is not None:
        rng = random.Random(seed)
        titles = rng.sample(titles, k=sample_size)

    print(f"📚 Loaded {len(titles)} documents.")
    return titles


def build_dataset_summary(
    documents: list[str],
    n_clusters: int = 5,
    examples_per_cluster: int = 2,
    seed: int = 42,
) -> str:
    # Vectorize documents using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(documents)

    # Fit K-means
    n_clusters = min(n_clusters, len(documents))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(X)

    # Pick examples from each cluster
    rng = random.Random(seed)
    sampled = []
    for cluster_id in range(n_clusters):
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        k = min(examples_per_cluster, len(indices))
        chosen = rng.sample(indices, k=k)
        sampled.extend(documents[i] for i in chosen)

    # Trim/pad to exactly n_examples if needed
    if len(sampled) > examples_per_cluster * n_clusters:
        sampled = rng.sample(sampled, k=examples_per_cluster * n_clusters)

    example_lines = "\n".join(f'  - "{doc}"' for doc in sampled)

    return (
        f"Dataset: StackOverflow question titles\n"
        f"Total documents: {len(documents)}\n"
        f"Clusters: {n_clusters}\n\n"
        f"Representative sample of documents ({examples_per_cluster} per cluster):\n"
        f"{example_lines}"
    )

