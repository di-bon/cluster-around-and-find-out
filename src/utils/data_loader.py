import urllib.request
import os
import random

def download_dataset(dataset_dir: str, url_path_pairs: list[tuple[str, str]]) -> None:
    os.makedirs(dataset_dir, exist_ok=True)
    for url, path in url_path_pairs:
        if not os.path.exists(path):
            print(f"⬇️  Downloading {os.path.basename(path)}…")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"✅ Already cached: {os.path.basename(path)}")


def load_dataset(titles_path: str, sample_size: int | None = None, seed: int = 42) -> list[str]:
    with open(titles_path, encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]

    if sample_size is not None:
        rng = random.Random(seed)
        titles = rng.sample(titles, k=sample_size)

    print(f"📚 Loaded {len(titles)} documents.")
    return titles

# ── Step 1: Build a dataset summary for the interview agent ──────────────────

def build_dataset_summary(
    documents: list[str],
    n_examples: int = 10,
    seed: int = 42,
) -> str:
    rng = random.Random(seed)
    examples = rng.sample(documents, k=min(n_examples, len(documents)))
    example_lines = "\n".join(f'  - "{doc}"' for doc in examples)

    return (
        f"Dataset: StackOverflow question titles\n"
        f"Total documents: {len(documents)}\n\n"
        f"Random sample of documents:\n{example_lines}"
    )