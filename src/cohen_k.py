"""
Compute weighted Cohen's Kappa between your manual ratings and the LLM judge.

Input: a CSV file with one row per (item, metric) rating pair.

Required columns:
    item_id     - any identifier for the conversation/example
    metric      - "coherence", "alignment", or "separation"
    your_score  - your manual rating (integer)
    llm_score   - the LLM judge's rating (integer)

Example CSV:
    item_id,metric,your_score,llm_score
    0,coherence,2,2
    0,alignment,1,2
    0,separation,3,3
    1,coherence,2,3
    ...

Usage:
    python cohens_kappa.py ratings.csv
"""

import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

N_RESAMPLES = 10_000
RNG_SEED = 42
CI_LEVEL = 0.95
METRICS = ("coherence", "alignment", "separation")

# Landis & Koch benchmarks
BENCHMARKS = [
    (0.81, "Almost perfect"),
    (0.61, "Substantial"),
    (0.41, "Moderate"),
    (0.21, "Fair"),
    (0.00, "Slight"),
    (-1.0, "Poor / less than chance"),
]


def interpret(kappa: float) -> str:
    for threshold, label in BENCHMARKS:
        if kappa >= threshold:
            return label
    return "Poor"


# ── Weighted Cohen's Kappa ────────────────────────────────────────────────────


def weighted_kappa(
    rater_a: np.ndarray, rater_b: np.ndarray, weights: str = "quadratic"
) -> float:
    """
    Compute weighted Cohen's Kappa for two arrays of ordinal ratings.
    weights: "quadratic" (default, recommended for ordinal) or "linear"
    """
    rater_a = np.asarray(rater_a, dtype=int)
    rater_b = np.asarray(rater_b, dtype=int)
    n = len(rater_a)

    if n == 0:
        return np.nan

    categories = np.unique(np.concatenate([rater_a, rater_b]))
    k = len(categories)

    if k == 1:
        return 1.0  # perfect agreement, only one category used

    cat_index = {c: i for i, c in enumerate(categories)}
    max_dist = (k - 1) ** 2 if weights == "quadratic" else (k - 1)

    # Observed weight matrix
    observed = np.zeros((k, k), dtype=float)
    for a, b in zip(rater_a, rater_b):
        observed[cat_index[a], cat_index[b]] += 1

    # Weight matrix
    weight_matrix = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            dist = (i - j) ** 2 if weights == "quadratic" else abs(i - j)
            weight_matrix[i, j] = dist / max_dist

    # Expected weight matrix (marginal products)
    row_marginals = observed.sum(axis=1)
    col_marginals = observed.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals) / n

    po = 1 - (weight_matrix * observed).sum() / n
    pe = 1 - (weight_matrix * expected).sum() / n

    if pe == 0:
        return 1.0

    return (po - pe) / (1 - pe)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────


def bootstrap_kappa_ci(
    rater_a: np.ndarray, rater_b: np.ndarray, n: int = N_RESAMPLES, seed: int = RNG_SEED
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(rater_a))
    kappas = np.empty(n)
    for i in range(n):
        idx = rng.choice(indices, size=len(indices), replace=True)
        kappas[i] = weighted_kappa(rater_a[idx], rater_b[idx])

    alpha = 1 - CI_LEVEL
    return (
        float(np.percentile(kappas, 100 * alpha / 2)),
        float(np.percentile(kappas, 100 * (1 - alpha / 2))),
    )


# ── Loading ───────────────────────────────────────────────────────────────────


def load_ratings(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Returns a dict: metric -> (your_scores, llm_scores).
    Includes:
      - one key per individual metric ("coherence", "alignment", "separation")
      - "mean_score": Kappa on rounded mean per item — the primary reliability metric
      - "overall": all individual ratings pooled (for reference; NOT the same as mean_score)
    """
    rows = defaultdict(lambda: {"yours": [], "llm": []})
    # per-item accumulator for mean score: item_id -> {metric: (yours, llm)}
    items: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"item_id", "metric", "your_score", "llm_score"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        all_yours, all_llm = [], []
        for row in reader:
            metric = row["metric"].strip().lower()
            item_id = row["item_id"].strip()
            try:
                y = int(row["your_score"])
                l = int(row["llm_score"])
            except ValueError:
                print(f"  [WARNING] Skipping row with non-integer scores: {dict(row)}")
                continue
            rows[metric]["yours"].append(y)
            rows[metric]["llm"].append(l)
            all_yours.append(y)
            all_llm.append(l)
            items[item_id][metric] = (y, l)

    result = {}
    for metric, data in rows.items():
        result[metric] = (np.array(data["yours"]), np.array(data["llm"]))
    result["overall"] = (np.array(all_yours), np.array(all_llm))

    # Mean score: average the three metrics per item, keep as float (no rounding).
    # Weighted kappa handles continuous-ish values by treating each unique value
    # as a category, so rounding is not required and would lose information.
    mean_yours, mean_llm = [], []
    for item_id, metric_scores in items.items():
        present = [m for m in METRICS if m in metric_scores]
        if len(present) < len(METRICS):
            print(
                f"  [WARNING] Item '{item_id}' is missing metrics "
                f"{set(METRICS) - set(present)}, skipping from mean_score."
            )
            continue
        mean_yours.append(np.mean([metric_scores[m][0] for m in METRICS]))
        mean_llm.append(np.mean([metric_scores[m][1] for m in METRICS]))

    if mean_yours:
        result["mean_score"] = (np.array(mean_yours), np.array(mean_llm))

    return result


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_kappa_row(label: str, yours: np.ndarray, llm: np.ndarray) -> None:
    n = len(yours)
    if n < 2:
        print(f"  {label:<20} {n:>4}  {'n/a':>7}  (need at least 2 ratings)")
        return
    kappa = weighted_kappa(yours, llm)
    lo, hi = bootstrap_kappa_ci(yours, llm)
    label_interp = interpret(kappa)
    print(
        f"  {label:<20} {n:>4}  {kappa:>7.3f}  [{lo:>+7.3f}, {hi:>+7.3f}]    {label_interp}"
    )


def print_kappa_table(ratings: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    header = f"  {'Metric':<20} {'N':>4}  {'Kappa':>7}  {'95% CI':<22}  Interpretation"
    divider = "  " + "─" * (len(header) - 2)

    # ── Primary: mean score ───────────────────────────────────────────────────
    print("  PRIMARY — Mean Score  (coherence + alignment + separation) / 3")
    print(divider)
    if "mean_score" in ratings:
        print_kappa_row("mean_score", *ratings["mean_score"])
    else:
        print("  (not enough data to compute mean score)")
    print()

    # ── Secondary: individual metrics ─────────────────────────────────────────
    print("  SECONDARY — Individual Metrics")
    print(divider)
    for metric in [m for m in METRICS if m in ratings]:
        print_kappa_row(metric, *ratings[metric])
    print()

    # ── Reference: overall pooled ─────────────────────────────────────────────
    print("  REFERENCE — Overall pooled  (all individual ratings, NOT mean score)")
    print(divider)
    if "overall" in ratings:
        print_kappa_row("overall", *ratings["overall"])
    print()


def small_sample_warning(n_items: int) -> None:
    if n_items < 10:
        print(f"\n  ⚠  Only {n_items} item(s) rated. Kappa estimates will be unstable")
        print("     and CIs very wide. Treat results as a rough sanity check only.\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) != 2:
        print("Usage: python cohens_kappa.py <ratings.csv>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    ratings = load_ratings(path)

    # Infer number of unique items from any metric
    sample_key = next((m for m in METRICS if m in ratings), "overall")
    n_items = len(ratings[sample_key][0]) if sample_key in ratings else 0

    print(f"\n{'═' * 60}")
    print(f"  Inter-rater Agreement: You vs. LLM Judge")
    print(f"  Weighted Cohen's Kappa (quadratic weights)")
    print(f"  {N_RESAMPLES:,}-resample bootstrap {int(CI_LEVEL * 100)}% CI")
    print(f"{'═' * 60}\n")

    small_sample_warning(n_items)

    print_kappa_table(ratings)

    print("  Scale: <0.20 slight | 0.21–0.40 fair | 0.41–0.60 moderate")
    print("         0.61–0.80 substantial | >0.80 almost perfect\n")


if __name__ == "__main__":
    main()
