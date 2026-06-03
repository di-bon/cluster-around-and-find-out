"""
Evaluate LLM conversation experiment results using bootstrap confidence intervals.

Input: one JSON file per question category.

Expected JSON structure per file:
{
  "experiments": [
    {
      "question_metrics": [
        { "condition": "interview_yes_no", "n_documents": 500,
          "metrics": { "turns": int, "total_tokens": int, "total_user_tokens": int,
            "judgement": { "coherence": float, "alignment": float, "separation": float } } }
      ],
      "baseline_metrics": [
        { "condition": "baseline", "n_documents": 500, "metrics": { ... } }
      ]
    },
    ...
  ]
}

Usage:
    python evaluate_experiment.py yes_no.json multiple_choice.json open_ended.json
"""

import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────

N_RESAMPLES = 10_000
RNG_SEED = 42
CI_LEVEL = 0.95


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class Sample:
    """Holds the outcome vectors for one group of conversations."""

    name: str
    turns: np.ndarray
    total_tokens: np.ndarray
    user_tokens: np.ndarray
    faithfulness: np.ndarray  # mean of coherence, alignment, separation
    coherence: np.ndarray
    alignment: np.ndarray
    separation: np.ndarray


# ── Loading ──────────────────────────────────────────────────────────────────


def parse_row(entry: dict) -> dict:
    """Extract a flat metric dict from a single question_metrics/baseline_metrics entry."""
    m = entry.get("metrics", {})
    j = m.get("judgement", {})
    return {
        "turns": m.get("turns", np.nan),
        "total_tokens": m.get("total_tokens", np.nan),
        "user_tokens": m.get("total_user_tokens", np.nan),
        "faithfulness": np.nanmean(
            [
                j.get("coherence", np.nan),
                j.get("alignment", np.nan),
                j.get("separation", np.nan),
            ]
        ),
        "coherence": j.get("coherence", np.nan),
        "alignment": j.get("alignment", np.nan),
        "separation": j.get("separation", np.nan),
    }


def load_file(path: Path) -> tuple[Sample, Sample]:
    """
    Parse a results JSON file and return (condition_sample, baseline_sample).

    Expected structure:
    {
        "experiments": [
            {
                "question_metrics": [ { "condition": ..., "metrics": { ... } }, ... ],
                "baseline_metrics": [ { "condition": "baseline", "metrics": { ... } }, ... ]
            },
            ...
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)

    experiments = data.get("experiments", [])
    if not experiments:
        raise ValueError(f"No 'experiments' key found in {path}")

    condition_rows, baseline_rows = [], []
    for exp in experiments:
        for entry in exp.get("question_metrics", []):
            condition_rows.append(parse_row(entry))
        for entry in exp.get("baseline_metrics", []):
            baseline_rows.append(parse_row(entry))

    if not condition_rows:
        raise ValueError(f"No question_metrics entries found in {path}")
    if not baseline_rows:
        raise ValueError(f"No baseline_metrics entries found in {path}")

    stem = path.stem  # e.g. "yes_no", "multiple_choice", "open_ended"

    def to_sample(rows, label):
        return Sample(
            name=label,
            turns=np.array([r["turns"] for r in rows], dtype=float),
            total_tokens=np.array([r["total_tokens"] for r in rows], dtype=float),
            user_tokens=np.array([r["user_tokens"] for r in rows], dtype=float),
            faithfulness=np.array([r["faithfulness"] for r in rows], dtype=float),
            coherence=np.array([r["coherence"] for r in rows], dtype=float),
            alignment=np.array([r["alignment"] for r in rows], dtype=float),
            separation=np.array([r["separation"] for r in rows], dtype=float),
        )

    return (
        to_sample(condition_rows, stem),
        to_sample(baseline_rows, f"{stem}_baseline"),
    )


# ── Bootstrap ────────────────────────────────────────────────────────────────


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n: int = N_RESAMPLES, seed: int = RNG_SEED
) -> tuple[float, float, float]:
    """
    Bootstrap the difference in means (a − b).
    Returns (observed_diff, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    observed = np.nanmean(a) - np.nanmean(b)

    diffs = np.empty(n)
    for i in range(n):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.nanmean(ra) - np.nanmean(rb)

    alpha = 1 - CI_LEVEL
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    return observed, lo, hi


def significant(lo: float, hi: float) -> bool:
    """CI excludes zero → statistically significant difference."""
    return not (lo <= 0 <= hi)


# ── Reporting ────────────────────────────────────────────────────────────────

OUTCOMES = [
    # (attribute, label, lower_is_better)
    ("turns", "Turns to convergence", True),
    ("total_tokens", "Total tokens", True),
    ("user_tokens", "User response tokens", True),
    ("faithfulness", "Mean faithfulness score", False),
    ("coherence", "  Coherence", False),
    ("alignment", "  Alignment", False),
    ("separation", "  Separation", False),
]


def direction_symbol(diff: float, lo: float, hi: float, lower_is_better: bool) -> str:
    """Return a human-readable verdict."""
    if not significant(lo, hi):
        return "~ no significant difference"
    if lower_is_better:
        return "✓ A better" if diff < 0 else "✗ B better"
    else:
        return "✓ A better" if diff > 0 else "✗ B better"


def print_comparison(
    label_a: str,
    label_b: str,
    sample_a: Sample,
    sample_b: Sample,
    primary_only: bool = False,
) -> None:
    """Print a full comparison table between two samples."""
    print(f"\n  A = {label_a}   vs   B = {label_b}")
    print(
        f"  {'Outcome':<30} {'Mean A':>8} {'Mean B':>8} "
        f"{'Diff (A−B)':>12} {'95% CI':>22}  Verdict"
    )
    print("  " + "-" * 100)

    for attr, outcome_label, lower_is_better in OUTCOMES:
        a_vals = getattr(sample_a, attr)
        b_vals = getattr(sample_b, attr)
        diff, lo, hi = bootstrap_diff(a_vals, b_vals)
        verdict = direction_symbol(diff, lo, hi, lower_is_better)
        tag = " [PRIMARY]" if attr == "turns" else ""
        print(
            f"  {outcome_label + tag:<30} "
            f"{np.nanmean(a_vals):>8.2f} "
            f"{np.nanmean(b_vals):>8.2f} "
            f"{diff:>+12.3f} "
            f"[{lo:>+9.3f}, {hi:>+9.3f}]  {verdict}"
        )

        if primary_only and attr == "turns":
            break


def separator(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print("═" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_experiment.py <file1.json> [file2.json] ...")
        sys.exit(1)

    files = [Path(a) for a in sys.argv[1:]]
    condition_samples: list[Sample] = []
    baseline_samples: list[Sample] = []

    for path in files:
        if not path.exists():
            print(f"[WARNING] Not found: {path}")
            continue
        cond, base = load_file(path)
        condition_samples.append(cond)
        baseline_samples.append(base)
        print(
            f"Loaded '{path.name}': "
            f"{len(cond.turns)} condition run(s), "
            f"{len(base.turns)} baseline run(s)."
        )

    if not condition_samples:
        print("No data loaded. Exiting.")
        sys.exit(1)

    print(
        f"\nBootstrap settings: {N_RESAMPLES:,} resamples, "
        f"{int(CI_LEVEL * 100)}% CI, seed={RNG_SEED}"
    )

    # ── Stage 1: each condition vs. its own baseline ──────────────────────────
    separator("STAGE 1 — Condition vs. Baseline (per question type)")
    for cond, base in zip(condition_samples, baseline_samples):
        print(f"\n{'─' * 60}")
        print(f"  Question type: {cond.name.upper()}")
        print_comparison(cond.name, base.name, cond, base)

    # ── Stage 2: condition types vs. each other ───────────────────────────────
    if len(condition_samples) >= 2:
        separator("STAGE 2 — Question Type vs. Question Type")
        for a, b in combinations(condition_samples, 2):
            print(f"\n{'─' * 60}")
            print_comparison(a.name, b.name, a, b)

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(condition_samples) >= 2:
        separator("SUMMARY — Primary Outcome: Turns to Convergence")
        print(f"\n  {'Question type':<25} {'Mean turns':>12} {'Rank':>6}")
        print("  " + "-" * 45)
        ranked = sorted(condition_samples, key=lambda s: np.nanmean(s.turns))
        for rank, s in enumerate(ranked, 1):
            print(f"  {s.name:<25} {np.nanmean(s.turns):>12.2f} {rank:>6}")
        print(f"\n  Best (fewest turns): {ranked[0].name}")
        print()


if __name__ == "__main__":
    main()
