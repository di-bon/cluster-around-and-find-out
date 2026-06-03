"""
Extract experiment metrics from JSONL log files.

For each log file, finds the entries that contain an "inputs.condition" field,
and saves their output.metrics (excluding the "reasoning" sub-field) to a JSON file.

Usage:
    python extract_metrics.py <logfile.jsonl> [<logfile2.jsonl> ...] [-o <output_file.json>]
"""

import argparse
import json
from pathlib import Path


def extract_metrics(path: Path) -> list[dict]:
    """Return a list of result dicts for each conditioned entry in the file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Keep only entries that have inputs.condition
            inputs = entry.get("inputs", {})
            if "condition" not in inputs:
                continue

            # Dig out outputs.metrics
            metrics = entry.get("outputs", {}).get("metrics", {})

            # Drop "reasoning" from the nested "judgement" dict if present
            judgement = metrics.get("judgement", {})
            cleaned_judgement = {k: v for k, v in judgement.items() if k != "reasoning"}

            cleaned_metrics = {**metrics}
            if "judgement" in cleaned_metrics:
                cleaned_metrics["judgement"] = cleaned_judgement

            results.append(
                {
                    "condition": inputs["condition"],
                    "n_documents": inputs.get("n_documents"),
                    "metrics": cleaned_metrics,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract metrics from JSONL logs into a structured JSON file."
    )
    parser.add_argument(
        "logfiles", nargs="+", type=Path, help="One or more JSONL log files"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("metrics_output.json"),
        help="Path to the output JSON file (default: metrics_output.json)",
    )

    args = parser.parse_args()

    # Initialize the required top-level structure
    output_data = {"experiments": []}

    for path in args.logfiles:
        if not path.exists():
            print(f"[WARNING] File not found: {path}")
            continue

        results = extract_metrics(path)
        if not results:
            print(
                f"[WARNING] No entries with a 'condition' field found in {path.name}."
            )
            continue

        # Separate the metrics based on their condition type
        question_metrics = []
        baseline_metrics = []

        for r in results:
            if r["condition"].startswith("interview_"):
                question_metrics.append(r)
            elif r["condition"] == "baseline":
                baseline_metrics.append(r)

        # Append the structured experiment dictionary to the list
        output_data["experiments"].append(
            {
                "question_metrics": question_metrics,
                "baseline_metrics": baseline_metrics,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Save the collected results to the chosen JSON file
    if output_data["experiments"]:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved extracted metrics to {args.output}")
    else:
        print("No metrics were extracted from any of the files.")


if __name__ == "__main__":
    main()
