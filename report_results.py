#!/usr/bin/env python3
"""
Collect Hits@k / MRR from pykeen results.json files and export them to CSV.

`KGC.py` trains each model with pykeen's `pipeline()` and saves its output
with `results.save_to_directory(...)`, which writes a `results.json`
alongside the trained model, one per `<results_path>/<model>/` directory
(e.g. `Output/french_royalty/enriched_synth_french_royalty/TuckER/results.json`). That file's
`"metrics"` key holds pykeen's full evaluation report, nested as
`metrics[rank_type][filtering]`, where `rank_type` is "head", "tail" or
"both" and `filtering` is "optimistic", "pessimistic" or "realistic".

This script walks a results directory recursively
(`<root>/<dataset...>/<model>/results.json`, where `<dataset...>` is one or
more folders such as `french_royalty/enriched_synth_french_royalty`), pulls Hits@1/3/5/10 and MRR out
of the "both"/"realistic" slice of each one -- "both" combines head and
tail prediction, "realistic" is the standard filtered-ranking evaluation
reported in the KG completion literature -- and writes one row per
dataset/model pair to a CSV.

Usage:

    python report_results.py
    python report_results.py --root Output --output Output/metrics_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "Output"
DEFAULT_OUTPUT = DEFAULT_ROOT / "metrics_report.csv"

CSV_FIELDS = [
    "dataset",
    "model",
    "hits_at_1",
    "hits_at_3",
    "hits_at_5",
    "hits_at_10",
    "mrr",
    "count",
]

# pykeen's evaluation report is sliced by rank_type ("head"/"tail"/"both")
# and filtering ("optimistic"/"pessimistic"/"realistic"). "both" combines
# head- and tail-prediction ranks; "realistic" is the standard filtered
# evaluation reported in the KG completion literature (see module
# docstring), so that's the slice pulled out here.
RANK_TYPE = "both"
FILTERING = "realistic"


def find_results(root: Path):
    """
    Yield (dataset, model, results_json_path) for every pykeen
    `results.json` found under `root`, assuming the
    `<root>/<dataset...>/<model>/results.json` layout that `KGC.py` produces
    (`<dataset...>` is the `results_path` folder chain, e.g.
    `french_royalty/enriched_synth_french_royalty`, reported with "/").
    """
    for results_json in sorted(root.rglob("results.json")):
        model_dir = results_json.parent
        dataset = model_dir.parent.relative_to(root).as_posix()
        if dataset == ".":
            continue
        yield dataset, model_dir.name, results_json


def extract_metrics(results_json: Path) -> dict | None:
    """
    Pull Hits@1/3/5/10 and MRR out of one pykeen `results.json` file.

    Returns None (and prints a warning) if the file can't be read or is
    missing the expected `metrics[RANK_TYPE][FILTERING]` slice, so a single
    malformed/incomplete result doesn't abort the whole report.
    """
    try:
        with results_json.open(encoding="utf-8") as f:
            data = json.load(f)
        metrics = data["metrics"][RANK_TYPE][FILTERING]
        return {
            "hits_at_1": metrics["hits_at_1"],
            "hits_at_3": metrics["hits_at_3"],
            "hits_at_5": metrics["hits_at_5"],
            "hits_at_10": metrics["hits_at_10"],
            # pykeen names MRR "inverse_harmonic_mean_rank" internally.
            "mrr": metrics["inverse_harmonic_mean_rank"],
            "count": metrics.get("count"),
        }
    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"warning: skipping {results_json} ({e})")
        return None


def build_report(root: Path) -> list[dict]:
    rows = []
    for dataset, model, results_json in find_results(root):
        metrics = extract_metrics(results_json)
        if metrics is None:
            continue
        rows.append({"dataset": dataset, "model": model, **metrics})
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect Hits@k / MRR from pykeen results.json files into one CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Results directory to scan, expected to contain <dataset...>/<model>/results.json",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.root.exists():
        print(f"error: results directory not found: {args.root}")
        return 1

    rows = build_report(args.root)
    if not rows:
        print(f"error: no results.json files found under {args.root}")
        return 1

    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
