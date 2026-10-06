"""
Collect Hits@k / MRR from pykeen results.json files and export them to CSV,
then analyse every single test prediction behind those numbers.

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

The per-prediction analysis goes below those averages. `results.json` only
stores aggregates, so the rank of every test prediction is recomputed from
the run's `trained_model.pkl` and the `test` split KGC.py saves next to the
model folders (`save_splits: true`), the same way pykeen ranked it. These
ranks average to the numbers in the table; a run whose recomputed MRR
doesn't match its results.json (its splits were rewritten by a later run)
is left out of the analysis.

The reference KG (`--reference`, `source` by default) is then compared
prediction by prediction with every sibling dataset matching `--compare`
(`skgg*` by default, e.g. `skgg_std=1`): synthetic graphs generated from it
that keep its entity names. They share no test triples with it  so
predictions are paired by query: a tail query `(h, r, ?)` or a head query
`(?, r, t)` held out in both graphs, each graph
ranking its own held-out answers. PyGraft renames every entity, so its
queries never pair.

Each relation is also compared between the reference and every other graph
of its KG, PyGraft included, over all the relation's test predictions: the
difference in MRR, and tests and an effect size for the rank distributions.

The analysis is written to `--analysis-dir`:

    predictions.csv                  one row per test prediction, every run
    relation_report.csv              the table's metrics per relation
    relation_comparison.csv          each relation's ranks, reference vs every graph
    query_comparison.csv             one row per query paired between two graphs
    query_comparison_summary.csv     correlation and agreement per stratum and k
    <kg>_relations.png               MRR and Hits@10 per relation, model and graph
    <kg>_<model>_hits_at_k.png       Hits@k curves per relation, one line per graph
    <kg>_<compared>_<model>.png      rank distributions and rank transitions

Usage:

    python report_results.py
    python report_results.py --root Output --output Output/metrics_report.csv
    python report_results.py --reference source --compare "skgg*" --analysis-dir Output/prediction_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch
from matplotlib import path as mpath
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.ticker import PercentFormatter
from pykeen.triples import TriplesFactory
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR / "Output"
DEFAULT_OUTPUT = DEFAULT_ROOT / "metrics_report.csv"
DEFAULT_ANALYSIS_DIR = DEFAULT_ROOT / "prediction_analysis"

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

# The dataset whose predictions the others are compared with, and
# shell-style patterns naming its sibling datasets to compare: graphs
# generated from that KG that keep its entity names (PyGraft's don't).
DEFAULT_REFERENCE = "source"
DEFAULT_COMPARED = ["skgg*"]

HITS_AT = (1, 3, 5, 10)
HITS_COLUMNS = [f"hits_at_{k}" for k in HITS_AT]
# A query is "accurate at k" in a graph when at least this share of its
# held-out answers rank within the top k. Most queries have one answer, so
# this is usually just a hit or a miss.
ACCURATE_SHARE = 0.5
# Largest gap allowed between the MRR recomputed from a run's saved model
# and splits and the MRR in its results.json. A bigger gap means the splits
# on disk were written by another run.
MRR_TOLERANCE = 1e-3
# Test triples scored per forward pass when recomputing ranks.
SCORE_BATCH_SIZE = 256
# Resamples of the bootstrap interval of each relation's MRR difference, with
# a fixed seed so that reruns write identical files.
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
# Holm-adjusted p-values below this are significant in the plots.
SIGNIFICANCE = 0.05
# Upper bounds of |Cliff's delta| for each effect size label (Romano et al.,
# 2006); anything above the last bound is "large".
EFFECT_SIZES = [(0.147, "negligible"), (0.33, "small"), (0.474, "medium")]

PREDICTION_COLUMNS = [
    "dataset",
    "model",
    "side",
    "head",
    "relation",
    "tail",
    "rank",
    "candidates",
    "reciprocal_rank",
    *HITS_COLUMNS,
]

# Rank buckets of the plots, as (lower, upper] edges. Realistic ranks are
# fractional when scores tie, e.g. 1.5 falls in "2–3".
BUCKET_EDGES = [0, 1, 3, 10, 100, np.inf]
BUCKET_LABELS = ["1", "2–3", "4–10", "11–100", ">100"]

# Plot colors and mark sizes, from the dataviz reference palette (light).
PLOT_DPI = 150
PX = 72 / PLOT_DPI  # one pixel, in points
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
# One color per graph, in the palette's validated order (see graph_order).
CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
SERIES_COLORS = CATEGORICAL[:2]  # reference graph, compared graph
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue",
    [
        "#cde2fb",
        "#b7d3f6",
        "#9ec5f4",
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    ],
)
BAR_WIDTH_PX = 24
BAR_GAP_PX = 2
BAR_RADIUS_PX = 4


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


def read_split(path: Path) -> np.ndarray:
    """Read a split KGC.py saved with `save_splits`: one tab-separated labeled triple per line."""
    return pd.read_csv(
        path, sep="\t", header=None, dtype=str, keep_default_na=False
    ).to_numpy()


def rank_predictions(dataset_dir: Path, model_dir: Path) -> pd.DataFrame | None:
    """
    Rank both predictions of every test triple of one run, as pykeen did.

    Each test triple (h, r, t) gives a tail prediction (h, r, ?) and a head
    prediction (?, r, t). Like `pipeline()`'s filtered evaluation, the other
    answers known to be true -- the training triples saved in
    `training_triples/` and the `test` split KGC.py saved in `dataset_dir` --
    are removed before ranking, and ties get the "realistic" rank: the mean
    of the best and worst position among equal scores.

    Returns one row per prediction (side, head, relation, tail, rank,
    candidates), where `side` is the entity being predicted and `candidates`
    the number of entities it was ranked among (all but the filtered
    answers, pykeen's "number of options"), or None (and prints a warning)
    if a file is missing or the test labels aren't in the model's vocabulary.
    """
    needed = [
        dataset_dir / "test",
        model_dir / "trained_model.pkl",
        model_dir / "training_triples",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        print(
            f"warning: no prediction analysis for {model_dir} (missing {', '.join(missing)})"
        )
        return None

    test = read_split(dataset_dir / "test")
    training = TriplesFactory.from_path_binary(model_dir / "training_triples")
    entity_to_id, relation_to_id = training.entity_to_id, training.relation_to_id
    try:
        test_ids = torch.tensor(
            [[entity_to_id[h], relation_to_id[r], entity_to_id[t]] for h, r, t in test]
        )
    except KeyError as e:
        print(
            f"warning: no prediction analysis for {model_dir} (test label {e} is not in the "
            "model's vocabulary, so its test split was written by another run)"
        )
        return None

    # Every answer known to be true, per tail query (h, r) and head query (r, t).
    tails, heads = defaultdict(list), defaultdict(list)
    for h, r, t in torch.cat([training.mapped_triples, test_ids]).tolist():
        tails[h, r].append(t)
        heads[r, t].append(h)

    # The pickle holds the whole model object, not just its weights.
    model = torch.load(
        model_dir / "trained_model.pkl", map_location="cpu", weights_only=False
    )
    model.eval()
    ranks, candidates = {}, {}
    with torch.inference_mode():
        for side, column in (("head", 0), ("tail", 2)):
            side_ranks, side_candidates = [], []
            for batch in torch.split(test_ids, SCORE_BATCH_SIZE):
                scores = model.predict(hrt_batch=batch, target=side)
                rows = torch.arange(len(batch))
                true_scores = scores[rows, batch[:, column]]
                for i, (h, r, t) in enumerate(batch.tolist()):
                    scores[i, heads[r, t] if side == "head" else tails[h, r]] = float(
                        "nan"
                    )
                scores[rows, batch[:, column]] = true_scores
                # Filtered (NaN) scores compare False, so they never count.
                better = (scores > true_scores[:, None]).sum(dim=1)
                not_worse = (scores >= true_scores[:, None]).sum(dim=1)
                side_ranks.append((better + 1 + not_worse) / 2)
                side_candidates.append(torch.isfinite(scores).sum(dim=1))
            ranks[side] = torch.cat(side_ranks).double().numpy()
            candidates[side] = torch.cat(side_candidates).numpy()

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "side": side,
                    "head": test[:, 0],
                    "relation": test[:, 1],
                    "tail": test[:, 2],
                    "rank": ranks[side],
                    "candidates": candidates[side],
                }
            )
            for side in ("head", "tail")
        ],
        ignore_index=True,
    )


def build_predictions(root: Path, rows: list[dict]) -> pd.DataFrame:
    """
    Recompute the per-prediction ranks of every run in the table (`rows`
    from `build_report`), keeping only the runs whose ranks reproduce the
    MRR in their results.json.
    """
    frames = []
    for row in rows:
        dataset, model = row["dataset"], row["model"]
        ranks = rank_predictions(root / dataset, root / dataset / model)
        if ranks is None:
            continue
        mrr = (1 / ranks["rank"]).mean()
        if abs(mrr - row["mrr"]) > MRR_TOLERANCE:
            print(
                f"warning: no prediction analysis for {dataset}/{model} (MRR {mrr:.4f} from its "
                f"saved model and splits vs {row['mrr']:.4f} in results.json, so the splits on "
                "disk were written by another run)"
            )
            continue
        print(f"Ranked {len(ranks)} test predictions of {dataset}/{model}")
        frames.append(ranks.assign(dataset=dataset, model=model))

    if not frames:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["reciprocal_rank"] = 1 / predictions["rank"]
    for k in HITS_AT:
        predictions[f"hits_at_{k}"] = (predictions["rank"] <= k).astype(int)
    return predictions[PREDICTION_COLUMNS]


def relation_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """The table's metrics per relation, head and tail predictions pooled like the "both" slice."""
    grouped = predictions.groupby(["dataset", "model", "relation"])
    report = grouped[HITS_COLUMNS].mean()
    report["mrr"] = grouped["reciprocal_rank"].mean()
    report["count"] = grouped.size()
    return report.reset_index()


def query_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Group predictions by query, one row per (dataset, model, side, relation,
    entity), where `entity` is the query's known entity: the head of a tail
    query (h, r, ?) and the tail of a head query (?, r, t). A query has one
    prediction per held-out answer; `answers` counts them, and `mrr` and
    `hits_at_k` (the share of answers within the top k) average over them.
    """
    entity = predictions["head"].where(
        predictions["side"] == "tail", predictions["tail"]
    )
    grouped = predictions.assign(entity=entity).groupby(
        ["dataset", "model", "side", "relation", "entity"]
    )
    queries = grouped[["reciprocal_rank", *HITS_COLUMNS]].mean()
    queries = queries.rename(columns={"reciprocal_rank": "mrr"})
    queries.insert(0, "answers", grouped.size())
    return queries.reset_index()


def split_dataset(table: pd.DataFrame) -> pd.DataFrame:
    """
    Add the `kg` and `graph` columns: the dataset's parent folder and its own
    name, e.g. `french_royalty` and `skgg_std=1`.
    """
    dataset = table["dataset"].map(PurePosixPath)
    return table.assign(
        kg=dataset.map(lambda p: p.parent.as_posix()),
        graph=dataset.map(lambda p: p.name),
    )


def is_compared(graph: str, reference: str, patterns: list[str]) -> bool:
    """Whether `graph` is compared with the reference graph: it matches a pattern."""
    return graph != reference and any(fnmatchcase(graph, p) for p in patterns)


def graph_order(graphs, reference: str, patterns: list[str]) -> list[str]:
    """
    Sort a KG's graphs for the plots: the reference first, then the graphs
    compared with it, then the rest, so the colors follow the graphs.
    """
    return sorted(
        set(graphs),
        key=lambda g: (g != reference, not is_compared(g, reference, patterns), g),
    )


def compare_queries(
    queries: pd.DataFrame, reference: str, patterns: list[str]
) -> pd.DataFrame:
    """
    Pair the queries of `<kg>/<reference>` with those of every sibling
    dataset `<kg>/<compared>` whose name matches one of `patterns`, for the
    models trained on both. A query pairs when it is held out in both
    graphs; each graph keeps its own answers and values, in the
    `reference_*` and `compared_*` columns.
    """
    queries = split_dataset(queries)
    keys = ["kg", "model", "side", "relation", "entity"]
    values = ["answers", "mrr", *HITS_COLUMNS]
    compared_rows = queries["graph"].map(
        lambda graph: is_compared(graph, reference, patterns)
    )
    ref = (
        queries[queries["graph"] == reference]
        .set_index(keys)[values]
        .add_prefix("reference_")
    )
    cmp = queries[compared_rows].set_index(keys)[["graph", *values]]
    cmp = cmp.rename(
        columns={"graph": "compared", **{v: f"compared_{v}" for v in values}}
    )
    paired = ref.join(cmp, how="inner").reset_index()
    paired.insert(1, "reference", reference)
    paired.insert(2, "compared", paired.pop("compared"))
    paired.insert(
        len(keys) + 2,
        "query",
        np.where(
            paired["side"] == "tail",
            "(" + paired["entity"] + ", " + paired["relation"] + ", ?)",
            "(?, " + paired["relation"] + ", " + paired["entity"] + ")",
        ),
    )
    return paired.sort_values(
        ["kg", "compared", "model", "relation", "side", "entity"], ignore_index=True
    )


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman's rho and its p-value; NaN under 3 pairs or when a side is constant."""
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def agreement(paired: pd.DataFrame, k: int) -> dict:
    """
    Cross-tabulate the paired queries by whether they are accurate at k in
    each graph, and how often the compared graph is accurate after the
    reference graph is (or isn't).
    """
    ref = paired[f"reference_hits_at_{k}"] >= ACCURATE_SHARE
    cmp = paired[f"compared_hits_at_{k}"] >= ACCURATE_SHARE
    both, only_ref = int((ref & cmp).sum()), int((ref & ~cmp).sum())
    only_cmp, neither = int((~ref & cmp).sum()), int((~ref & ~cmp).sum())
    # Cohen's kappa: the observed agreement corrected for the agreement
    # expected by chance from how often each graph is accurate.
    observed = (both + neither) / len(paired)
    expected = ref.mean() * cmp.mean() + (1 - ref.mean()) * (1 - cmp.mean())
    return {
        "both_accurate": both,
        "only_reference": only_ref,
        "only_compared": only_cmp,
        "neither": neither,
        "compared_accurate_when_reference_accurate": ratio(both, both + only_ref),
        "compared_accurate_when_reference_inaccurate": ratio(
            only_cmp, only_cmp + neither
        ),
        "cohen_kappa": ratio(observed - expected, 1 - expected),
    }


def summarize_comparison(paired: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each pair of graphs per KG and model, over all paired queries,
    head and tail queries apart, and each relation (the strata), with one
    row per k: the per-query MRR of each graph, their rank correlation, and
    the agreement on which queries are accurate at k.
    """
    rows = []
    for (kg, reference, compared, model), group in paired.groupby(
        ["kg", "reference", "compared", "model"]
    ):
        strata = [("all", "both", group)]
        strata += [("all", side, g) for side, g in group.groupby("side")]
        strata += [(relation, "both", g) for relation, g in group.groupby("relation")]
        for relation, side, g in strata:
            rho, p = spearman(g["reference_mrr"], g["compared_mrr"])
            for k in HITS_AT:
                rows.append(
                    {
                        "kg": kg,
                        "reference": reference,
                        "compared": compared,
                        "model": model,
                        "relation": relation,
                        "side": side,
                        "k": k,
                        "paired_queries": len(g),
                        "reference_mrr": g["reference_mrr"].mean(),
                        "compared_mrr": g["compared_mrr"].mean(),
                        "spearman_rho": rho,
                        "spearman_p": p,
                        **agreement(g, k),
                    }
                )
    return pd.DataFrame(rows)


def normalized_rank(predictions: pd.DataFrame) -> pd.Series:
    """
    The rank as a share of the candidates, (rank - 1) / (candidates - 1): 0
    when ranked first, 1 when ranked last, 0.5 on average for a random
    guess. It puts graphs with different numbers of entities on one scale.
    """
    return (predictions["rank"] - 1) / (predictions["candidates"] - 1)


def holm(p_values: pd.Series) -> pd.Series:
    """Holm-adjust one family of p-values (step-down, capped at 1)."""
    ordered = p_values.sort_values()
    adjusted = (ordered * np.arange(len(ordered), 0, -1)).cummax().clip(upper=1)
    return adjusted.reindex(p_values.index)


def effect_size(delta: float) -> str:
    """Label the size of a Cliff's delta: negligible, small, medium or large."""
    for bound, label in EFFECT_SIZES:
        if abs(delta) < bound:
            return label
    return "large"


def bootstrap_difference(
    x: pd.Series, y: pd.Series, rng: np.random.Generator
) -> tuple[float, float]:
    """95% percentile bootstrap interval of mean(y) - mean(x), resampling each."""
    x, y = x.to_numpy(), y.to_numpy()
    means_x = x[rng.integers(0, len(x), (BOOTSTRAP_SAMPLES, len(x)))].mean(axis=1)
    means_y = y[rng.integers(0, len(y), (BOOTSTRAP_SAMPLES, len(y)))].mean(axis=1)
    low, high = np.percentile(means_y - means_x, [2.5, 97.5])
    return float(low), float(high)


def compare_relations(predictions: pd.DataFrame, reference: str) -> pd.DataFrame:
    """
    Compare each relation's test predictions in the reference graph with
    those in every other graph of the same KG (`predictions` with the `kg`
    and `graph` columns of `split_dataset`), for each model trained on both.
    Unlike `compare_queries`, this needs no shared entities, so PyGraft is
    compared too, and it uses every test prediction.

    `mrr_difference` (compared - reference) comes with a 95% bootstrap
    interval. The rank distributions are compared on `normalized_rank`:
    Cliff's delta is P(compared ranks better) - P(reference ranks better),
    negative when the compared graph does worse; the Mann-Whitney test asks
    whether one graph tends to rank better, the KS test whether the
    distributions differ at all. Their p-values are Holm-adjusted over the
    relations of each graph pair and model.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    k = max(HITS_AT)
    predictions = predictions.assign(normalized_rank=normalized_rank(predictions))
    rows = []
    for (kg, model), runs in predictions.groupby(["kg", "model"]):
        ref = runs[runs["graph"] == reference]
        for compared, cmp in runs[runs["graph"] != reference].groupby("graph"):
            for relation in sorted(set(ref["relation"]) & set(cmp["relation"])):
                x = ref[ref["relation"] == relation]
                y = cmp[cmp["relation"] == relation]
                # U counts the pairs where the reference ranks further down
                # than the compared graph, i.e. where the compared one is better.
                mann_whitney = mannwhitneyu(
                    x["normalized_rank"], y["normalized_rank"], alternative="two-sided"
                )
                ks = ks_2samp(x["normalized_rank"], y["normalized_rank"])
                delta = 2 * mann_whitney.statistic / (len(x) * len(y)) - 1
                low, high = bootstrap_difference(
                    x["reciprocal_rank"], y["reciprocal_rank"], rng
                )
                rows.append(
                    {
                        "kg": kg,
                        "reference": reference,
                        "compared": compared,
                        "model": model,
                        "relation": relation,
                        "reference_predictions": len(x),
                        "compared_predictions": len(y),
                        "reference_mrr": x["reciprocal_rank"].mean(),
                        "compared_mrr": y["reciprocal_rank"].mean(),
                        "mrr_difference": (
                            y["reciprocal_rank"].mean() - x["reciprocal_rank"].mean()
                        ),
                        "mrr_difference_low": low,
                        "mrr_difference_high": high,
                        f"reference_hits_at_{k}": x[f"hits_at_{k}"].mean(),
                        f"compared_hits_at_{k}": y[f"hits_at_{k}"].mean(),
                        "cliffs_delta": delta,
                        "effect": effect_size(delta),
                        "mannwhitney_p": mann_whitney.pvalue,
                        "ks_statistic": ks.statistic,
                        "ks_p": ks.pvalue,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["kg", "reference", "compared", "model", "relation"]
        )

    comparison = pd.DataFrame(rows)
    for test in ("mannwhitney", "ks"):
        family_p = comparison.groupby(["kg", "compared", "model"])[f"{test}_p"]
        comparison.insert(
            comparison.columns.get_loc(f"{test}_p") + 1,
            f"{test}_p_holm",
            family_p.transform(holm),
        )
    return comparison


def rank_buckets(ranks: pd.Series) -> pd.Series:
    """Label each rank with its plot bucket ("1", "2–3", ..., ">100")."""
    return pd.cut(ranks, bins=BUCKET_EDGES, labels=BUCKET_LABELS)


def ink_on(fill) -> str:
    """Ink or white, whichever has the higher WCAG contrast against `fill`."""

    def luminance(color) -> float:
        c = np.array(to_rgb(color))
        c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return float(c @ [0.2126, 0.7152, 0.0722])

    fill_luminance = luminance(fill)
    on_white = 1.05 / (fill_luminance + 0.05)
    on_ink = (fill_luminance + 0.05) / (luminance(INK) + 0.05)
    return "#ffffff" if on_white > on_ink else INK


def draw_column(ax, x: float, width: float, height: float, color: str) -> None:
    """
    Draw a column from 0 up to `height`, square at the baseline and with
    rounded top corners. The corner radius is converted from pixels to data
    units, so call this once the axes' limits and position are final.
    """
    if height <= 0:
        return
    px_x, px_y = np.abs(ax.transData.transform((1, 1)) - ax.transData.transform((0, 0)))
    rx, ry = min(BAR_RADIUS_PX / px_x, width / 2), min(BAR_RADIUS_PX / px_y, height)
    left, right = x - width / 2, x + width / 2
    P = mpath.Path
    vertices = [
        (left, 0),
        (left, height - ry),
        (left, height),
        (left + rx, height),
        (right - rx, height),
        (right, height),
        (right, height - ry),
        (right, 0),
        (left, 0),
    ]
    codes = [
        P.MOVETO,
        P.LINETO,
        P.CURVE3,
        P.CURVE3,
        P.LINETO,
        P.CURVE3,
        P.CURVE3,
        P.LINETO,
        P.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(P(vertices, codes), facecolor=color, edgecolor="none"))


def style_axes(ax) -> None:
    """Chart surface, muted tick labels, and no ticks or frame."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(length=0, colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_grouped_columns(
    ax, labels: list[str], series: list[tuple[np.ndarray, str]], top: float
) -> None:
    """
    Draw one group of columns per label, one column per (values, color) in
    `series`, on a y axis from 0 to `top` with a hairline grid and baseline.
    A NaN value leaves its column out. The axes limits are set first, since
    the column widths are given in pixels.
    """
    x = np.arange(len(labels))
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, top)
    px_per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    gap = BAR_GAP_PX / px_per_unit
    # Columns are at most BAR_WIDTH_PX wide, and a group fills at most 80% of its slot.
    width = min(
        BAR_WIDTH_PX / px_per_unit, (0.8 - (len(series) - 1) * gap) / len(series)
    )
    span = len(series) * width + (len(series) - 1) * gap
    for i, (values, color) in enumerate(series):
        offset = (width - span) / 2 + i * (width + gap)
        for xi, value in zip(x + offset, values):
            if not np.isnan(value):
                draw_column(ax, xi, width, value, color)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", color=GRIDLINE, linewidth=PX)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set(visible=True, color=BASELINE, linewidth=PX)


def fmt(value: float, spec: str) -> str:
    return "n/a" if pd.isna(value) else format(value, spec)


def plot_comparison(
    predictions: pd.DataFrame, paired: pd.DataFrame, path: Path
) -> None:
    """
    Plot one comparison of two graphs with one model (`paired` holds its
    `compare_queries` rows only). Left: how the ranks of all test
    predictions of each graph spread over the rank buckets. Right: for the
    paired queries, where the queries of each reference bucket land in the
    compared graph, as a share of the row and a count. A query is bucketed
    by 1 / its MRR, its rank when it has a single held-out answer.
    """
    kg, reference, compared, model = paired[
        ["kg", "reference", "compared", "model"]
    ].iloc[0]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 5.6), dpi=PLOT_DPI, gridspec_kw={"width_ratios": [1.2, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.72, bottom=0.12, wspace=0.28)
    for ax in (left, right):
        style_axes(ax)

    # Headline: does accurate in the reference graph mean accurate in the other?
    k = max(HITS_AT)
    rho, _ = spearman(paired["reference_mrr"], paired["compared_mrr"])
    rates = agreement(paired, k)
    when_accurate = rates["compared_accurate_when_reference_accurate"]
    when_inaccurate = rates["compared_accurate_when_reference_inaccurate"]
    title = " / ".join(PurePosixPath(kg).parts + (model,))
    fig.text(
        0.07,
        0.95,
        f"{title}: {reference} vs {compared}",
        color=INK,
        fontsize=13,
        fontweight="semibold",
        va="top",
    )
    fig.text(
        0.07,
        0.895,
        f"{len(paired):,} queries held out in both graphs · Spearman ρ = {fmt(rho, '.2f')} "
        f"between their per-query MRRs\nAt Hits@{k}, {compared} is accurate on "
        f"{fmt(when_accurate, '.0%')} of the queries {reference} gets right, and on "
        f"{fmt(when_inaccurate, '.0%')} of those it misses",
        color=INK_SECONDARY,
        fontsize=9.5,
        va="top",
        linespacing=1.5,
    )

    # Left: rank distribution of every test prediction, one column per graph and bucket.
    shares, totals = {}, {}
    for graph in (reference, compared):
        run = (predictions["dataset"] == (PurePosixPath(kg) / graph).as_posix()) & (
            predictions["model"] == model
        )
        totals[graph] = int(run.sum())
        shares[graph] = rank_buckets(predictions.loc[run, "rank"]).value_counts(
            normalize=True, sort=False
        )
    draw_grouped_columns(
        left,
        BUCKET_LABELS,
        [
            (shares[graph].reindex(BUCKET_LABELS).to_numpy(dtype=float), color)
            for graph, color in zip((reference, compared), SERIES_COLORS)
        ],
        top=max(s.max() for s in shares.values()) * 1.12,
    )
    left.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    left.set_xlabel("Rank of the true answer", color=INK_SECONDARY, fontsize=9.5)
    left.set_ylabel("Share of test predictions", color=INK_SECONDARY, fontsize=9.5)
    left.set_title(
        "All test predictions",
        loc="left",
        color=INK,
        fontsize=10.5,
        fontweight="semibold",
        pad=32,
    )
    left.legend(
        handles=[
            Patch(facecolor=color, label=f"{graph} ({totals[graph]:,} predictions)")
            for graph, color in zip((reference, compared), SERIES_COLORS)
        ],
        loc="lower left",
        bbox_to_anchor=(0, 1.03),
        ncols=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        handlelength=0.9,
        handleheight=0.9,
        borderaxespad=0.1,
        borderpad=0,
    )

    # Right: rank transitions of the paired queries, one row per reference bucket.
    counts = pd.crosstab(
        rank_buckets(1 / paired["reference_mrr"]),
        rank_buckets(1 / paired["compared_mrr"]),
        dropna=False,
    ).reindex(index=BUCKET_LABELS, columns=BUCKET_LABELS, fill_value=0)
    row_shares = counts.div(counts.sum(axis=1), axis=0)
    right.pcolormesh(
        np.ma.masked_invalid(row_shares.to_numpy(dtype=float)),
        cmap=SEQUENTIAL,
        vmin=0,
        vmax=1,
        edgecolors=SURFACE,
        linewidth=BAR_GAP_PX * PX,
    )
    right.set_aspect("equal")
    right.invert_yaxis()
    for i in range(len(BUCKET_LABELS)):
        for j in range(len(BUCKET_LABELS)):
            share = row_shares.iat[i, j]
            if np.isnan(share):
                continue
            color = ink_on(SEQUENTIAL(share))
            right.text(
                j + 0.5,
                i + 0.47,
                f"{share:.0%}",
                ha="center",
                va="bottom",
                color=color,
                fontsize=10,
                fontweight="semibold",
            )
            right.text(
                j + 0.5,
                i + 0.55,
                f"n={counts.iat[i, j]:,}",
                ha="center",
                va="top",
                color=color,
                fontsize=8,
            )
    ticks = np.arange(len(BUCKET_LABELS)) + 0.5
    right.set_xticks(ticks, BUCKET_LABELS)
    right.set_yticks(ticks, BUCKET_LABELS)
    right.set_xlabel(f"Rank in {compared}", color=INK_SECONDARY, fontsize=9.5)
    right.set_ylabel(f"Rank in {reference}", color=INK_SECONDARY, fontsize=9.5)
    right.set_title(
        "Paired queries, by rank in each graph",
        loc="left",
        color=INK,
        fontsize=10.5,
        fontweight="semibold",
        pad=32,
    )
    right.text(
        0,
        1.035,
        "Each cell: share of its row, and number of queries",
        transform=right.transAxes,
        color=INK_SECONDARY,
        fontsize=9,
    )

    fig.savefig(path, dpi=PLOT_DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_relations(report: pd.DataFrame, graphs: list[str], path: Path) -> None:
    """
    Plot one KG's `relation_report` rows (with `kg` and `graph` columns): one
    row of panels per model, MRR on the left and Hits@10 on the right, with
    one group of columns per relation and one column per graph trained with
    that model, colored in the order of `graphs`. A graph without a column
    for a relation doesn't have that relation.
    """
    kg = report["kg"].iat[0]
    models = sorted(report["model"].unique())
    relations = sorted(report["relation"].unique())
    k = max(HITS_AT)
    metrics = [("mrr", "MRR"), (f"hits_at_{k}", f"Hits@{k}")]
    colors = dict(zip(graphs, CATEGORICAL))

    # Heights in inches: the header, one row of panels per model, the bottom axis.
    header, row_height, footer = 1.6, 2.7, 0.5
    height = header + row_height * len(models) + footer
    fig, axes = plt.subplots(
        len(models), 2, figsize=(13, height), dpi=PLOT_DPI, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        top=1 - header / height,
        bottom=footer / height,
        wspace=0.12,
        hspace=0.5,
    )

    name = " / ".join(PurePosixPath(kg).parts)
    fig.text(
        0.06,
        1 - 0.3 / height,
        f"{name}: metrics per relation" if name else "Metrics per relation",
        color=INK,
        fontsize=13,
        fontweight="semibold",
        va="top",
    )
    fig.text(
        0.06,
        1 - 0.68 / height,
        "Head and tail predictions pooled, as in metrics_report.csv. "
        "A missing column: the relation is not in that graph.",
        color=INK_SECONDARY,
        fontsize=9.5,
        va="top",
    )
    fig.legend(
        handles=[Patch(facecolor=colors[graph], label=graph) for graph in graphs],
        loc="upper left",
        bbox_to_anchor=(0.06, 1 - 1.0 / height),
        ncols=len(graphs),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        handlelength=0.9,
        handleheight=0.9,
        borderaxespad=0,
        borderpad=0,
    )

    for row, model in zip(axes, models):
        runs = report[report["model"] == model]
        trained = [graph for graph in graphs if graph in set(runs["graph"])]
        for ax, (column, label) in zip(row, metrics):
            style_axes(ax)
            series = [
                (
                    runs[runs["graph"] == graph]
                    .set_index("relation")[column]
                    .reindex(relations)
                    .to_numpy(dtype=float),
                    colors[graph],
                )
                for graph in trained
            ]
            draw_grouped_columns(ax, relations, series, top=1.05)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
            ax.tick_params(axis="x", labelsize=8.5)
            ax.set_title(
                f"{model} · {label}",
                loc="left",
                color=INK,
                fontsize=10.5,
                fontweight="semibold",
                pad=8,
            )

    fig.savefig(path, dpi=PLOT_DPI, facecolor=SURFACE)
    plt.close(fig)


def delta_note(comparison: pd.DataFrame, graphs: list[str]) -> str:
    """
    One line with the Cliff's delta of every graph against the reference,
    from one relation's `compare_relations` rows, in the order of `graphs`,
    e.g. "δ vs source: skgg_std=1 −0.72, pygraft −0.95 n.s.", where n.s.
    marks a Holm-adjusted Mann-Whitney p of SIGNIFICANCE or more.
    """
    if comparison.empty:
        return ""
    rows = comparison.set_index("compared")
    entries = []
    for graph in graphs:
        if graph in rows.index:
            row = rows.loc[graph]
            delta = f"{row['cliffs_delta']:+.2f}".replace("-", "−")
            significant = row["mannwhitney_p_holm"] < SIGNIFICANCE
            entries.append(f"{graph} {delta}" + ("" if significant else " n.s."))
    return f"δ vs {comparison['reference'].iat[0]}: " + ", ".join(entries)


def plot_hits_curves(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    graphs: list[str],
    reference: str,
    path: Path,
) -> None:
    """
    Plot the Hits@k curves of one KG and model (`predictions` holds only its
    rows, with the `kg` and `graph` columns of `split_dataset`): one panel
    per relation and one line per graph, giving the share of predictions
    ranked within the top k for every k, so the height at k = 1 and k = 10
    is Hits@1 and Hits@10. Lines are colored in the order of `graphs`. Each
    panel lists the Cliff's deltas against the reference from `comparison`,
    the `compare_relations` rows of this KG and model.
    """
    kg, model = predictions["kg"].iat[0], predictions["model"].iat[0]
    relations = sorted(predictions["relation"].unique())
    colors = dict(zip(graphs, CATEGORICAL))
    trained = [graph for graph in graphs if graph in set(predictions["graph"])]
    k_max = predictions["candidates"].max()

    # Heights in inches: the header, one row per 4 relations, the bottom axis.
    columns = 4
    rows = -(-len(relations) // columns)
    header, row_height, footer = 2.0, 2.7, 0.6
    height = header + row_height * rows + footer
    fig, axes = plt.subplots(
        rows, columns, figsize=(13, height), dpi=PLOT_DPI, squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(
        left=0.06,
        right=0.98,
        top=1 - header / height,
        bottom=footer / height,
        wspace=0.18,
        hspace=0.45,
    )

    fig.text(
        0.06,
        1 - 0.3 / height,
        f"{' / '.join(PurePosixPath(kg).parts + (model,))}: Hits@k per relation",
        color=INK,
        fontsize=13,
        fontweight="semibold",
        va="top",
    )
    subtitle = (
        "Share of test predictions whose true answer ranks in the top k: "
        "the height at k = 1 and k = 10 is Hits@1 and Hits@10."
    )
    if not comparison.empty:
        subtitle += (
            f"\nδ: Cliff's delta against {reference} on candidate-normalized ranks, "
            "negative when the graph does worse; n.s.: Holm-adjusted Mann-Whitney "
            f"p ≥ {SIGNIFICANCE}."
        )
    fig.text(
        0.06,
        1 - 0.68 / height,
        subtitle,
        color=INK_SECONDARY,
        fontsize=9.5,
        va="top",
        linespacing=1.5,
    )
    fig.legend(
        handles=[
            Line2D([], [], color=colors[graph], linewidth=2 * PX, label=graph)
            for graph in trained
        ],
        loc="upper left",
        bbox_to_anchor=(0.06, 1 - 1.3 / height),
        ncols=len(trained),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        handlelength=1.6,
        borderaxespad=0,
        borderpad=0,
    )

    for i, ax in enumerate(axes.flat):
        if i >= len(relations):
            ax.set_visible(False)
            continue
        relation = relations[i]
        style_axes(ax)
        runs = predictions[predictions["relation"] == relation]
        # The reference graph last, so that its line is drawn on top.
        for graph in reversed(trained):
            ranks = runs.loc[runs["graph"] == graph, "rank"].to_numpy()
            if len(ranks) == 0:
                continue
            values, counts = np.unique(ranks, return_counts=True)
            shares = np.cumsum(counts) / len(ranks)
            if values[0] > 1:  # no answer ranked first, so the curve starts at 0
                values, shares = np.r_[1, values], np.r_[0, shares]
            ax.step(
                np.r_[values, k_max],
                np.r_[shares, 1],
                where="post",
                color=colors[graph],
                linewidth=2 * PX,
                solid_joinstyle="round",
                solid_capstyle="round",
            )
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xlim(1, k_max)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(color=GRIDLINE, linewidth=PX)
        ax.set_axisbelow(True)
        ax.spines["bottom"].set(visible=True, color=BASELINE, linewidth=PX)
        ax.set_title(
            relation,
            loc="left",
            color=INK,
            fontsize=10.5,
            fontweight="semibold",
            pad=20,
        )
        ax.text(
            0,
            1.03,
            delta_note(comparison[comparison["relation"] == relation], graphs),
            transform=ax.transAxes,
            color=INK_SECONDARY,
            fontsize=8,
            va="bottom",
        )
        if i % columns == 0:
            ax.set_ylabel("Share ranked ≤ k", color=INK_SECONDARY, fontsize=9.5)
        if i + columns >= len(relations):  # no panel below this one
            ax.set_xlabel("Rank threshold k", color=INK_SECONDARY, fontsize=9.5)

    fig.savefig(path, dpi=PLOT_DPI, facecolor=SURFACE)
    plt.close(fig)


def write_analysis(
    root: Path,
    rows: list[dict],
    analysis_dir: Path,
    reference: str,
    patterns: list[str],
) -> None:
    """Write the per-prediction analysis (see the module docstring) to `analysis_dir`."""
    predictions = build_predictions(root, rows)
    if predictions.empty:
        print("warning: no run could be ranked, so no prediction analysis was written")
        return

    runs = split_dataset(predictions)
    relation_comparison = compare_relations(runs, reference)
    paired = compare_queries(query_table(predictions), reference, patterns)
    tables = {
        "predictions.csv": predictions,
        "relation_report.csv": relation_report(predictions),
    }
    if relation_comparison.empty:
        print(f"note: no KG has a '{reference}' graph, so no relations were compared")
    else:
        tables["relation_comparison.csv"] = relation_comparison
    if paired.empty:
        print(
            f"note: no query is held out in both '{reference}' and a sibling dataset matching "
            f"{' '.join(patterns)} with the same model, so no graphs were compared"
        )
    else:
        tables["query_comparison.csv"] = paired
        tables["query_comparison_summary.csv"] = summarize_comparison(paired)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(analysis_dir / name, index=False)
        print(f"Wrote {len(table)} row(s) to {analysis_dir / name}")
    reports = split_dataset(tables["relation_report.csv"])
    for kg, kg_runs in runs.groupby("kg"):
        # One graph order per KG, so that each graph keeps its color in every plot.
        graphs = graph_order(kg_runs["graph"], reference, patterns)
        if len(graphs) > len(CATEGORICAL):
            print(
                f"warning: {kg} has {len(graphs)} graphs, the relation plots show "
                f"the first {len(CATEGORICAL)}"
            )
            graphs = graphs[: len(CATEGORICAL)]
        stem = "_".join(PurePosixPath(kg).parts + ("relations",))
        plot = analysis_dir / f"{stem}.png"
        report = reports[(reports["kg"] == kg) & reports["graph"].isin(graphs)]
        plot_relations(report, graphs, plot)
        print(f"Wrote {plot}")
        shown = kg_runs[kg_runs["graph"].isin(graphs)]
        for model, model_runs in shown.groupby("model"):
            stem = "_".join(PurePosixPath(kg).parts + (model, "hits_at_k"))
            plot = analysis_dir / f"{stem}.png"
            deltas = relation_comparison[
                (relation_comparison["kg"] == kg)
                & (relation_comparison["model"] == model)
            ]
            plot_hits_curves(model_runs, deltas, graphs, reference, plot)
            print(f"Wrote {plot}")
    for (kg, compared, model), group in paired.groupby(["kg", "compared", "model"]):
        plot = (
            analysis_dir
            / f"{'_'.join(PurePosixPath(kg).parts + (compared, model))}.png"
        )
        plot_comparison(predictions, group, plot)
        print(f"Wrote {plot}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect Hits@k / MRR from pykeen results.json files into one CSV, "
        "and analyse every test prediction behind them.",
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
    p.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="Directory for the per-prediction analysis (CSV files and plots)",
    )
    p.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Dataset folder whose predictions are compared with its siblings'",
    )
    p.add_argument(
        "--compare",
        nargs="+",
        default=DEFAULT_COMPARED,
        metavar="PATTERN",
        help="Sibling dataset folders compared with --reference query by query, as "
        "shell-style patterns (quote them): graphs that keep the reference's entity names",
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

    write_analysis(args.root, rows, args.analysis_dir, args.reference, args.compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
