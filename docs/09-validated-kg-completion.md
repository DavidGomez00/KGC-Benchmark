# 09 - Validated KG completion

`Completion/` measures what normalization does for link prediction. It trains knowledge graph
embedding models with [PyKEEN](https://pykeen.readthedocs.io/) on a graph (typically the normalized KG from
part 1) and reports standard link-prediction metrics.

```
Completion/
├── KGC.py               Train and evaluate models with fixed hyperparameters
├── input_KGC.json       Configuration for KGC.py
├── KGC_hpo.py           Train and evaluate models with hyperparameter optimization
├── input_KGC_hpo.json   Configuration for KGC_hpo.py
└── report_results.py    Collect Hits@k and MRR into one CSV and analyse every test prediction
```

## Input

Both scripts (`KGC.py` and `KGC_hpo.py`) read a **tab-separated** `.tsv` file of triples. The normalization pipeline writes N-Triples, so the normalized `.nt` must be converted to TSV first, e.g., with `utils/nt_to_tsv.py` ([07](07-tsv-to-nt.md)). Some benchmarks in the repository already have a `.tsv` beside the `.nt` (e.g., `Output/SGKG/transformed/SGKG_normalized.tsv`).

## KGC.py
```bash
cd Completion
python KGC.py
```

It reads `input_KGC.json` (see [03](03-configuration.md#validated_kg_completioninput_kgcjson)) and:

1. Loads the `.tsv` into a PyKEEN `TriplesFactory`.
2. Splits it into training and testing sets.
3. If `save_splits` is true, writes `train` and `test` tab separated files to `results_path`.
4. For every model in `models`: trains it with PyKEEN's `pipeline` using the sLCWA training loop, the configured embedding size, batch size and epochs, and optionally filtered negative sampling.
5. Saves the pipeline results to `<results_path>/<model>/` and a `loss_plot.png` next to them.

Example configuration:

```json
{
  "kg_path": "data/YAGO3-10/TransformedKG/TransformedKG_YAGO3-10.tsv",
  "results_path": "data/YAGO3-10/TransformedKG",
  "models": ["TuckER"],
  "num_epochs": 100,
  "embedding_dim": 50,
  "batch_size": 32,
  "random_seed": 1235,
  "create_inverse_triples": false,
  "filtered_negative_sampling": true,
  "save_splits": true,
  "log_level": "INFO"
}
```

## KGC_hpo.py
```bash
cd Completion
python KGC_hpo.py
```

Reads `input_KGC_hpo.json`. Uses PyKEEN's `hpo_pipeline` with `n_trials` trials per model, with
model-specific search ranges defined in `get_model_specific_params`. Results are saved to
`<output_dir>/<model>/`.

## Comparing the effect of normalization

To measure the impact of the pipeline, run the same script with the same seed on two graphs:

| Run | `kg_path` |
|---|---|
| Baseline | The original graph converted to TSV |
| Normalized | The `<rdf_stem>.normalized` graph converted to TSV |

and compare the metrics saved by PyKEEN.

The supported models named in the project are TransE, TransH, TransD, RotatE, ComplEx, TuckER and CompGCN.

> Predicate-object expansion (see [05](05-normalization-pipeline.md#31-predicate-object-expansion)) gives each triple a predicate specific to its object, so the normalized graph has many more distinct relations than the original. Keep this in mind when comparing models and reading their memory use.

## Evaluation metrics

### How a test triple is ranked

Every metric is a summary of **ranks**. For each test triple `(h, r, t)`, PyKEEN makes two predictions:

- **Tail prediction** `(h, r, ?)`: score `(h, r, e)` for every entity `e` in the graph, sort the candidates from best to worst score, and record the position of the true tail `t`. Rank 1 means the model put the correct answer first.
- **Head prediction** `(?, r, t)`: the same, scoring `(e, r, t)` for every entity `e` and recording the position of `h`.

The evaluation is **filtered**, which is PyKEEN's default and the standard in the literature. Before ranking, every candidate that forms a triple already known to be true (in the training set or the test set) is removed, except the one being tested. Without this, a query such as `(Louis_XIV, child, ?)`, which has several correct answers, would penalize the model for ranking another real child above the one being tested.

### Where the numbers are stored

Each `<results_path>/<model>/results.json` stores the metrics as `metrics[<side>][<ties>][<metric>]`:

| Level | Values | Meaning |
|---|---|---|
| `<side>` | `head`, `tail`, `both` | Ranks from head prediction, tail prediction, or both pooled together. |
| `<ties>` | `optimistic`, `pessimistic`, `realistic` | How ties are ranked when other candidates get exactly the same score as the true answer: optimistic puts the true answer first among them, pessimistic puts it last, and realistic uses the average of the two. A large gap between optimistic and pessimistic means the model gives many entities the same score. |

`report_results.py` reads the `both` / `realistic` slice, which is the one to report.

### Metrics collected by `report_results.py`

| Metric | Key in `results.json` | Range | Better | Meaning |
|---|---|---|---|---|
| Hits@k (k = 1, 3, 5, 10) | `hits_at_1`, `hits_at_3`, `hits_at_5`, `hits_at_10` | 0 to 1 | Higher | Fraction of test predictions where the true entity is among the top *k* candidates. Hits@1 is the fraction where the model's single best guess is correct. Hits@10 = 0.40 means the right answer is in the top 10 for 40% of predictions. |
| Mean Reciprocal Rank (MRR) | `inverse_harmonic_mean_rank` | 0 to 1 | Higher | Mean of `1 / rank` over all predictions. Rank 1 adds 1, rank 2 adds 0.5, rank 10 adds 0.1, and rank 1000 adds almost nothing. It rewards answers at or near the top, and it barely changes whether a missed answer sits at rank 200 or rank 2000. |
| count | `count` | Up to 2 × test triples | n/a | Number of ranks the metrics average over. In the `both` slice this is two per test triple (one head prediction and one tail prediction). |

### Other metrics in `results.json`

PyKEEN stores further rank statistics that `report_results.py` does not collect. They can be read directly from `results.json`.

| Metric | Key | Better | Meaning |
|---|---|---|---|
| Mean Rank (MR) | `arithmetic_mean_rank` | Lower | Average rank, from 1 up to the number of entities. Unlike MRR, a few very bad ranks can pull it up a lot. |
| Geometric / harmonic mean rank | `geometric_mean_rank`, `harmonic_mean_rank` | Lower | Other averages of the ranks. The harmonic mean rank is `1 / MRR`. |
| Median rank | `median_rank` | Lower | Rank of the middle prediction. Half the predictions rank better than this. |
| Inverse ranks | `inverse_arithmetic_mean_rank`, `inverse_geometric_mean_rank`, `inverse_median_rank` | Higher | `1 /` the corresponding rank statistic. |
| Spread | `standard_deviation`, `variance`, `median_absolute_deviation` | n/a | How much the ranks vary across predictions. |
| Adjusted Mean Rank (AMR) | `adjusted_arithmetic_mean_rank` | Lower | MR divided by the MR expected from random scoring. 1 = random, near 0 = perfect, range 0 to 2. |
| Adjusted Mean Rank Index (AMRI) | `adjusted_arithmetic_mean_rank_index` | Higher | MR rescaled so that 1 = perfect, 0 = random and −1 = worst possible. `adjusted_geometric_mean_rank_index` does the same for the geometric mean rank. |
| Adjusted MRR | `adjusted_inverse_harmonic_mean_rank` | Higher | MRR rescaled so that 1 = perfect and 0 = random. Negative values are worse than random. |
| Adjusted Hits@10 | `adjusted_hits_at_k` | Higher | Hits@10 rescaled the same way: 1 = perfect, 0 = random. |
| z-scores | `z_arithmetic_mean_rank`, `z_geometric_mean_rank`, `z_inverse_harmonic_mean_rank`, `z_hits_at_k` | Higher | How many standard deviations the result lies above random scoring. They grow with test-set size, so they show how confident one can be that the model beats random, not how good it is. |

### Comparing metrics across graphs

The expected score of a random model depends on how many candidate entities each query has. For example, a random guess is in the top 10 far more often among 500 entities than among 5,000. Hits@k, MRR and MR are therefore not on the same scale for two graphs with different numbers of entities, such as a baseline graph and its normalized version. The adjusted metrics (AMRI, adjusted MRR, adjusted Hits@10) subtract out that random baseline. Report them alongside Hits@k and MRR when comparing graphs of different sizes.

## Per-prediction analysis

After writing the table, `report_results.py` looks at every test prediction behind it. The results go to `Output/prediction_analysis/` (`--analysis-dir`).

```bash
cd Completion
python report_results.py                                         # source vs every skgg* folder
python report_results.py --reference source --compare "skgg_std=*"
```

### Recomputing the ranks

`results.json` stores only aggregates. For every run, the script reloads three things:

- `trained_model.pkl`
- the id mappings and training triples in `training_triples/`
- the `test` file KGC.py wrote to `results_path`, so `save_splits` must be `true`

It then ranks both predictions of every test triple the way PyKEEN's evaluation does: filtered, with `realistic` ties. The ranks of a run therefore average to its row in `metrics_report.csv`.

KGC.py writes `train` and `test` before it trains the models. While it is retraining a `results_path`, the `test` file on disk does not belong to the models still in that folder. A run whose recomputed MRR differs from its `results.json` by more than 0.001 is therefore left out of the analysis with a warning.

### Pairing predictions between graphs

The comparison answers one question: **when a prediction is accurate in one graph, is it also accurate in the other?** It compares a reference graph (`--reference`, default `source`) with every sibling folder that matches `--compare` (default `skgg*`, e.g. `skgg_std=1`). Only models trained on both graphs are compared.

These graphs share no test triples, even when one is generated from the other. SKGG keeps each entity's relations but rewires their targets: `Marie_Antoinette` keeps her 3 `child` triples, but with different children. Each graph is also split on its own. Predictions are therefore paired by **query**:

- a tail query `(h, r, ?)`, such as `(Marie_Antoinette, child, ?)`
- a head query `(?, r, t)`, such as `(?, child, Marie_Antoinette)`

A query pairs when it has at least one answer in the `test` file of both graphs. Each graph ranks its own held-out answers. A query is **accurate at k** in a graph when at least half of its held-out answers rank in the top k. Most queries have a single answer, so this is simply a hit or a miss.

PyGraft names its entities `E1`, `E2`, ..., so none of its queries pair with the source graph. Its runs still appear in `predictions.csv` and `relation_report.csv`.

### Output files

| File | One row per | Contents |
|---|---|---|
| `predictions.csv` | test prediction of every run | `side` (the entity predicted, `head` or `tail`), the test triple, `rank`, `reciprocal_rank`, and `hits_at_1` to `hits_at_10` (1 or 0) |
| `relation_report.csv` | dataset, model and relation | The columns of `metrics_report.csv`, computed per relation |
| `query_comparison.csv` | query paired between two graphs | The `query`, then for each graph (`reference_` and `compared_` columns): its number of held-out `answers`, their `mrr`, and the share of them in the top k (`hits_at_k`) |
| `query_comparison_summary.csv` | comparison, model, group of queries and k | See the table below |
| `<kg>_relations.png` | KG | `relation_report.csv` as a chart: one row of panels per model, MRR and Hits@10 per relation, and one column per graph. Each graph keeps its color in every plot: the reference first, then the compared graphs, then the rest (e.g. source, skgg_std=1, pygraft) |
| `<kg>_<compared>_<model>.png` | comparison and model | Left: how the ranks of all test predictions of each graph spread over rank buckets. Right: for the paired queries, how each rank bucket of the reference graph spreads over the buckets of the compared graph |

`query_comparison_summary.csv` has one row per group of queries and per k (1, 3, 5 and 10). There are three kinds of group:

- all paired queries (`relation` = `all`, `side` = `both`)
- the head queries and the tail queries (`side` = `head` or `tail`)
- the queries of each relation

| Column | Meaning |
|---|---|
| `paired_queries` | Number of queries in the group. |
| `reference_mrr`, `compared_mrr` | Mean per-query MRR in each graph. |
| `spearman_rho`, `spearman_p` | Rank correlation between the two graphs' per-query MRR, and its p-value. Near 0, how hard a query is in one graph says nothing about how hard it is in the other. Empty below 3 queries, or when one graph gives every query the same value. |
| `both_accurate`, `only_reference`, `only_compared`, `neither` | Queries accurate at k in both graphs, in only one of them, or in neither. |
| `compared_accurate_when_reference_accurate` | Share of the queries accurate in the reference graph that are also accurate in the compared graph. |
| `compared_accurate_when_reference_inaccurate` | The same share among the queries the reference graph gets wrong. The gap between these two shares shows how much accuracy in one graph predicts accuracy in the other. |
| `cohen_kappa` | Agreement on accurate versus inaccurate, corrected for chance. 1 means the graphs always agree, 0 means no better than chance, and a negative value means worse than chance. |

## Requirements

`torch`, `pykeen`, `pandas`, `numpy`, `scipy` and `matplotlib`. See [02](02-installation.md#extra-packages-for-part-2).
