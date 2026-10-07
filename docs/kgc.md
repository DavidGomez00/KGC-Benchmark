# KG completion and evaluation

KGC-Benchmark trains knowledge graph embedding models with [PyKEEN](https://pykeen.readthedocs.io/) on one
or more graphs, reports standard link-prediction metrics, and compares the graphs prediction by prediction.

```
KGC-Benchmark/
├── KGC.py               Train and evaluate models, optionally after hyperparameter optimization
├── input.json           Configuration for KGC.py
└── report_results.py    Collect Hits@k and MRR into one CSV and analyse every test prediction
```

## Input

`KGC.py` reads a **tab-separated** `.tsv` file of triples: one triple per line, head, relation and tail separated by tabs.

It also reads **N-Triples** (`.nt`) files. When `kg_path` ends in `.nt`, KGC.py first converts the file to a `.tsv` with the same name in the same folder (`.data/kg/graph.nt` becomes `.data/kg/graph.tsv`), then runs as usual on the `.tsv`:

- IRIs are written without their angle brackets: `<http://example.org/spouse>` becomes `http://example.org/spouse`.
- Blank nodes (`_:b1`) and literals (`"42"^^<http://www.w3.org/2001/XMLSchema#integer>`, `"Anne"@en`) are written as they appear in the `.nt` file. A literal is an entity like any other, so remove the literal triples from the graph first if you don't want them in the training and test splits.
- Empty lines and comments are skipped. A line that isn't an N-Triples statement stops the run with the line number.
- If the `.tsv` already exists and is newer than the `.nt` file, KGC.py uses it without converting again. Otherwise, it **overwrites** the `.tsv`, so don't keep another graph under that name.

A graph in another RDF format, such as Turtle, must be converted to N-Triples or TSV first.

## KGC.py
```bash
python KGC.py
```

It reads `input.json` (see [configuration](configuration.md)) and:

1. Converts the graph to a `.tsv` if it's an `.nt` file (see [Input](#input)), and loads the `.tsv` into a PyKEEN `TriplesFactory`.
2. Splits it into training (80%) and testing (20%) sets.
3. If `save_splits` is true, writes `train` and `test` tab separated files to `results_path`.
4. For every model in `models`: trains it on the training split with PyKEEN's `pipeline` using the sLCWA training loop, the configured embedding size, batch size and epochs, and optionally filtered negative sampling.
5. Saves the pipeline results to `<results_path>/<model>/` and a `loss_plot.png` next to them.

Example configuration, training every model on the PyGraft version of the French royalty KG:

```json
{
  "kg_path": ".data/french_royalty/pygraft/french_royalty.pygraft.tsv",
  "results_path": "Output/french_royalty/pygraft",
  "models": ["TransE", "TransH", "TransD", "RotatE", "ComplEx", "TuckER", "CompGCN"],
  "batch_size": 32,
  "create_inverse_triples": true
}
```

Every key left out takes its default (see [configuration](configuration.md)).

### Hyperparameter optimization

With `"hpo": true`, KGC.py searches each model's hyperparameters before step 4, using PyKEEN's
`hpo_pipeline`:

1. Holds `validation_ratio` of the training split out as a validation set, once for all models. The `test`
   split is the same as in a run without HPO with the same `random_seed`, so both runs can be compared.
2. Runs `n_trials` trials per model on the rest of the training split, each scored by Hits@1 on the
   validation set. The trials search the embedding size, the batch size, the number of negatives per
   positive, PyKEEN's default ranges for the optimizer and loss, and model-specific ranges defined in
   `get_model_specific_params`. `embedding_dim` and `batch_size` from `input.json` are ignored.
3. Saves the study to `<results_path>/<model>/hpo/`: `study.json`, `trials.tsv` and
   `best_pipeline/pipeline_config.json`.
4. Trains the model with the best trial's hyperparameters on the **whole** training split and saves it like a
   run without HPO (step 5), so `report_results.py` reads it too.

## Comparing graphs

To compare graphs, for example a KG and synthetic graphs generated from it, run KGC.py once per graph with the same models and the same `random_seed`. Give each graph its own `results_path` under a common folder (see [Folder layout](#folder-layout)), then run `report_results.py`.

The models used so far are TransE, TransH, TransD, RotatE, ComplEx, TuckER and CompGCN. Other PyKEEN models can be listed in `models` too, but haven't been tried.

> A graph normalized by [VANILLA](https://github.com/SDM-TIB/VANILLA) gives each triple a predicate specific to its object, so it has many more distinct relations than the original. Keep this in mind when comparing models and reading their memory use.

## report_results.py

`report_results.py` gathers the runs of every graph into one place, in two steps:

1. **The metrics table.** It writes Hits@1 to Hits@15, MRR and count, from the `both` / `realistic` slice (see [Evaluation metrics](#evaluation-metrics)), to `Output/metrics_report.csv`. There is one row per dataset and model. MRR, count and Hits@1, 3, 5 and 10 come from the run's `results.json`. PyKEEN doesn't store the other values of k, so they are counted from the ranks of the per-prediction analysis, and stay empty for a run it leaves out.
2. **The per-prediction analysis.** It recomputes the rank of every test prediction behind that table, and compares the graphs relation by relation and query by query. The results go to `Output/prediction_analysis/`. See [Per-prediction analysis](#per-prediction-analysis).

```bash
python report_results.py                                         # source vs every skgg* folder
python report_results.py --root Output --output Output/metrics_report.csv
python report_results.py --reference source --compare "skgg.std=*"
```

### Folder layout

The script finds every `results.json` under `--root` and expects the layout KGC.py writes:

```
<root>/<dataset...>/<model>/results.json
```

`<dataset...>` is the `results_path` of the run, relative to `--root`. It can be more than one folder, and the `dataset` column reports it with `/`, e.g. `french_royalty/skgg.std=1.filled`. The analysis splits it in two:

| Part | Example | Meaning |
|---|---|---|
| `kg` | `french_royalty` | The parent folders: the KG that the graphs come from. |
| `graph` | `skgg.std=1.filled` | The last folder: one graph of that KG. |

Graphs are only compared with graphs of the same `kg`. Give each graph of a KG its own `results_path` under a common parent folder, for example:

```
Output/french_royalty/
├── source/              the reference graph (--reference)
├── skgg.std=1.filled/   compared query by query (matches --compare "skgg*")
└── pygraft/             compared relation by relation only
```

A folder whose `results.json` can't be read, or lacks the `both` / `realistic` slice, is skipped with a warning. The rest of the report still runs.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--root` | `Output` | Results folder to scan. |
| `--output`, `-o` | `Output/metrics_report.csv` | Path of the metrics table. |
| `--analysis-dir` | `Output/prediction_analysis` | Folder of the per-prediction analysis (CSV files and plots). |
| `--reference` | `source` | Graph folder that every other graph of its KG is compared with. |
| `--compare` | `skgg*` | Graph folders compared with the reference query by query: graphs that keep the reference's entity names. Shell-style patterns, several allowed. Quote them so the shell does not expand them. |

The defaults are relative to the repository root, wherever the script is run from. Paths passed on the command line are relative to the current directory.

Each run overwrites the files in `--analysis-dir`, but it does not delete files from earlier runs. A plot of a graph or model that is no longer under `--root` stays, and so does a comparison CSV that the new run did not write (see the `note:` messages in [troubleshooting](troubleshooting.md#report_resultspy)). Empty the folder to start clean.

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
| Hits@k (k = 1 to 15) | `hits_at_1` to `hits_at_15` | 0 to 1 | Higher | Fraction of test predictions where the true entity is among the top *k* candidates. Hits@1 is the fraction where the model's single best guess is correct. Hits@10 = 0.40 means the right answer is in the top 10 for 40% of predictions. Hits@k never decreases as k grows. `results.json` only holds k = 1, 3, 5 and 10. |
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

The expected score of a random model depends on how many candidate entities each query has. For example, a random guess is in the top 10 far more often among 500 entities than among 5,000. Hits@k, MRR and MR are therefore not on the same scale for two graphs with different numbers of entities, such as a KG and a synthetic graph generated from it. The adjusted metrics (AMRI, adjusted MRR, adjusted Hits@10) subtract out that random baseline. Report them alongside Hits@k and MRR when comparing graphs of different sizes.

## Per-prediction analysis

After writing the table, [`report_results.py`](#report_resultspy) looks at every test prediction behind it. The results go to `Output/prediction_analysis/` (`--analysis-dir`).

### Recomputing the ranks

`results.json` stores only aggregates. For every run, the script reloads three things:

- `trained_model.pkl`
- the id mappings and training triples in `training_triples/`
- the `test` file KGC.py wrote to `results_path`, so `save_splits` must be `true`

It then ranks both predictions of every test triple the way PyKEEN's evaluation does: filtered, with `realistic` ties. The ranks of a run therefore average to its row in `metrics_report.csv`.

KGC.py writes `train` and `test` before it trains the models. While it is retraining a `results_path`, the `test` file on disk does not belong to the models still in that folder. A run whose recomputed MRR differs from its `results.json` by more than 0.001 is therefore left out of the analysis with a warning.

### Pairing predictions between graphs

The comparison answers one question: **when a prediction is accurate in one graph, is it also accurate in the other?** It compares a reference graph (`--reference`, default `source`) with every sibling folder that matches `--compare` (default `skgg*`, e.g. `skgg.std=1.filled`). Only models trained on both graphs are compared.

These graphs share no test triples, even when one is generated from the other. SKGG keeps each entity's relations but rewires their targets: `Marie_Antoinette` keeps her 3 `child` triples, but with different children. Each graph is also split on its own. Predictions are therefore paired by **query**:

- a tail query `(h, r, ?)`, such as `(Marie_Antoinette, child, ?)`
- a head query `(?, r, t)`, such as `(?, child, Marie_Antoinette)`

A query pairs when it has at least one answer in the `test` file of both graphs. Each graph ranks its own held-out answers. A query is **accurate at k** in a graph when at least half of its held-out answers rank in the top k. Most queries have a single answer, so this is simply a hit or a miss.

PyGraft names its entities `E1`, `E2`, ..., so none of its queries pair with the source graph. Its runs still appear in `predictions.csv` and `relation_report.csv`.

### Output files

| File | One row per | Contents |
|---|---|---|
| `predictions.csv` | test prediction of every run | `side` (the entity predicted, `head` or `tail`), the test triple, `rank`, `candidates` (the entities it was ranked among, after filtering), `reciprocal_rank`, and `hits_at_1` to `hits_at_15` (1 or 0) |
| `relation_report.csv` | dataset, model and relation | The columns of `metrics_report.csv`, computed per relation |
| `relation_comparison.csv` | KG, compared graph, model and relation | Each relation in the reference graph against every other graph, see [Comparing relations across graphs](#comparing-relations-across-graphs) |
| `query_comparison.csv` | query paired between two graphs | The `query`, then for each graph (`reference_` and `compared_` columns): its number of held-out `answers`, their `mrr`, and the share of them in the top k (`hits_at_k`, for k = 1, 3, 5 and 10) |
| `query_comparison_summary.csv` | comparison, model, group of queries and k | See the table below |
| `<kg>_relations.png` | KG | `relation_report.csv` as a chart: one row of panels per model, MRR and Hits@10 per relation, and one column per graph. Each graph keeps its color in every plot: the reference first, then the compared graphs, then the rest (e.g. source, skgg.std=1.filled, pygraft) |
| `<kg>_<model>_hits_at_k.png` | KG and model | One panel per relation and one line per graph: the share of test predictions whose true answer ranks in the top k, for every k (its height at k = 1 and k = 10 is Hits@1 and Hits@10). Each panel lists the Cliff's delta of every graph against the reference, marked `n.s.` when not significant |
| `<kg>_models_hits_at_1-15.png` | KG | One panel per graph and one line per model: Hits@1 to Hits@15, to compare how fast each model's accuracy grows with k. Each model keeps its color in every plot of this kind |
| `<kg>_<graph>_relations_hits_at_1-15.png` | KG and graph | The same lines, in a first panel over all relations (the graph's panel above) and then one panel per relation. Each panel gives its number of test predictions per model, since a rare relation rests on few of them |
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

### Comparing relations across graphs

`relation_comparison.csv` asks, for each relation and model: **are the ranks in another graph distributed differently from the reference graph's?** It differs from the paired-query comparison in two ways:

- It uses every test prediction of the relation in each graph, not only the queries held out in both.
- It needs no shared entities, so it compares the reference with every other graph of the KG, PyGraft included. A relation is compared when both graphs have it; `type` exists only in PyGraft's graph, so it is not compared.

The graphs have different numbers of entities (PyGraft 1,089, source 2,038), so a raw rank of 500 is a random guess in one and above random in the other (see [Comparing metrics across graphs](#comparing-metrics-across-graphs)). The distribution tests therefore use the **normalized rank** `(rank − 1) / (candidates − 1)`: 0 means ranked first, 1 means ranked last, and a random guess averages 0.5. PyKEEN's AMRI (`adjusted_arithmetic_mean_rank_index`) is 1 − 2 × their mean, when every query has the same number of candidates.

| Column | Meaning |
|---|---|
| `reference_predictions`, `compared_predictions` | Number of test predictions of the relation in each graph. |
| `reference_mrr`, `compared_mrr`, `mrr_difference` | MRR of the relation in each graph, and compared minus reference. |
| `mrr_difference_low`, `mrr_difference_high` | 95% bootstrap interval of `mrr_difference` (2,000 resamples of each graph's predictions, fixed seed). An interval without 0 means the MRR difference is unlikely to be chance. |
| `reference_hits_at_10`, `compared_hits_at_10` | Hits@10 of the relation in each graph. |
| `cliffs_delta` | The chance that a prediction of the compared graph ranks better than one of the reference graph, minus the reverse, on normalized ranks. From −1 (the compared graph always ranks worse) to +1 (always better); 0 means neither tends to rank better. |
| `effect` | The size of `cliffs_delta`: negligible below 0.147, small below 0.33, medium below 0.474, large above (Romano et al., 2006). |
| `mannwhitney_p` | Mann-Whitney U test: does one graph tend to rank better than the other? |
| `ks_statistic`, `ks_p` | Kolmogorov-Smirnov test: do the two rank distributions differ in any way, including in shape? The statistic is the largest gap between their Hits@k curves (on normalized ranks). |
| `mannwhitney_p_holm`, `ks_p_holm` | The p-values after a Holm correction over the relations of the same graph pair and model. Use these to decide significance. |

The bootstrap interval and the tests can disagree. The interval is about the mean of the reciprocal ranks, which only the top of the ranking moves. The tests weigh every prediction, wherever it ranks. A relation can therefore gain MRR in a few top-ranked predictions while its distribution as a whole does not change significantly.

## Requirements

`torch`, `pykeen`, `pandas`, `numpy`, `scipy` and `matplotlib`. See [Quick start](../README.md#quick-start).
