# KGC-Benchmark

Train knowledge graph embedding models on one or more graphs with [PyKEEN](https://pykeen.readthedocs.io/),
and compare how well each graph supports link prediction: overall, per relation, and query by query.

```
KGC-Benchmark/
├── KGC.py               Train and evaluate models, optionally after hyperparameter optimization
├── input.json           Configuration for KGC.py
├── report_results.py    Collect Hits@k and MRR into one CSV and analyse every test prediction
└── docs/                Documentation
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU speeds up training, but is not required.

## Usage

All commands run from the repository root.

1. **Configure** `input.json`: the graph to train on (`kg_path`, a tab-separated file of triples), where to
   save the results (`results_path`), the models and their hyperparameters.

   ```json
   {
     "kg_path": ".data/french_royalty/french_royalty.tsv",
     "results_path": "Output/french_royalty/source",
     "models": ["TransE", "TuckER", "CompGCN"],
     "num_epochs": 100,
     "embedding_dim": 50,
     "batch_size": 32,
     "random_seed": 1235,
     "create_inverse_triples": false,
     "filtered_negative_sampling": true,
     "save_splits": true,
     "log_level": "INFO",
     "hpo": false,
     "n_trials": 30,
     "validation_ratio": 0.1
   }
   ```

   Set `"hpo": true` to search each model's hyperparameters (`n_trials` trials, scored on a validation set
   holding `validation_ratio` of the training triples) and train it with the best ones.

2. **Train and evaluate** the models:

   ```bash
   python KGC.py
   ```

   Repeat for each graph, with its own `results_path` under a common folder (e.g.
   `Output/french_royalty/source`, `Output/french_royalty/skgg.std=1.filled`).

3. **Collect and analyse** the results:

   ```bash
   python report_results.py
   ```

   It writes `Output/metrics_report.csv` (Hits@1/3/5/10 and MRR, one row per graph and model) and
   `Output/prediction_analysis/`: the rank of every test prediction, the metrics per relation, and a
   comparison of a reference graph (`--reference`, default `source`) with the other graphs of its KG,
   relation by relation and query by query (`--compare`, default `skgg*`), with plots. The graphs are
   compared with a bootstrap interval of the MRR difference, Cliff's delta, Mann-Whitney and
   Kolmogorov-Smirnov tests, and, for queries held out in both graphs, Spearman's rank correlation and
   Cohen's kappa.

## Documentation

| Page | Contents |
|---|---|
| [KG completion and evaluation](docs/kgc.md) | What KGC.py and report_results.py do, the metrics, the output files and the statistics |
| [Configuration](docs/configuration.md) | Every key of `input.json` |
| [Troubleshooting](docs/troubleshooting.md) | The messages the scripts print, and what to do about them |

## Origin and license

This project started as the KG completion part of [VANILLA](https://github.com/SDM-TIB/VANILLA), developed by
the Scientific Data Management Group at TIB by Disha Purohit and Yashrajsinh Chudasama, supervised by
Maria-Esther Vidal. Its history was carried over from that repository. It is distributed under the MIT
license in [LICENSE.txt](LICENSE.txt).
