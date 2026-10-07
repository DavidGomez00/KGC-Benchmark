## VANILLA: Validated KG Completion

![Validated KG Completion](https://raw.githubusercontent.com/SDM-TIB/VANILLA/main/images/DesignPattern(b)VANILLA.png "Validated KG Completion")
## 🔍 Overview

The design pattern components of VANILLA for Validated KG Completion process. <br>
VANILLA utilizes the Normalized and Validate KG to show the impact of KG normalization on the
downstream task of KG completion using link prediction. <br>

## 🚀 Running the Pipeline of Validates KG Completion

1. **Configure input**
   Modify `input.json` to select the KG, the models and their hyperparameters.
```json
{
  "kg_path": "path_to_your_dataset/TransformedKG_YAGO3-10.tsv",
  "results_path": "path_to_your_dataset/TransformedKG",
  "models": ["TuckER"],
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

2. **Executing KG Completion**

```python
python KGC.py
```

3. **Collecting and analysing the results**

```python
python report_results.py
```

   It reads the `results.json` of every run under `Output/<dataset...>/<model>/` and writes:
   - `Output/metrics_report.csv`: Hits@k and MRR, one row per dataset and model.
   - `Output/prediction_analysis/`: the rank of every test prediction, the metrics per relation, and a
     comparison of a reference graph (`--reference`, default `source`) with the other graphs of its KG,
     relation by relation and query by query (`--compare`, default `skgg*`), with plots.

   The analysis needs the `test` split, so keep `"save_splits": true`. Give each graph its own
   `results_path` under a common folder, e.g. `Output/french_royalty/source` and
   `Output/french_royalty/skgg.std=1.filled`. Options, outputs and statistics are described in
   [docs/09](../docs/09-validated-kg-completion.md#report_resultspy).

## 📈 Evaluation Metrics

We evaluate KG completion using embedding models:
- **TransE**, **TransH**, **TransD**
- **RotatE**, **ComplEx**, **TuckER**
- **CompGCN**

Metrics reported:
- Hits@1, Hits@3, Hits@5, Hits@10
- Mean Reciprocal Rank (MRR)

`report_results.py` also reports them per relation, and compares graphs with a bootstrap interval of the MRR
difference, Cliff's delta, Mann-Whitney and Kolmogorov-Smirnov tests, and, for queries held out in both
graphs, Spearman's rank correlation and Cohen's kappa.

---

## 🧠 Graphical Summary

The VANILLA framework integrates **symbolic rules**, **domain constraints**, and **neural embeddings** for high-quality knowledge graph completion. It identifies valid and invalid triples using evolving logical constraints and employs numerical models to infer missing links, ensuring semantic consistency and logical soundness in the normalized KG.

---

## 📄 License

This project is licensed under the terms of the [LICENSE.txt](LICENSE.txt).

## Authors
VANILLA has been developed by members of the Scientific Data Management Group at TIB, as an ongoing research effort.
The development is co-ordinated and supervised by Maria-Esther Vidal.
We strongly encourage you to report any issues you have with VANILLA.
Please, use the GitHub issue tracker to do so.
VANILLA has been implemented in joint work by Disha Purohit, and Yashrajsinh Chudasama.
