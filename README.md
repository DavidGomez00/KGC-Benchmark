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

## 📈 Evaluation Metrics

We evaluate KG completion using embedding models:
- **TransE**, **TransH**, **TransD**
- **RotatE**, **ComplEx**, **TuckER**
- **CompGCN**

Metrics reported:
- Hits@1, Hits@3, Hits@5, Hits@10
- Mean Reciprocal Rank (MRR)

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
