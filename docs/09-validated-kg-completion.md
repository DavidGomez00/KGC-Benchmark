# 09 - Validated KG completion

`Validated_KG_Completion/` measures what normalization does for link prediction. It trains knowledge graph
embedding models with [PyKEEN](https://pykeen.readthedocs.io/) on a graph (typically the normalized KG from
part 1) and reports standard link-prediction metrics.

```
Validated_KG_Completion/
├── KGC.py               Train and evaluate models with fixed hyperparameters
├── input_KGC.json       Configuration for KGC.py
├── KGC_hpo.py           Train and evaluate models with hyperparameter optimization
└── input_KGC_hpo.json   Configuration for KGC_hpo.py
```

## Input

Both scripts read a **tab-separated** file of triples (`subject<TAB>predicate<TAB>object`, no header).
The normalization pipeline writes N-Triples, so the normalized `.nt` must be converted to TSV first. Some
benchmarks in the repository already have a `.tsv` beside the `.nt` (for example `Transformed_SGKG4/`).

## KGC.py - fixed hyperparameters

```bash
cd Validated_KG_Completion
python KGC.py
```

It reads `input_KGC.json` (see [03](03-configuration.md#validated_kg_completioninput_kgcjson)) and:

1. Loads the TSV into a PyKEEN `TriplesFactory`.
2. Splits it into training and testing sets with `tf.split(random_state=random_seed)`.
3. If `save_splits` is true, writes `train` and `test` files (tab-separated) to `results_path`.
4. For every model in `models`: trains it with PyKEEN's `pipeline` using the sLCWA training loop, the
   configured embedding size, batch size and epochs, and optionally filtered negative sampling.
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

## KGC_hpo.py - hyperparameter optimization

```bash
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
| Normalized | The `TransformedKG_<KG>` graph converted to TSV |

and compare the metrics saved by PyKEEN.

Metrics reported by PyKEEN's evaluator include Hits@1, Hits@3, Hits@5, Hits@10 and Mean Reciprocal Rank
(MRR). The supported models named in the project are TransE, TransH, TransD, RotatE, ComplEx, TuckER and
CompGCN.

> Predicate-object expansion (see [05](05-normalization-pipeline.md#31-predicate-object-expansion)) gives
> each triple a predicate specific to its object, so the normalized graph has many more distinct relations
> than the original. Keep this in mind when comparing models and reading their memory use.

## Requirements

`torch`, `pykeen`, `pandas`, `numpy` and `matplotlib`. See [02](02-installation.md#extra-packages-for-part-2).
