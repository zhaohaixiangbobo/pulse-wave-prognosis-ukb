# SAGM-Net: structured-anchored pulse-wave prediction

Core implementation accompanying the manuscript revision: a structured encoder, an InceptionTime-lite waveform encoder, and an engineered-feature encoder with scalar gated residual fusion.

## Release scope

This package includes exact-date outcome construction, waveform/engineered-feature preprocessing, the full model, the revised plateau-triggered training schedule (N9), final structured tree comparators, and prediction evaluation utilities. It contains **no participant-level data, IDs, split assignments, predictions, or trained weights**. Authorized researchers must supply their own data.

The main model starts with AdamW lr=0.0002, weight decay=0.01, dropout=0.2, batch size=512, maximum 80 epochs. Global early stopping uses validation mean(AP, ROC-AUC), patience=10, min_delta=0.0005. After three plateau epochs, the following epoch halves lr, multiplies weight decay by four and increases enabled dropout by 0.1 (cap 0.5), at most three times. All towers remain trainable. The final RF/XGBoost/LightGBM settings are applied by `run_trees.py`; boosting round selection uses validation scores on a 25-round grid.

## Files

- `run_main.py`: portable entry point for the final full model; defaults to argument preview.
- `run_trees.py`: final RF T1, XGBoost T1, LightGBM T2 entry point.
- `prepare_inputs.py`: exact-date labels and advanced-feature cache from an already selected authorized source cohort.
- `evaluate_predictions.py`: validation-fitted Platt scaling, test AP/ROC, Brier, Top-5%, thresholds and net benefit.
- `src/`: archived model and preprocessing implementations, including dependencies. Legacy optional model branches are retained for import compatibility; they are not all the final manuscript configuration.
- `configs/n9_arguments.json`: explicit final main-task arguments, without private paths.
- `docs/DATA_SCHEMA.md`: required input schema and outcome rules.
- `docs/REPRODUCIBILITY.md`: source mapping, environment and verification limits.
- `SOURCE_MANIFEST.json`: source and release hashes. Public wrappers replace local orchestration, not model equations.

## Environment and running

Use Python 3.12 and the packages listed in `requirements.txt`. Install a PyTorch build appropriate to your machine before using the package. `docs/ENVIRONMENT.json` records the packaging/smoke-check environment, including the locally installed CUDA build; it is not a claim that every historical experiment used that environment. No dependencies are installed automatically.

With your Python 3.12 interpreter, invoke:

```text
run_main.py --data data/cohort.csv --p4205-cache data/p4205_advanced_features.csv --split-assignments data/splits/seed42.csv --output-dir outputs/main/seed42 --seed 42
```

This previews the frozen configuration. Add `--execute` to train; add `--evaluate-test` to produce held-out test predictions after checkpoint selection. Use a different empty output directory for each seed. Original split seeds were 42, 1998, 2026, 2027 and 3407. Recreating fresh random splits does not reproduce the original frozen participant assignments.

```text
run_trees.py --model xgboost --data data/cohort.csv --p4205-cache data/p4205_advanced_features.csv --split-assignments data/splits/seed42.csv --output-dir outputs/xgboost/seed42 --seed 42 --execute --evaluate-test
evaluate_predictions.py --validation outputs/xgboost/seed42/validation_predictions.csv --test outputs/xgboost/seed42/test_predictions.csv --output outputs/xgboost/seed42/calibrated_metrics.json
```

For the neural entry point, prediction files are under the `sagm_gated_inception_lite` model subdirectory. Core training exports may also contain uncalibrated metrics; use the explicit evaluation step for the calibration/threshold definitions documented here. The lightweight evaluator does not replace the complete manuscript bootstrap and figure-production archive.

## Data availability and licensing

UK Biobank data access must be obtained independently through its authorized process. Participant-level data and derived individual-level outputs are not distributed here. Public code alone cannot reproduce the exact study estimates without the authorized frozen cohort and split assignments.

A reuse license has not yet been selected by the authors for this prepared release. Public visibility should not be interpreted as an unrestricted reuse license. The authors should select a license and add the final paper citation and repository URL when preparing the public release. No DOI is claimed.
