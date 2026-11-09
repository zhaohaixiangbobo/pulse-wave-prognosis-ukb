# Reproducibility scope

- Main sources: training_regularization_20261107/main_v2/code and main_v2/n9/n9_control.py.
- Final tree sources: trees_v1, whose shared training/preprocessing modules were checked against the packaged main_v2 copies and matched byte-for-byte before path normalization.
- Main archived arguments are extracted into configs/n9_arguments.json; lr is set to the final 2e-4 setting and only the full model is enabled.
- Exact labels: exact_date_v2_cached/code/exact_labels.py.
- Source modules preserve algorithms; machine-specific default paths are normalized. Portable wrappers remove dependencies on private COMPLETE.json, historical signature gates and local experiment ledgers.
- The main wrapper explicitly uses the archived numpy_2_4_1 waveform percentile compatibility mode.
- Tests include the original N9 CPU controller test, syntax/import checks, argument preview and synthetic outcome/metric checks. No full-cohort training was rerun for this release, so cross-machine bitwise reproduction has not been established.
- Secondary tasks and optional architectures exist in the dependency sources but do not have dedicated frozen public orchestration in this core release. This is not a complete reproduction bundle for every table/figure in the paper.
- Full bootstrap/figure archives remain separate. The lightweight evaluator implements raw AP/ROC and validation-calibrated Brier/threshold metrics; it does not implement all manuscript confidence intervals or subgroup analyses.

Training may take hours across five seeds depending on CPU preprocessing, GPU, storage and early stopping. Full runs should be launched by the researcher after checking their local authorized inputs.
