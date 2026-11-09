"""
文件说明：训练严格 clinical_no_p4205 的结构化对比基线。

脚本按冻结的 eid 切分训练 LR、RF、XGBoost、LightGBM；所有预处理只在训练集拟合，
并输出与 SAGM paper_main_table.csv 兼容的测试集指标和逐样本预测。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "models") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "models"))

from p4205_feature_policy import feature_policy_table
from train_sagm_net import load_frozen_split_indices
from train_two_tower_classification import (
    DEFAULT_DATA,
    DEFAULT_P4205_CACHE,
    EID_COLUMN,
    binary_metrics,
    build_structured_frame,
    build_tabular_preprocessor,
    load_data,
    apply_cohort_filter,
    select_threshold,
)


MODEL_CHOICES = ("lr", "rf", "xgboost", "lightgbm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="严格结构化临床特征对比基线")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--p4205-cache", type=Path, default=DEFAULT_P4205_CACHE)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="与冻结 split 对应的 seed 标识")
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--target-years", type=float, default=10.0)
    parser.add_argument("--cohort", default="all_p4205_valid")
    parser.add_argument("--max-rows", type=int, default=None, help="仅用于 smoke test")
    return parser.parse_args()


def positive_weight(y: np.ndarray) -> float:
    return float((y == 0).sum() / max(1, (y == 1).sum()))


def make_model(name: str, y_train: np.ndarray, seed: int):
    """固定、可审计的基线超参数；不在验证或测试集上调参。"""
    weight = positive_weight(y_train)
    if name == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=20.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=6.0,
            scale_pos_weight=weight,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=120,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=6.0,
            scale_pos_weight=weight,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"未知模型：{name}")


def save_predictions(path: Path, eids: np.ndarray, y: np.ndarray, probability: np.ndarray, threshold: float) -> None:
    pd.DataFrame(
        {
            EID_COLUMN: eids,
            "all_cause_true": y.astype(np.int8),
            "all_cause_mask": np.ones(len(y), dtype=np.int8),
            "all_cause_probability": probability,
            "prediction": (probability >= threshold).astype(np.int8),
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{args.output_dir}")
    train_seed = int(args.train_seed) if args.train_seed is not None else int(args.seed)
    frame = load_data(args.data, args.target_years, args.max_rows)
    frame = apply_cohort_filter(frame, args.cohort, args.p4205_cache)
    split_indices = load_frozen_split_indices(frame, args.split_assignments)
    train_idx, val_idx, test_idx = (split_indices["train"], split_indices["validation"], split_indices["test"])
    y = frame["target"].to_numpy(dtype=np.int8)

    # 中文说明：该函数按现有策略移除所有 P4205 直接和派生临床列。
    structured = build_structured_frame(frame, feature_set="clinical_no_p4205")
    preprocessor, numeric_columns, categorical_columns = build_tabular_preprocessor(structured.iloc[train_idx])
    x_train = preprocessor.fit_transform(structured.iloc[train_idx]).astype(np.float32, copy=False)
    x_val = preprocessor.transform(structured.iloc[val_idx]).astype(np.float32, copy=False)
    x_test = preprocessor.transform(structured.iloc[test_idx]).astype(np.float32, copy=False)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    feature_policy_table(structured.columns).to_csv(args.output_dir / "structured_feature_policy.csv", index=False, encoding="utf-8-sig")
    config = {
        "model_family": "structured_comparison_baselines",
        "task_mode": "binary",
        "target_years": args.target_years,
        "cohort": args.cohort,
        "structured_feature_set": "clinical_no_p4205",
        "seed": int(args.seed),
        "train_seed": train_seed,
        "split_source": str(args.split_assignments.resolve()),
        "row_count": int(len(frame)),
        "positive_count_all_cause": int(y.sum()),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "structured_raw_feature_count": int(structured.shape[1]),
        "structured_encoded_dim": int(x_train.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "models": args.models,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.read_csv(args.split_assignments).to_csv(args.output_dir / "split_assignments.csv", index=False, encoding="utf-8-sig")

    rows = []
    result_payload: dict[str, object] = {}
    for index, name in enumerate(args.models, start=1):
        model_seed = train_seed + index * 1009
        print(f"[structured] {index}/{len(args.models)} start model={name} seed={model_seed}", flush=True)
        model = make_model(name, y[train_idx], model_seed)
        model_name = f"structured_{name}"
        model_dir = args.output_dir / model_name
        model_dir.mkdir()
        from tree_diagnostics import fit_with_history
        fit_with_history(model,name,x_train,y[train_idx],x_val,y[val_idx],model_dir)
        import joblib
        joblib.dump(preprocessor, model_dir/'preprocessor.joblib')
        val_probability = model.predict_proba(x_val)[:, 1]
        test_probability = model.predict_proba(x_test)[:, 1]
        threshold, _ = select_threshold(y[val_idx], val_probability)
        metrics = binary_metrics(y[test_idx], test_probability, threshold)
        model_name = f"structured_{name}"
        model_dir = args.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        save_predictions(model_dir / "test_predictions.csv", frame[EID_COLUMN].iloc[test_idx].to_numpy(), y[test_idx], test_probability, threshold)
        save_predictions(model_dir / "validation_predictions.csv", frame[EID_COLUMN].iloc[val_idx].to_numpy(), y[val_idx], val_probability, threshold)
        payload = {"model_name": model_name, "test": metrics, "validation_threshold": threshold, "model_seed": model_seed}
        (model_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result_payload[model_name] = payload
        rows.append(
            {
                "model_name": model_name,
                "wave_encoder": "none",
                "split": "test",
                "task": "all_cause",
                "n": int(len(test_idx)),
                "positive_count": int(y[test_idx].sum()),
                "positive_rate": float(y[test_idx].mean()),
                **metrics,
            }
        )
        print(f"[structured] {index}/{len(args.models)} done model={name} roc={metrics['roc_auc']:.4f} pr={metrics['pr_auc']:.4f}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "paper_main_table.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "run_results.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[structured] 完成：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
