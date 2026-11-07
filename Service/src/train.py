"""
Training script and artifact export pipeline for CTR prediction in RTB auctions.
Produces serialized prediction model, preprocessor, and metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.preprocessor import CTRDataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def compute_target_ctr_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_ctrs: List[float] = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10],
) -> List[Dict[str, Any]]:
    """
    Simulates bidding strategies to reach target CTR KPIs.
    When we only bid on impressions where pCTR >= threshold, what realized CTR
    and impression volume (percentage of won bids) do we get?

    Returns said thresholds
    """
    results = []
    # Sort by predicted probability descending
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]

    cum_clicks = np.cumsum(y_true_sorted)
    cum_impressions = np.arange(1, len(y_true) + 1)
    realized_ctrs = cum_clicks / cum_impressions

    for target_ctr in target_ctrs:
        # Find index where realized CTR is >= target_ctr
        valid_indices = np.nonzero(realized_ctrs >= target_ctr)[0]
        if len(valid_indices) > 0:
            last_idx = valid_indices[-1]
            threshold = float(y_prob_sorted[last_idx])
            volume_pct = float((last_idx + 1) / len(y_true) * 100.0)
            actual_ctr = float(realized_ctrs[last_idx])
        else:
            threshold = float(y_prob_sorted[0])
            volume_pct = float(1.0 / len(y_true) * 100.0)
            actual_ctr = float(realized_ctrs[0])

        results.append(
            {
                "target_ctr": target_ctr,
                "bid_threshold": round(threshold, 6),
                "realized_ctr": round(actual_ctr, 6),
                "won_impressions_pct": round(volume_pct, 2),
            }
        )

    return results


def benchmark_latency(
    preprocessor: CTRDataPreprocessor,
    model: xgb.XGBClassifier,
    sample_record: Dict[str, Any],
    n_runs: int = 1000,
) -> Dict[str, float]:
    """
    Measures micro-benchmarks for single-request inference (preprocessor + model).
    """
    logger.info(f"Benchmarking single-request latency over {n_runs} iterations...")

    # Warmup
    for _ in range(50):
        vec = preprocessor.transform_single(sample_record)
        _ = model.predict_proba(vec)

    transform_times = []
    predict_times = []
    total_times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        vec = preprocessor.transform_single(sample_record)
        t1 = time.perf_counter()
        _ = model.predict_proba(vec)[0, 1]
        t2 = time.perf_counter()

        transform_times.append((t1 - t0) * 1000.0)
        predict_times.append((t2 - t1) * 1000.0)
        total_times.append((t2 - t0) * 1000.0)

    return {
        "transform_p50_ms": round(float(np.percentile(transform_times, 50)), 3),
        "transform_p95_ms": round(float(np.percentile(transform_times, 95)), 3),
        "transform_p99_ms": round(float(np.percentile(transform_times, 99)), 3),
        "predict_p50_ms": round(float(np.percentile(predict_times, 50)), 3),
        "predict_p95_ms": round(float(np.percentile(predict_times, 95)), 3),
        "predict_p99_ms": round(float(np.percentile(predict_times, 99)), 3),
        "total_p50_ms": round(float(np.percentile(total_times, 50)), 3),
        "total_p95_ms": round(float(np.percentile(total_times, 95)), 3),
        "total_p99_ms": round(float(np.percentile(total_times, 99)), 3),
    }


def train_pipeline(
    data_path: str,
    output_dir: str,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.123366,
    reg_lambda: float = 0.574781,
    reg_alpha: float = 5.31857e-4,
    gamma: float = 3.391157e-7,
    grow_policy: str = "lossguide",
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    smoothing: float = 5.0,
    use_oof: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int = 4,
) -> Dict[str, Any]:
    """
    Full model training pipeline: data loading, preprocessing, training and evaluation.

    Default model hyperparameters were taken from the research notebook where they were optimized.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw dataset from {data_path}...")
    t_load_start = time.time()
    df = pd.read_csv(data_path, sep="\t", index_col=0)
    logger.info(
        f"Loaded {len(df)} rows in {time.time() - t_load_start:.2f} s. Target mean CTR: {df['is_clicked'].mean():.4%}"
    )

    # Train/Test split
    logger.info(f"Splitting dataset (test_size={test_size}, shuffle=False)...")
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        shuffle=False,
        random_state=random_state,
    )
    y_train = df_train["is_clicked"].to_numpy(dtype=int)
    y_test = df_test["is_clicked"].to_numpy(dtype=int)

    # Preprocessing
    logger.info("Initializing and fitting CTRDataPreprocessor...")
    preprocessor = CTRDataPreprocessor(
        smoothing=smoothing,
        use_oof=use_oof,
        random_state=random_state,
    )

    t_prep_start = time.time()
    X_train_mat = preprocessor.fit_transform(df_train, y_train)
    t_prep_end = time.time()
    logger.info(
        f"Preprocessor fit_transform done in {t_prep_end - t_prep_start:.2f} s. Shape: {X_train_mat.shape}"
    )

    logger.info("Transforming test dataset...")
    X_test_mat = preprocessor.transform(df_test)
    logger.info(f"Test transform done. Shape: {X_test_mat.shape}")

    # XGBoost Training
    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        gamma=gamma,
        grow_policy=grow_policy,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        device="cpu",
        objective="binary:logistic",
        eval_metric=["auc", "logloss"],
        random_state=random_state,
        n_jobs=n_jobs,
    )

    t_train_start = time.time()
    model.fit(
        X_train_mat,
        y_train,
        eval_set=[(X_train_mat, y_train), (X_test_mat, y_test)],
        verbose=False,
    )
    t_train_end = time.time()
    logger.info(f"Model training completed in {t_train_end - t_train_start:.2f} s.")

    # Predictions & Metric Evaluations
    logger.info("Evaluating predictions on test set...")
    y_test_proba = model.predict_proba(X_test_mat)[:, 1]

    roc_auc = float(roc_auc_score(y_test, y_test_proba))
    pr_auc = float(average_precision_score(y_test, y_test_proba))
    loss = float(log_loss(y_test, y_test_proba))

    # Threshold analysis for F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * (precisions * recalls) / (precisions + recalls),
        0.0,
    )
    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    prec_at_best_f1 = float(precisions[best_idx])
    rec_at_best_f1 = float(recalls[best_idx])

    logger.info("=" * 60)
    logger.info("EVALUATION METRICS:")
    logger.info(f"  Test ROC-AUC:   {roc_auc:.4f}")
    logger.info(f"  Test PR-AUC:    {pr_auc:.4f}")
    logger.info(f"  Test Log-Loss:  {loss:.4f}")
    logger.info(f"  Best F1-Score:  {best_f1:.4f} (at threshold {best_threshold:.4f})")
    logger.info(f"  Precision @ F1: {prec_at_best_f1:.4f}")
    logger.info(f"  Recall @ F1:    {rec_at_best_f1:.4f}")
    logger.info("=" * 60)

    # Target CTR KPI analysis
    kpi_table = compute_target_ctr_thresholds(y_test, y_test_proba)
    logger.info("TARGET CTR KPI SIMULATION:")
    for row in kpi_table:
        logger.info(
            f"  Target CTR: {row['target_ctr']:.1%} -> Bid Threshold: {row['bid_threshold']:.4f} "
            f"| Realized CTR: {row['realized_ctr']:.2%} | Won Impressions: {row['won_impressions_pct']:.1f}%"
        )

    # Latency benchmarking
    sample_record = df_test.iloc[0].to_dict()
    latency_stats = benchmark_latency(preprocessor, model, sample_record, n_runs=1000)
    logger.info("LATENCY BENCHMARKS (1000 single requests):")
    logger.info(
        f"  Transform Latency:  p50={latency_stats['transform_p50_ms']} ms, p95={latency_stats['transform_p95_ms']} ms, p99={latency_stats['transform_p99_ms']} ms"
    )
    logger.info(
        f"  Predict Latency:    p50={latency_stats['predict_p50_ms']} ms, p95={latency_stats['predict_p95_ms']} ms, p99={latency_stats['predict_p99_ms']} ms"
    )
    logger.info(
        f"  Total Single RTB:   p50={latency_stats['total_p50_ms']} ms, p95={latency_stats['total_p95_ms']} ms, p99={latency_stats['total_p99_ms']} ms"
    )

    # Save artifacts
    logger.info(f"Saving artifacts to {output_path}...")
    preprocessor_file = output_path / "preprocessor.joblib"
    model_joblib_file = output_path / "model.joblib"
    model_json_file = output_path / "model.json"
    metadata_file = output_path / "metadata.json"

    joblib.dump(preprocessor, preprocessor_file)
    joblib.dump(model, model_joblib_file)
    model.save_model(str(model_json_file))

    metadata = {
        "model_name": "CTR_XGBoost_Classifier",
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "algorithm": "xgboost.XGBClassifier",
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "gamma": gamma,
            "grow_policy": grow_policy,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "tree_method": "hist",
            "device": "cpu",
            "smoothing": smoothing,
            "use_oof": use_oof,
        },
        "dataset_stats": {
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "total_rows": int(len(df)),
            "feature_count": int(len(preprocessor.feature_names_)),
            "train_ctr": float(y_train.mean()),
            "test_ctr": float(y_test.mean()),
        },
        "metrics": {
            "roc_auc": round(roc_auc, 5),
            "pr_auc": round(pr_auc, 5),
            "log_loss": round(loss, 5),
            "best_f1": round(best_f1, 5),
            "best_f1_threshold": round(best_threshold, 5),
            "precision_at_best_f1": round(prec_at_best_f1, 5),
            "recall_at_best_f1": round(rec_at_best_f1, 5),
            "default_bid_threshold": round(best_threshold, 5),
        },
        "kpi_simulation": kpi_table,
        "latency_benchmarks_ms": latency_stats,
        "features": preprocessor.feature_names_,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("All artifacts saved successfully!")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and serialize CTR prediction pipeline."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Research/ctr_task_dataset.csv",
        help="Path to input TSV dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Service/artifacts",
        help="Directory to save artifacts",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200, help="Number of boosting rounds"
    )
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth")
    parser.add_argument(
        "--learning-rate", type=float, default=0.123366, help="Learning rate"
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.574781,
        help="L2 regularization term on weights",
    )
    parser.add_argument(
        "--reg_alpha",
        type=float,
        default=5.31857e-4,
        help="L1 regularization term on weights",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=3.391157e-7,
        help="Minimum loss reduction required to make a further partition on a leaf node of the tree.",
    )
    parser.add_argument(
        "--grow_policy",
        type=str,
        default="lossguide",
        help="Tree growing policy:\n  'depthwise': Favors splitting at nodes closest to the node\n  'lossguide': Favors splitting at nodes with highest loss change.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=5.0,
        help="Target encoding smoothing parameter m",
    )
    parser.add_argument(
        "--use-oof",
        action="store_true",
        help="Use Out-Of-Fold target encoding on training data",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction for test set"
    )
    parser.add_argument("--random-state", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--n-jobs", type=int, default=4, help="Number of parallel threads"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        grow_policy=args.grow_policy,
        smoothing=args.smoothing,
        use_oof=args.use_oof,
        test_size=args.test_size,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
