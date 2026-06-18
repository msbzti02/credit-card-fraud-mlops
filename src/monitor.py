import os
import sys
import tempfile
import warnings

import matplotlib
import mlflow
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_config, load_processed_data
from src.evaluate import compute_metrics

warnings.filterwarnings("ignore")


def load_production_model(config: dict):

    model_name = config["mlflow"]["registry_model_name"]
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"[INFO] Loaded Production model: {model_name}")
    return model


def create_batches(X: pd.DataFrame, y: pd.Series, n_batches: int) -> list:

    batch_size = len(X) // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size if i < n_batches - 1 else len(X)
        batches.append((X.iloc[start:end], y.iloc[start:end]))
    return batches


def inject_drift(X_batch: pd.DataFrame, drift_magnitude: float = 5.0) -> pd.DataFrame:

    drifted = X_batch.copy()

    drift_cols = [c for c in drifted.columns if c.startswith("V")][:10]
    for col in drift_cols:
        drifted[col] = drifted[col] + drift_magnitude * drifted[col].std()
    return drifted


def run_evidently_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    save_path: str,
) -> dict:

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(save_path)

        result = report.as_dict()
        drift_metrics = result["metrics"][0]["result"]
        return {
            "share_drifted_features": drift_metrics.get("share_of_drifted_columns", 0),
            "dataset_drift": drift_metrics.get("dataset_drift", False),
        }
    except ImportError:
        print("[WARN] Evidently not installed — using fallback drift check.")
        return _fallback_drift_check(reference, current)
    except Exception as e:
        print(f"[WARN] Evidently error: {e} — using fallback drift check.")
        return _fallback_drift_check(reference, current)


def _fallback_drift_check(reference: pd.DataFrame, current: pd.DataFrame) -> dict:

    from scipy.stats import ks_2samp

    drifted_count = 0
    total = len(reference.columns)
    for col in reference.columns:
        stat, p_val = ks_2samp(reference[col].dropna(), current[col].dropna())
        if p_val < 0.05:
            drifted_count += 1

    share = drifted_count / total if total > 0 else 0
    return {
        "share_drifted_features": round(share, 4),
        "dataset_drift": share > 0.3,
    }


def plot_performance_over_time(batch_metrics: list, save_path: str) -> str:

    batches = list(range(1, len(batch_metrics) + 1))
    f1s = [m["f1_score"] for m in batch_metrics]
    aucs = [m.get("roc_auc", 0) for m in batch_metrics]
    precs = [m["precision"] for m in batch_metrics]
    recs = [m["recall"] for m in batch_metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batches, f1s, "o-", label="F1 Score", linewidth=2)
    ax.plot(batches, aucs, "s-", label="ROC AUC", linewidth=2)
    ax.plot(batches, precs, "^-", label="Precision", linewidth=2)
    ax.plot(batches, recs, "D-", label="Recall", linewidth=2)
    ax.set_xlabel("Batch Number", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Model Performance Over Time", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def main():

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    n_batches = config["monitoring"]["n_batches"]

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(config)
    model = load_production_model(config)

    ref_data = X_val.sample(n=min(5000, len(X_val)), random_state=42)

    batches = create_batches(X_test, y_test, n_batches)

    print(f"\n{'='*60}")
    print(f"  PERFORMANCE MONITORING - {n_batches} batches")
    print(f"{'='*60}\n")

    all_metrics = []

    with mlflow.start_run(run_name="Monitoring_Pipeline"):
        mlflow.set_tag("monitoring_type", "batch-simulation")
        mlflow.set_tag("n_batches", n_batches)

        with tempfile.TemporaryDirectory() as tmp:
            for i, (X_batch, y_batch) in enumerate(batches):
                batch_num = i + 1
                is_drift_batch = batch_num == n_batches

                print(
                    f"\n--- Batch {batch_num}/{n_batches} "
                    f"({'DRIFT INJECTED' if is_drift_batch else 'normal'}) ---"
                )

                X_current = inject_drift(X_batch) if is_drift_batch else X_batch

                with mlflow.start_run(nested=True, run_name=f"batch-{batch_num}"):
                    mlflow.set_tag("batch_number", batch_num)
                    mlflow.set_tag("drift_injected", is_drift_batch)

                    y_pred = model.predict(X_current)
                    try:
                        unwrapped = model._model_impl
                        if hasattr(unwrapped, "predict_proba"):
                            y_prob = unwrapped.predict_proba(X_current)[:, 1]
                        else:
                            y_prob = None
                    except Exception:
                        y_prob = None

                    metrics = compute_metrics(y_batch, y_pred, y_prob)
                    all_metrics.append(metrics)

                    for name, val in metrics.items():
                        mlflow.log_metric(name, val)
                    print(
                        f"  F1={metrics['f1_score']:.4f}  "
                        f"Precision={metrics['precision']:.4f}  "
                        f"Recall={metrics['recall']:.4f}"
                    )

                    drift_path = os.path.join(
                        tmp, f"drift_report_batch_{batch_num}.html"
                    )
                    drift_result = run_evidently_drift(ref_data, X_current, drift_path)

                    mlflow.log_metric(
                        "drift_share",
                        drift_result["share_drifted_features"],
                    )
                    mlflow.log_metric(
                        "dataset_drift",
                        int(drift_result["dataset_drift"]),
                    )

                    if os.path.exists(drift_path):
                        mlflow.log_artifact(drift_path)

                    drift_flag = (
                        "[!] DRIFT" if drift_result["dataset_drift"] else "[OK]"
                    )
                    print(
                        f"  Drift: {drift_result['share_drifted_features']:.2%} "
                        f"features drifted  [{drift_flag}]"
                    )

            perf_path = os.path.join(tmp, "performance_over_time.png")
            plot_performance_over_time(all_metrics, perf_path)
            mlflow.log_artifact(perf_path)

    print(f"\n{'='*60}")
    print("  MONITORING COMPLETE")
    print(f"  {n_batches} batches processed, drift reports logged to MLflow")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
