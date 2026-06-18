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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_config, load_processed_data
from src.evaluate import compute_metrics, log_artifacts_to_mlflow, log_metrics_to_mlflow

warnings.filterwarnings("ignore")

COST_FALSE_NEGATIVE = 500
COST_FALSE_POSITIVE = 10


def compute_cost_at_threshold(y_true, y_prob, threshold):

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE

    return {
        "threshold": threshold,
        "total_cost": total_cost,
        "fn_cost": fn * COST_FALSE_NEGATIVE,
        "fp_cost": fp * COST_FALSE_POSITIVE,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def find_optimal_threshold(y_true, y_prob, n_thresholds=200):

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    results = [compute_cost_at_threshold(y_true, y_prob, t) for t in thresholds]
    df = pd.DataFrame(results)

    optimal_idx = df["total_cost"].idxmin()
    optimal = df.iloc[optimal_idx]

    return optimal["threshold"], df


def plot_cost_curve(results_df, optimal_threshold, save_path):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(
        results_df["threshold"],
        results_df["total_cost"],
        "b-",
        linewidth=2,
        label="Total Cost",
    )
    ax1.plot(
        results_df["threshold"],
        results_df["fn_cost"],
        "r--",
        linewidth=1,
        alpha=0.7,
        label="Missed Fraud Cost ($500/FN)",
    )
    ax1.plot(
        results_df["threshold"],
        results_df["fp_cost"],
        "g--",
        linewidth=1,
        alpha=0.7,
        label="False Alarm Cost ($10/FP)",
    )
    ax1.axvline(
        x=optimal_threshold,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Optimal: {optimal_threshold:.3f}",
    )
    ax1.axvline(
        x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Default: 0.5"
    )
    ax1.set_xlabel("Classification Threshold", fontsize=12)
    ax1.set_ylabel("Total Business Cost ($)", fontsize=12)
    ax1.set_title("Business Cost vs. Threshold", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(
        results_df["threshold"], results_df["f1"], "b-", linewidth=2, label="F1 Score"
    )
    ax2.plot(
        results_df["threshold"],
        results_df["precision"],
        "g-",
        linewidth=1.5,
        label="Precision",
    )
    ax2.plot(
        results_df["threshold"],
        results_df["recall"],
        "r-",
        linewidth=1.5,
        label="Recall",
    )
    ax2.axvline(
        x=optimal_threshold,
        color="red",
        linestyle=":",
        linewidth=2,
        label=f"Optimal: {optimal_threshold:.3f}",
    )
    ax2.set_xlabel("Classification Threshold", fontsize=12)
    ax2.set_ylabel("Metric Value", fontsize=12)
    ax2.set_title("Metrics vs. Threshold", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Cost curve saved to {save_path}")
    return save_path


def main():

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(config)

    model_name = config["mlflow"]["registry_model_name"]
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    print(f"[INFO] Loaded Production model: {model_name}")

    try:

        if hasattr(model, "_model_impl") and hasattr(model._model_impl, "python_model"):
            unwrapped = model._model_impl.python_model
        elif hasattr(model, "_model_impl"):
            unwrapped = model._model_impl
        else:
            unwrapped = model

        if hasattr(unwrapped, "predict_proba"):
            y_prob = unwrapped.predict_proba(X_test)[:, 1]
        elif (
            hasattr(unwrapped, "predict") and "xgboost" in str(type(unwrapped)).lower()
        ):

            y_prob = unwrapped.predict(X_test)
        else:
            print("[WARN] Model does not support predict_proba. Exiting.")
            return
    except Exception as e:
        print(f"[WARN] Cannot get probabilities: {e}")
        return

    print(f"\n{'='*60}")
    print(f"  THRESHOLD OPTIMIZATION")
    print(f"  Cost model: FN=${COST_FALSE_NEGATIVE}, FP=${COST_FALSE_POSITIVE}")
    print(f"{'='*60}\n")

    optimal_threshold, results_df = find_optimal_threshold(y_test, y_prob)

    default_result = compute_cost_at_threshold(y_test, y_prob, 0.5)
    optimal_result = compute_cost_at_threshold(y_test, y_prob, optimal_threshold)

    print(f"  Default Threshold (0.50):")
    print(
        f"    F1={default_result['f1']:.4f}  "
        f"Cost=${default_result['total_cost']:,.0f}  "
        f"FN={default_result['fn']}  FP={default_result['fp']}"
    )

    print(f"\n  Optimal Threshold ({optimal_threshold:.3f}):")
    print(
        f"    F1={optimal_result['f1']:.4f}  "
        f"Cost=${optimal_result['total_cost']:,.0f}  "
        f"FN={optimal_result['fn']}  FP={optimal_result['fp']}"
    )

    savings = default_result["total_cost"] - optimal_result["total_cost"]
    print(
        f"\n  Cost Savings: ${savings:,.0f} "
        f"({savings/max(default_result['total_cost'],1)*100:.1f}%)"
    )

    with mlflow.start_run(run_name="Threshold_Optimization"):
        mlflow.set_tag("analysis_type", "threshold_optimization")
        mlflow.set_tag("cost_fn", f"FN=${COST_FALSE_NEGATIVE}")
        mlflow.set_tag("cost_fp", f"FP=${COST_FALSE_POSITIVE}")

        mlflow.log_param("default_threshold", 0.5)
        mlflow.log_param("optimal_threshold", round(optimal_threshold, 4))
        mlflow.log_param("cost_false_negative", COST_FALSE_NEGATIVE)
        mlflow.log_param("cost_false_positive", COST_FALSE_POSITIVE)

        mlflow.log_metric("default_cost", default_result["total_cost"])
        mlflow.log_metric("optimal_cost", optimal_result["total_cost"])
        mlflow.log_metric("cost_savings", savings)
        mlflow.log_metric("optimal_f1", optimal_result["f1"])
        mlflow.log_metric("optimal_precision", optimal_result["precision"])
        mlflow.log_metric("optimal_recall", optimal_result["recall"])

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = []
            cost_path = plot_cost_curve(
                results_df,
                optimal_threshold,
                os.path.join(tmp, "cost_vs_threshold.png"),
            )
            artifacts.append(cost_path)

            csv_path = os.path.join(tmp, "threshold_analysis.csv")
            results_df.to_csv(csv_path, index=False)
            artifacts.append(csv_path)

            log_artifacts_to_mlflow(artifacts)

    print(f"\n{'='*60}")
    print(f"  THRESHOLD OPTIMIZATION COMPLETE")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Cost savings: ${savings:,.0f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
