import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)
        metrics["log_loss"] = log_loss(y_true, y_prob)
    return metrics


def log_metrics_to_mlflow(metrics: dict) -> None:

    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    print(f"[INFO] Logged {len(metrics)} metrics to MLflow")


def plot_confusion_matrix(
    y_true, y_pred, save_path: str = "confusion_matrix.png"
) -> str:

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to {save_path}")
    return save_path


def plot_roc_curve(y_true, y_prob, save_path: str = "roc_curve.png") -> str:

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#3366CC", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] ROC curve saved to {save_path}")
    return save_path


def plot_precision_recall_curve(y_true, y_prob, save_path: str = "pr_curve.png") -> str:

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall, precision, color="#DC3912", lw=2, label=f"PR Curve (AUC = {pr_auc:.4f})"
    )
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] PR curve saved to {save_path}")
    return save_path


def plot_feature_importance(
    model, feature_names, top_n: int = 15, save_path: str = "feature_importance.png"
) -> str:

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color="#109618",
        edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Feature importance saved to {save_path}")
    return save_path


def save_classification_report(
    y_true, y_pred, save_path: str = "classification_report.txt"
) -> str:

    report = classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"])
    with open(save_path, "w") as f:
        f.write(report)
    print(f"[INFO] Classification report saved to {save_path}")
    return save_path


def log_artifacts_to_mlflow(artifact_paths: list) -> None:

    for path in artifact_paths:
        if os.path.exists(path):
            mlflow.log_artifact(path)
    print(f"[INFO] Logged {len(artifact_paths)} artifacts to MLflow")
