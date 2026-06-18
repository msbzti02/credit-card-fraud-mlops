import os
import sys
import tempfile
import warnings

import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (
    load_config,
    load_data,
    load_processed_data,
    preprocess_data,
)
from src.evaluate import (
    compute_metrics,
    log_artifacts_to_mlflow,
    log_metrics_to_mlflow,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_classification_report,
)

warnings.filterwarnings("ignore")


def check_drift_status(config):

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(config["mlflow"]["experiment_name"])

    monitoring_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.monitoring_type = 'batch-simulation'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not monitoring_runs:
        print("[INFO] No monitoring runs found. Assuming no drift.")
        return {"drift_detected": False, "reason": "no monitoring data"}

    run = monitoring_runs[0]
    run_id = run.info.run_id

    child_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
        order_by=["start_time DESC"],
    )

    drift_count = 0
    degraded_count = 0
    total_batches = len(child_runs)

    for child in child_runs:
        metrics = child.data.metrics
        if metrics.get("share_drifted_features", 0) > 0.25:
            drift_count += 1
        if metrics.get("f1_score", 1.0) < 0.5:
            degraded_count += 1

    drift_detected = drift_count > 0 or degraded_count > 0

    return {
        "drift_detected": drift_detected,
        "drift_batches": drift_count,
        "degraded_batches": degraded_count,
        "total_batches": total_batches,
        "reason": (
            f"{drift_count} drifted + {degraded_count} degraded batches"
            if drift_detected
            else "all batches healthy"
        ),
    }


def get_production_metrics(config):

    client = mlflow.tracking.MlflowClient()
    model_name = config["mlflow"]["registry_model_name"]

    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        prod_version = versions[0]
        run = client.get_run(prod_version.run_id)
        return run.data.metrics
    except Exception as e:
        print(f"[WARN] Cannot get production metrics: {e}")
        return None


def retrain_model(config, X_train, y_train, X_test, y_test, feature_names):

    xgb_cfg = config["models"]["xgboost"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=xgb_cfg["max_depth"],
        learning_rate=0.05,
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=xgb_cfg["scale_pos_weight"],
        random_state=42,
        eval_metric="aucpr",
        use_label_encoder=False,
    )

    with mlflow.start_run(run_name="Auto_Retrain_XGBoost"):
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("trigger", "automatic_retraining")
        mlflow.set_tag("reason", "drift_detected")
        mlflow.set_tag("author", config["project"]["author"])

        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("max_depth", xgb_cfg["max_depth"])
        mlflow.log_param("training_samples", len(X_train))

        print("[INFO] Retraining XGBoost with optimized parameters...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        log_metrics_to_mlflow(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = [
                plot_confusion_matrix(
                    y_test, y_pred, os.path.join(tmp, "confusion_matrix.png")
                ),
                plot_roc_curve(y_test, y_prob, os.path.join(tmp, "roc_curve.png")),
                plot_precision_recall_curve(
                    y_test, y_prob, os.path.join(tmp, "pr_curve.png")
                ),
                save_classification_report(
                    y_test, y_pred, os.path.join(tmp, "classification_report.txt")
                ),
            ]
            log_artifacts_to_mlflow(artifacts)

        mlflow.xgboost.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

    return run_id, metrics


def promote_if_better(config, new_run_id, new_metrics, prod_metrics):

    client = mlflow.tracking.MlflowClient()
    model_name = config["mlflow"]["registry_model_name"]

    new_f1 = new_metrics.get("f1_score", 0)
    prod_f1 = prod_metrics.get("f1_score", 0) if prod_metrics else 0

    new_auc = new_metrics.get("roc_auc", 0)
    prod_auc = prod_metrics.get("roc_auc", 0) if prod_metrics else 0

    print(f"\n  Model Comparison:")
    print(f"    Current Production:  F1={prod_f1:.4f}  AUC={prod_auc:.4f}")
    print(f"    Retrained Model:     F1={new_f1:.4f}  AUC={new_auc:.4f}")

    if new_f1 > prod_f1 or new_auc > prod_auc:
        print(f"\n  [OK] New model is BETTER -> Promoting to Production")

        model_uri = f"runs:/{new_run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )

        client.update_model_version(
            name=model_name,
            version=mv.version,
            description=(
                f"Version {mv.version} -- Auto-retrained after drift detection. "
                f"F1={new_f1:.4f}, AUC={new_auc:.4f}"
            ),
        )

        print(f"  [OK] Model v{mv.version} promoted to Production")
        return True
    else:
        print(f"\n  [INFO] New model is NOT better -> Keeping current Production")
        return False


def main():

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    print(f"\n{'='*60}")
    print(f"  AUTOMATED RETRAINING PIPELINE")
    print(f"{'='*60}\n")

    print("[STEP 1] Checking drift status...")
    drift_status = check_drift_status(config)
    print(f"  Drift detected: {drift_status['drift_detected']}")
    print(f"  Reason: {drift_status['reason']}")

    if not drift_status["drift_detected"]:
        print("\n  [OK] No drift detected. Retraining not needed.")
        print(f"\n  To force retraining anyway, run monitoring first with")
        print(f"  drift injection, then re-run this script.\n")

        print("  [INFO] Proceeding with retraining for demonstration...\n")

    print("[STEP 2] Getting current production model metrics...")
    prod_metrics = get_production_metrics(config)
    if prod_metrics:
        print(f"  Production F1:  {prod_metrics.get('f1_score', 'N/A')}")
        print(f"  Production AUC: {prod_metrics.get('roc_auc', 'N/A')}")
    else:
        print("  No production model found.")

    print("\n[STEP 3] Loading data and retraining...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(config)
    except FileNotFoundError:
        df = load_data(config, use_augmented=True)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
            df, config, apply_smote=True, save=True, apply_feature_engineering=True
        )

    feature_names = list(X_train.columns)
    new_run_id, new_metrics = retrain_model(
        config, X_train, y_train, X_test, y_test, feature_names
    )

    print(f"\n  Retrained model metrics:")
    for name, val in new_metrics.items():
        print(f"    {name:>12s}: {val:.4f}")

    print("\n[STEP 4] Comparing models and deciding promotion...")
    promoted = promote_if_better(config, new_run_id, new_metrics, prod_metrics)

    print(f"\n{'='*60}")
    print(f"  AUTO-RETRAIN PIPELINE COMPLETE")
    print(f"  Drift detected:     {drift_status['drift_detected']}")
    print(f"  Model retrained:    Yes")
    print(f"  New model promoted: {'Yes' if promoted else 'No (current is better)'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
