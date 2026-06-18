import os
import sys
import tempfile
import warnings

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_config, load_data, preprocess_data
from src.evaluate import (
    compute_metrics,
    log_artifacts_to_mlflow,
    log_metrics_to_mlflow,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_classification_report,
)

warnings.filterwarnings("ignore")


def train_and_log_model(
    model,
    model_name: str,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names: list,
    config: dict,
    log_feature_imp: bool = False,
):

    with mlflow.start_run(run_name=model_name):

        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset_version", "v1")
        mlflow.set_tag("author", config["project"]["author"])
        mlflow.set_tag("pipeline_stage", "training")
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("framework", "sklearn/xgboost/lightgbm/catboost")

        note_content = f"""
# Model: {model_name}
This run trains a **{model_name}** model for the Credit Card Fraud Detection pipeline.

## Pipeline Configurations
* **SMOTE**: Applied (Handling class imbalance)
* **Dataset**: Augmented
* **Target**: `Class` (0: Legitimate, 1: Fraud)

## Artifacts Logged
* Confusion Matrix
* ROC & PR Curves
* Feature Importances
* System Metrics (CPU/RAM/GPU)
* Model Signature & Input Examples
"""
        mlflow.set_tag("mlflow.note.content", note_content.strip())

        if os.path.exists("configs/config.yaml"):
            mlflow.log_artifact("configs/config.yaml", artifact_path="configuration")

        params = model.get_params()
        mlflow.log_params(params)

        print(f"\n{'='*60}")
        print(f"  Training: {model_name}")
        print(f"{'='*60}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        log_metrics_to_mlflow(metrics)

        for name, val in metrics.items():
            print(f"  {name:>12s}: {val:.4f}")

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = []

            cm_path = plot_confusion_matrix(
                y_test,
                y_pred,
                save_path=os.path.join(tmp, "confusion_matrix.png"),
            )
            artifacts.append(cm_path)

            if y_prob is not None:
                roc_path = plot_roc_curve(
                    y_test,
                    y_prob,
                    save_path=os.path.join(tmp, "roc_curve.png"),
                )
                pr_path = plot_precision_recall_curve(
                    y_test,
                    y_prob,
                    save_path=os.path.join(tmp, "pr_curve.png"),
                )
                artifacts.extend([roc_path, pr_path])

            if log_feature_imp and hasattr(model, "feature_importances_"):
                fi_path = plot_feature_importance(
                    model,
                    feature_names,
                    save_path=os.path.join(tmp, "feature_importance.png"),
                )
                artifacts.append(fi_path)

            cr_path = save_classification_report(
                y_test,
                y_pred,
                save_path=os.path.join(tmp, "classification_report.txt"),
            )
            artifacts.append(cr_path)

            log_artifacts_to_mlflow(artifacts)

        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5] if hasattr(X_train, "iloc") else X_train[:5]

        if "xgb" in model_name.lower() or "xgboost" in model_name.lower():
            mlflow.xgboost.log_model(
                model, "model", signature=signature, input_example=input_example
            )
        elif "catboost" in model_name.lower():
            mlflow.catboost.log_model(
                model, "model", signature=signature, input_example=input_example
            )
        elif "lightgbm" in model_name.lower() or "lgbm" in model_name.lower():
            mlflow.lightgbm.log_model(
                model, "model", signature=signature, input_example=input_example
            )
        else:
            mlflow.sklearn.log_model(
                model, "model", signature=signature, input_example=input_example
            )

        print(f"  [OK] {model_name} logged to MLflow with Signature")
        return mlflow.active_run().info.run_id


def main():

    config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    try:
        mlflow.enable_system_metrics_logging()
        print("[INFO] System metrics logging enabled.")
    except Exception as e:
        print(f"[WARN] Could not enable system metrics logging: {e}")

    df = load_data(config, use_augmented=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        df, config, apply_smote=True, save=True, apply_feature_engineering=True
    )
    feature_names = list(X_train.columns)

    lr_cfg = config["models"]["logistic_regression"]
    rf_cfg = config["models"]["random_forest"]
    xgb_cfg = config["models"]["xgboost"]
    lgbm_cfg = config["models"]["lightgbm"]
    cb_cfg = config["models"]["catboost"]
    mlp_cfg = config["models"]["mlp_neural_network"]

    models = [
        (
            "Logistic_Regression",
            LogisticRegression(
                C=lr_cfg["C"],
                max_iter=lr_cfg["max_iter"],
                class_weight=lr_cfg["class_weight"],
                solver=lr_cfg["solver"],
            ),
            False,
        ),
        (
            "Random_Forest",
            RandomForestClassifier(
                n_estimators=rf_cfg["n_estimators"],
                max_depth=rf_cfg["max_depth"],
                min_samples_split=rf_cfg["min_samples_split"],
                class_weight=rf_cfg["class_weight"],
                random_state=rf_cfg["random_state"],
                n_jobs=rf_cfg["n_jobs"],
            ),
            True,
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=xgb_cfg["n_estimators"],
                max_depth=xgb_cfg["max_depth"],
                learning_rate=xgb_cfg["learning_rate"],
                subsample=xgb_cfg["subsample"],
                colsample_bytree=xgb_cfg["colsample_bytree"],
                scale_pos_weight=xgb_cfg["scale_pos_weight"],
                random_state=xgb_cfg["random_state"],
                eval_metric=xgb_cfg["eval_metric"],
                use_label_encoder=False,
                device=xgb_cfg.get("device", "cpu"),
            ),
            True,
        ),
        (
            "LightGBM",
            LGBMClassifier(
                n_estimators=lgbm_cfg["n_estimators"],
                max_depth=lgbm_cfg["max_depth"],
                learning_rate=lgbm_cfg["learning_rate"],
                subsample=lgbm_cfg["subsample"],
                colsample_bytree=lgbm_cfg["colsample_bytree"],
                scale_pos_weight=lgbm_cfg["scale_pos_weight"],
                random_state=lgbm_cfg["random_state"],
                num_leaves=lgbm_cfg["num_leaves"],
                min_child_samples=lgbm_cfg["min_child_samples"],
                n_jobs=lgbm_cfg["n_jobs"],
                verbose=-1,
            ),
            True,
        ),
        (
            "CatBoost",
            CatBoostClassifier(
                iterations=cb_cfg["iterations"],
                depth=cb_cfg["depth"],
                learning_rate=cb_cfg["learning_rate"],
                random_seed=cb_cfg["random_seed"],
                auto_class_weights=cb_cfg["auto_class_weights"],
                verbose=cb_cfg["verbose"],
                task_type=cb_cfg.get("task_type", "CPU"),
            ),
            True,
        ),
        (
            "MLP_Neural_Network",
            MLPClassifier(
                hidden_layer_sizes=tuple(mlp_cfg["hidden_layers"]),
                max_iter=mlp_cfg["max_iter"],
                learning_rate_init=mlp_cfg["learning_rate_init"],
                random_state=mlp_cfg["random_state"],
                early_stopping=mlp_cfg["early_stopping"],
                validation_fraction=0.1,
            ),
            False,
        ),
    ]

    run_ids = {}
    for name, model, log_fi in models:
        rid = train_and_log_model(
            model=model,
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            config=config,
            log_feature_imp=log_fi,
        )
        run_ids[name] = rid

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — Run IDs")
    print("=" * 60)
    for name, rid in run_ids.items():
        print(f"  {name:>25s} : {rid}")
    print(
        f"\n  View results:  mlflow ui --backend-store-uri {config['mlflow']['tracking_uri']}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
