import os
import sys
import tempfile
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
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


def main():

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(config)
        print("[INFO] Loaded pre-processed data.")
    except FileNotFoundError:
        df = load_data(config, use_augmented=True)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
            df, config, apply_smote=True, save=True, apply_feature_engineering=True
        )

    print(f"\n{'='*60}")
    print(f"  STACKING ENSEMBLE TRAINING")
    print(f"  Base models: LR, RF, XGBoost, LightGBM, CatBoost")
    print(f"  Meta-learner: Logistic Regression")
    print(f"{'='*60}\n")

    estimators = [
        (
            "lr",
            LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs"
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=580,
                random_state=42,
                eval_metric="aucpr",
                use_label_encoder=False,
            ),
        ),
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=580,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            ),
        ),
        (
            "catboost",
            CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=0,
            ),
        ),
    ]

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="Stacking_Ensemble"):
        mlflow.set_tag("model_type", "StackingEnsemble")
        mlflow.set_tag("base_models", "LR,RF,XGBoost,LightGBM,CatBoost")
        mlflow.set_tag("meta_learner", "LogisticRegression")
        mlflow.set_tag("author", config["project"]["author"])

        mlflow.log_param("n_base_models", len(estimators))
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("stack_method", "predict_proba")

        print("[INFO] Training stacking ensemble (this may take a while) ...")
        stack.fit(X_train, y_train)
        print("[INFO] Training complete.")

        y_pred = stack.predict(X_test)
        y_prob = stack.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        log_metrics_to_mlflow(metrics)

        print("\n  Stacking Ensemble Test Metrics:")
        for name, val in metrics.items():
            print(f"    {name:>12s}: {val:.4f}")

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

        mlflow.sklearn.log_model(stack, "model")

        run_id = mlflow.active_run().info.run_id

    print(f"\n{'='*60}")
    print(f"  STACKING ENSEMBLE COMPLETE")
    print(f"  Run ID: {run_id}")
    print(f"  F1: {metrics['f1_score']:.4f}  |  AUC: {metrics['roc_auc']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
