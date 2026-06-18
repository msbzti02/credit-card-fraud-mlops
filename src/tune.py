import os
import sys
import tempfile
import warnings

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
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
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_classification_report,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_objective(X_train, y_train, X_val, y_val, config):

    ss = config["tuning"]["search_space"]

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int(
                "max_depth", ss["max_depth"][0], ss["max_depth"][1]
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", ss["n_estimators"][0], ss["n_estimators"][1]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                ss["learning_rate"][0],
                ss["learning_rate"][1],
                log=True,
            ),
            "subsample": trial.suggest_float(
                "subsample", ss["subsample"][0], ss["subsample"][1]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", ss["colsample_bytree"][0], ss["colsample_bytree"][1]
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", ss["min_child_weight"][0], ss["min_child_weight"][1]
            ),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight",
                ss["scale_pos_weight"][0],
                ss["scale_pos_weight"][1],
            ),
        }

        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)
            mlflow.set_tag("trial_number", trial.number)

            model = XGBClassifier(
                **params,
                random_state=config["data"]["random_state"],
                eval_metric="aucpr",
                use_label_encoder=False,
                device=config["models"]["xgboost"].get("device", "cpu"),
            )
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]

            score = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_prob)

            mlflow.log_metric("f1_score", score)
            mlflow.log_metric("roc_auc", auc)

        return score

    return objective


def main():

    config = load_config()
    n_trials = config["tuning"]["n_trials"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(config)
        print("[INFO] Loaded pre-processed data.")
    except FileNotFoundError:
        print("[INFO] Processed data not found — running preprocessing...")
        df = load_data(config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
            df, config, apply_smote=True, save=True
        )

    feature_names = list(X_train.columns)

    print(f"\n{'='*60}")
    print(f"  Hyperparameter Tuning — {n_trials} trials")
    print(f"{'='*60}\n")

    with mlflow.start_run(run_name="Optuna_XGBoost_Tuning") as parent_run:
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("tuning_method", "Optuna-TPE")
        mlflow.set_tag("n_trials", n_trials)

        study = optuna.create_study(direction=config["tuning"]["direction"])
        objective = create_objective(X_train, y_train, X_val, y_val, config)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        print(f"\n  Best Trial #{best.number}")
        print(f"  Best F1: {best.value:.4f}")
        print(f"  Best Params: {best.params}")

        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_f1_score", best.value)

        print("\n  Retraining best model on full training data ...")
        best_model = XGBClassifier(
            **best.params,
            random_state=config["data"]["random_state"],
            eval_metric="aucpr",
            use_label_encoder=False,
        )
        best_model.fit(X_train, y_train, verbose=False)

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        log_metrics_to_mlflow(metrics)

        print("\n  Final Test Metrics:")
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
                plot_feature_importance(
                    best_model,
                    feature_names,
                    save_path=os.path.join(tmp, "feature_importance.png"),
                ),
                save_classification_report(
                    y_test, y_pred, os.path.join(tmp, "classification_report.txt")
                ),
            ]

            try:
                fig_hist = optuna.visualization.plot_optimization_history(study)
                hist_path = os.path.join(tmp, "optimization_history.html")
                fig_hist.write_html(hist_path)
                artifacts.append(hist_path)
            except Exception:
                pass

            try:
                fig_imp = optuna.visualization.plot_param_importances(study)
                imp_path = os.path.join(tmp, "param_importances.html")
                fig_imp.write_html(imp_path)
                artifacts.append(imp_path)
            except Exception:
                pass

            log_artifacts_to_mlflow(artifacts)

        mlflow.xgboost.log_model(best_model, "best_model")

        parent_run_id = parent_run.info.run_id

    print(f"\n{'='*60}")
    print(f"  TUNING COMPLETE")
    print(f"  Parent Run ID : {parent_run_id}")
    print(f"  Best F1       : {best.value:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
