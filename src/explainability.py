import json
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
from src.evaluate import log_artifacts_to_mlflow

warnings.filterwarnings("ignore")


def get_shap_explainer(model, X_sample):

    import shap

    model_type = type(model).__name__.lower()

    if any(
        t in model_type for t in ["xgb", "lgbm", "lightgbm", "randomforest", "gradient"]
    ):
        return shap.TreeExplainer(model)
    else:

        return shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X_sample, 100),
        )


def plot_shap_summary(shap_values, X_sample, save_path):

    import shap

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("SHAP Feature Importance (Top 20)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[INFO] SHAP summary plot saved to {save_path}")
    return save_path


def plot_shap_bar(shap_values, X_sample, save_path):

    import shap

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_sample, plot_type="bar", show=False, max_display=20
    )
    plt.title("Mean |SHAP| Feature Importance", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[INFO] SHAP bar plot saved to {save_path}")
    return save_path


def plot_shap_waterfall(explainer, shap_values_single, X_single, save_path):

    import shap

    fig, ax = plt.subplots(figsize=(10, 8))

    base_val = explainer.expected_value
    if not np.isscalar(base_val) and len(base_val) > 1:
        base_val = base_val[1]

    explanation = shap.Explanation(
        values=shap_values_single,
        base_values=float(base_val),
        data=X_single.values,
        feature_names=list(X_single.index),
    )

    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title("SHAP Waterfall - Single Prediction", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[INFO] SHAP waterfall plot saved to {save_path}")
    return save_path


def explain_single_prediction(model, explainer, X_single, feature_names):

    import shap

    sv = explainer.shap_values(X_single)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv.flatten()

    indices = np.argsort(np.abs(sv))[::-1]
    top_features = []
    for i in indices[:10]:
        top_features.append(
            {
                "feature": feature_names[i],
                "shap_value": round(float(sv[i]), 6),
                "feature_value": round(float(X_single.iloc[0, i]), 6),
                "direction": (
                    "increases fraud risk" if sv[i] > 0 else "decreases fraud risk"
                ),
            }
        )

    base_val = explainer.expected_value
    if not np.isscalar(base_val) and len(base_val) > 1:
        base_val = base_val[1]

    return {
        "top_contributing_features": top_features,
        "base_value": float(base_val),
    }


def main():

    import shap

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(config)

    model_name = config["mlflow"]["registry_model_name"]
    pyfunc_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

    try:
        model = pyfunc_model._model_impl.python_model
        if hasattr(model, "predict_proba"):
            native_model = model
        else:
            native_model = pyfunc_model._model_impl
    except Exception:
        native_model = pyfunc_model._model_impl

    if hasattr(native_model, "xgb_model"):
        native_model = native_model.xgb_model
    elif hasattr(native_model, "lgb_model"):
        native_model = native_model.lgb_model

    print(f"[INFO] Model type: {type(native_model).__name__}")

    print(f"\n{'='*60}")
    print(f"  SHAP EXPLAINABILITY ANALYSIS")
    print(f"{'='*60}\n")

    n_explain = min(1000, len(X_test))
    X_sample = X_test.head(n_explain)

    print(f"[INFO] Creating SHAP explainer for {type(native_model).__name__}...")
    explainer = get_shap_explainer(native_model, X_train.head(500))

    print(f"[INFO] Computing SHAP values for {n_explain} samples...")
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values_fraud = shap_values[1]
    else:
        shap_values_fraud = shap_values

    mean_abs_shap = np.abs(shap_values_fraud).mean(axis=0)
    feature_importance = sorted(
        zip(X_sample.columns, mean_abs_shap), key=lambda x: x[1], reverse=True
    )

    print("\n  Top 10 Most Important Features (by SHAP):")
    for feat, imp in feature_importance[:10]:
        print(f"    {feat:>25s}: {imp:.6f}")

    fraud_indices = y_test[y_test == 1].index
    if len(fraud_indices) > 0:
        fraud_idx = fraud_indices[0]
        rel_idx = X_test.index.get_loc(fraud_idx)
        X_fraud = X_test.iloc[[rel_idx]]
        explanation = explain_single_prediction(
            native_model, explainer, X_fraud, list(X_test.columns)
        )
        print("\n  Single Fraud Prediction Explanation:")
        for feat_info in explanation["top_contributing_features"][:5]:
            print(
                f"    {feat_info['feature']:>25s}: "
                f"SHAP={feat_info['shap_value']:+.4f} "
                f"({feat_info['direction']})"
            )

    with mlflow.start_run(run_name="SHAP_Explainability"):
        mlflow.set_tag("analysis_type", "explainability")
        mlflow.set_tag("method", "SHAP")
        mlflow.set_tag("model_type", type(native_model).__name__)

        mlflow.log_param("n_samples_explained", n_explain)
        mlflow.log_param("n_features", len(X_sample.columns))

        for feat, imp in feature_importance[:15]:
            safe_name = feat.replace("(", "").replace(")", "").replace(" ", "_")
            mlflow.log_metric(f"shap_{safe_name}", round(float(imp), 6))

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = []

            artifacts.append(
                plot_shap_summary(
                    shap_values_fraud, X_sample, os.path.join(tmp, "shap_summary.png")
                )
            )

            artifacts.append(
                plot_shap_bar(
                    shap_values_fraud, X_sample, os.path.join(tmp, "shap_bar.png")
                )
            )

            if len(fraud_indices) > 0:
                try:
                    sv_single = shap_values_fraud[rel_idx]
                    artifacts.append(
                        plot_shap_waterfall(
                            explainer,
                            sv_single,
                            X_test.iloc[rel_idx],
                            os.path.join(tmp, "shap_waterfall_fraud.png"),
                        )
                    )
                except Exception as e:
                    print(f"[WARN] Waterfall plot failed: {e}")

            if len(fraud_indices) > 0:
                json_path = os.path.join(tmp, "fraud_explanation.json")
                with open(json_path, "w") as f:
                    json.dump(explanation, f, indent=2)
                artifacts.append(json_path)

            rank_path = os.path.join(tmp, "shap_feature_ranking.csv")
            pd.DataFrame(
                feature_importance, columns=["feature", "mean_abs_shap"]
            ).to_csv(rank_path, index=False)
            artifacts.append(rank_path)

            log_artifacts_to_mlflow(artifacts)

    print(f"\n{'='*60}")
    print(f"  SHAP ANALYSIS COMPLETE")
    print(f"  {n_explain} samples explained, results logged to MLflow")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
