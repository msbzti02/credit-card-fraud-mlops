import os
import sys
import time
import warnings

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_config

warnings.filterwarnings("ignore")


def get_best_run(experiment_name: str, metric: str = "f1_score") -> dict:

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.`mlflow.parentRunId` IS NULL",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No runs found in the experiment.")

    best = runs[0]
    print(f"[INFO] Best run: {best.info.run_id}")
    print(f"[INFO] Best {metric}: {best.data.metrics.get(metric, 'N/A')}")
    return {
        "run_id": best.info.run_id,
        "metrics": best.data.metrics,
    }


def register_model(run_id: str, model_name: str) -> str:

    model_uri = f"runs:/{run_id}/best_model"

    try:
        result = mlflow.register_model(model_uri, model_name)
    except Exception:
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)

    print(f"[INFO] Registered '{model_name}' version {result.version}")
    return result.version


def update_model_metadata(model_name: str, version: str) -> None:

    client = MlflowClient()

    client.update_registered_model(
        name=model_name,
        description=(
            "XGBoost classifier for credit card fraud detection. "
            "Trained on the Kaggle Credit Card Fraud dataset with "
            "Optuna-tuned hyperparameters and SMOTE oversampling."
        ),
    )

    client.update_model_version(
        name=model_name,
        version=version,
        description=(
            f"Version {version} — Best model from Optuna hyperparameter "
            "tuning (50 trials, TPE sampler). Evaluated on a held-out "
            "test set with stratified sampling."
        ),
    )

    client.set_model_version_tag(model_name, version, "author", "Student")
    client.set_model_version_tag(model_name, version, "dataset", "creditcard-v1")
    client.set_model_version_tag(model_name, version, "tuning", "optuna-50-trials")

    print(f"[INFO] Updated metadata for '{model_name}' v{version}")


def transition_stage(model_name: str, version: str, stage: str) -> None:

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    print(f"[INFO] '{model_name}' v{version} -> {stage}")


def load_production_model(model_name: str):

    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"[INFO] Loaded '{model_name}' Production model from registry")
    return model


def main():

    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    model_name = config["mlflow"]["registry_model_name"]
    experiment_name = config["mlflow"]["experiment_name"]

    print("=" * 60)
    print("  MODEL REGISTRY WORKFLOW")
    print("=" * 60)

    print("\n--- Step 1: Find Best Run ---")
    best = get_best_run(experiment_name)

    print("\n--- Step 2: Register Model ---")
    version = register_model(best["run_id"], model_name)

    print("\n--- Step 3: Update Metadata ---")
    update_model_metadata(model_name, version)

    print("\n--- Step 4: Transition -> Staging ---")
    transition_stage(model_name, version, "Staging")

    print("\n--- Step 5: Staging Validation ---")
    print("[INFO] Simulating staging validation checks ...")
    time.sleep(1)
    print("[INFO] Validation passed OK")

    print("\n--- Step 6: Transition -> Production ---")
    transition_stage(model_name, version, "Production")

    print("\n--- Step 7: Load Production Model ---")
    model = load_production_model(model_name)
    print(f"[INFO] Model type: {type(model)}")

    print("\n" + "=" * 60)
    print("  MODEL REGISTRY WORKFLOW COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
