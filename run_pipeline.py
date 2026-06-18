import argparse
import subprocess
import sys
import time

STAGES = {
    "generate": {
        "script": "src/synthetic_data.py",
        "description": "Synthetic Data Generation (284K -> 1M rows)",
    },
    "preprocess": {
        "script": "src/data_preprocessing.py",
        "description": "Data Preprocessing (feature eng, scale, split, SMOTE)",
    },
    "train": {
        "script": "src/train.py",
        "description": "Model Training (6 models: LR, RF, XGB, LGBM, CB, MLP)",
    },
    "stack": {
        "script": "src/stacking.py",
        "description": "Stacking Ensemble (5 base models + meta-learner)",
    },
    "tune": {
        "script": "src/tune.py",
        "description": "Hyperparameter Tuning (Optuna + MLflow, 50 trials)",
    },
    "register": {
        "script": "src/register_model.py",
        "description": "Model Registry (register, staging, production)",
    },
    "threshold": {
        "script": "src/threshold_optimization.py",
        "description": "Threshold Optimization (business cost analysis)",
    },
    "explain": {
        "script": "src/explainability.py",
        "description": "SHAP Explainability (feature explanations)",
    },
    "monitor": {
        "script": "src/monitor.py",
        "description": "Performance Monitoring (drift detection)",
    },
    "retrain": {
        "script": "src/auto_retrain.py",
        "description": "Auto-Retraining (drift -> retrain -> promote)",
    },
}


def run_stage(name: str, info: dict) -> bool:

    print(f"\n{'='*60}")
    print(f"  STAGE: {name.upper()}")
    print(f"  {info['description']}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, info["script"]],
        capture_output=False,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  [OK] {name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  [FAIL] {name} FAILED (exit code {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the ML lifecycle pipeline.")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default=None,
        help="Run a specific stage (default: run all stages).",
    )
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("#  MLOps Pipeline — Credit Card Fraud Detection")
    print("#" * 60)

    if args.stage:
        stages_to_run = {args.stage: STAGES[args.stage]}
    else:
        stages_to_run = STAGES

    results = {}
    for name, info in stages_to_run.items():
        success = run_stage(name, info)
        results[name] = success
        if not success and not args.stage:
            print(f"\n[ERROR] Stage '{name}' failed — aborting pipeline.")
            break

    print(f"\n\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        icon = "[OK]" if success else "[FAIL]"
        print(f"  {icon}  {name:>12s} - {STAGES[name]['description']}")
    print("=" * 60)

    all_ok = all(results.values())
    if all_ok:
        print("\n  All stages completed successfully!")
        print("  View results:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
    else:
        print("\n  Some stages failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
