import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import load_config, load_data


class TestConfiguration:

    def test_config_loads_successfully(self):

        config = load_config()
        assert isinstance(config, dict)

    def test_config_has_all_sections(self):

        config = load_config()
        required = [
            "project",
            "data",
            "mlflow",
            "models",
            "tuning",
            "monitoring",
            "serving",
        ]
        for section in required:
            assert section in config, f"Missing section: {section}"

    def test_data_config_values(self):

        config = load_config()
        assert 0 < config["data"]["test_size"] < 1
        assert 0 < config["data"]["val_size"] < 1
        total_split = config["data"]["test_size"] + config["data"]["val_size"]
        assert total_split < 1, "test_size + val_size must be less than 1"

    def test_model_configs_exist(self):

        config = load_config()
        assert "logistic_regression" in config["models"]
        assert "random_forest" in config["models"]
        assert "xgboost" in config["models"]

    def test_tuning_config_valid(self):

        config = load_config()
        assert config["tuning"]["n_trials"] > 0
        assert config["tuning"]["direction"] in ("maximize", "minimize")
        assert "search_space" in config["tuning"]

    def test_mlflow_config(self):

        config = load_config()
        assert config["mlflow"]["tracking_uri"] is not None
        assert config["mlflow"]["experiment_name"] is not None
        assert config["mlflow"]["registry_model_name"] is not None

    def test_serving_config(self):

        config = load_config()
        assert config["serving"]["port"] > 0
        assert config["serving"]["host"] is not None


class TestDataLoading:

    def test_raw_data_exists(self):

        config = load_config()
        assert os.path.exists(
            config["data"]["raw_path"]
        ), f"Dataset not found at {config['data']['raw_path']}"

    def test_data_shape(self):

        config = load_config()
        df = load_data(config)
        assert df.shape[1] == 31, f"Expected 31 columns, got {df.shape[1]}"

    def test_data_has_required_columns(self):

        config = load_config()
        df = load_data(config)
        for col in ["Class", "Amount", "Time"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_is_binary(self):

        config = load_config()
        df = load_data(config)
        assert set(df["Class"].unique()) == {0, 1}

    def test_no_nulls_in_data(self):

        config = load_config()
        df = load_data(config)
        assert df.isnull().sum().sum() == 0

    def test_data_has_sufficient_rows(self):

        config = load_config()
        df = load_data(config)
        assert len(df) > 100000, f"Dataset too small: {len(df)} rows"

    def test_class_imbalance_exists(self):

        config = load_config()
        df = load_data(config)
        fraud_ratio = df["Class"].mean()
        assert fraud_ratio < 0.01, f"Fraud ratio {fraud_ratio:.4%} too high"


class TestPreprocessing:

    def test_preprocessing_runs(self):

        from src.data_preprocessing import preprocess_data

        config = load_config()
        df = load_data(config)

        df_small = pd.concat(
            [
                df[df["Class"] == 0].head(1000),
                df[df["Class"] == 1].head(100),
            ]
        ).reset_index(drop=True)
        result = preprocess_data(df_small, config, apply_smote=True, save=False)
        assert len(result) == 6

    def test_scaling_removes_raw_columns(self):

        from src.data_preprocessing import preprocess_data

        config = load_config()
        df = load_data(config)
        df_small = pd.concat(
            [
                df[df["Class"] == 0].head(1000),
                df[df["Class"] == 1].head(100),
            ]
        ).reset_index(drop=True)
        X_train, _, _, _, _, _ = preprocess_data(
            df_small, config, apply_smote=False, save=False
        )
        assert "Amount" not in X_train.columns
        assert "Time" not in X_train.columns
        assert "Amount_scaled" in X_train.columns
        assert "Time_scaled" in X_train.columns

    def test_smote_balances_training_data(self):

        from src.data_preprocessing import preprocess_data

        config = load_config()
        df = load_data(config)
        df_small = pd.concat(
            [
                df[df["Class"] == 0].head(1000),
                df[df["Class"] == 1].head(100),
            ]
        ).reset_index(drop=True)
        _, _, _, y_train, _, _ = preprocess_data(
            df_small, config, apply_smote=True, save=False
        )
        fraud_ratio = y_train.mean()
        assert (
            0.4 <= fraud_ratio <= 0.6
        ), f"SMOTE did not balance data: fraud_ratio={fraud_ratio:.2%}"

    def test_no_data_leakage(self):

        from src.data_preprocessing import preprocess_data

        config = load_config()
        df = load_data(config)
        df_small = pd.concat(
            [
                df[df["Class"] == 0].head(1000),
                df[df["Class"] == 1].head(100),
            ]
        ).reset_index(drop=True)
        X_train, _, X_test, _, _, _ = preprocess_data(
            df_small, config, apply_smote=False, save=False
        )

        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, f"Data leakage: {len(overlap)} shared indices"

    def test_stratified_split_preserves_ratio(self):

        from src.data_preprocessing import preprocess_data

        config = load_config()
        df = load_data(config)
        df_small = pd.concat(
            [
                df[df["Class"] == 0].head(1000),
                df[df["Class"] == 1].head(100),
            ]
        ).reset_index(drop=True)
        _, _, _, _, _, y_test = preprocess_data(
            df_small, config, apply_smote=False, save=False
        )
        original_ratio = 100 / 1100
        test_ratio = y_test.mean()

        assert abs(test_ratio - original_ratio) < 0.05, (
            f"Stratification broken: original={original_ratio:.2%}, "
            f"test={test_ratio:.2%}"
        )


class TestEvaluation:

    def test_compute_metrics_keys(self):

        from src.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.1])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        expected_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc",
        }
        assert expected_keys == set(metrics.keys())

    def test_metrics_in_valid_range(self):

        from src.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.1])

        metrics = compute_metrics(y_true, y_pred, y_prob)
        for name, val in metrics.items():
            assert 0 <= val <= 1, f"{name} = {val} is out of range"

    def test_perfect_predictions(self):

        from src.evaluate import compute_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["f1_score"] == 1.0
        assert metrics["accuracy"] == 1.0

    def test_confusion_matrix_plot(self):

        from src.evaluate import plot_confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_confusion_matrix(
                y_true,
                y_pred,
                save_path=os.path.join(tmp, "cm.png"),
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_roc_curve_plot(self):

        from src.evaluate import plot_roc_curve

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.8, 0.9])
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_roc_curve(
                y_true,
                y_prob,
                save_path=os.path.join(tmp, "roc.png"),
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_classification_report_saves(self):

        from src.evaluate import save_classification_report

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            path = save_classification_report(
                y_true,
                y_pred,
                save_path=os.path.join(tmp, "report.txt"),
            )
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "precision" in content.lower()
            assert "recall" in content.lower()


class TestMonitoring:

    def test_create_batches(self):

        from src.monitor import create_batches

        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        batches = create_batches(X, y, 5)
        assert len(batches) == 5
        total_rows = sum(len(b[0]) for b in batches)
        assert total_rows == 100

    def test_inject_drift_changes_data(self):

        from src.monitor import inject_drift

        X = pd.DataFrame(
            {
                "V1": np.random.randn(100),
                "V2": np.random.randn(100),
                "V3": np.random.randn(100),
                "Amount_scaled": np.random.randn(100),
            }
        )
        X_drifted = inject_drift(X, drift_magnitude=3.0)

        assert not np.allclose(X["V1"].values, X_drifted["V1"].values)

        assert np.allclose(X["Amount_scaled"].values, X_drifted["Amount_scaled"].values)

    def test_fallback_drift_no_drift(self):

        from src.monitor import _fallback_drift_check

        data = pd.DataFrame(
            np.random.randn(500, 5), columns=[f"f{i}" for i in range(5)]
        )
        result = _fallback_drift_check(data, data)
        assert result["share_drifted_features"] == 0.0
        assert result["dataset_drift"] is False

    def test_fallback_drift_detects_shift(self):

        from src.monitor import _fallback_drift_check

        ref = pd.DataFrame(np.random.randn(500, 5), columns=[f"f{i}" for i in range(5)])
        shifted = ref + 10
        result = _fallback_drift_check(ref, shifted)
        assert result["share_drifted_features"] > 0.5
        assert result["dataset_drift"] is True

    def test_performance_plot_saves(self):

        from src.monitor import plot_performance_over_time

        batch_metrics = [
            {"f1_score": 0.8, "roc_auc": 0.9, "precision": 0.7, "recall": 0.9},
            {"f1_score": 0.75, "roc_auc": 0.88, "precision": 0.65, "recall": 0.85},
            {"f1_score": 0.6, "roc_auc": 0.7, "precision": 0.5, "recall": 0.7},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_performance_over_time(
                batch_metrics, os.path.join(tmp, "perf.png")
            )
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0


class TestServingSchemas:

    def test_transaction_valid(self):

        from src.serve import Transaction

        t = Transaction(features=[0.0] * 30)
        assert len(t.features) == 30

    def test_transaction_wrong_length_rejected(self):

        from src.serve import Transaction

        with pytest.raises(Exception):
            Transaction(features=[0.0] * 25)

    def test_batch_transaction_valid(self):

        from src.serve import BatchTransactions, Transaction

        batch = BatchTransactions(
            transactions=[Transaction(features=[0.0] * 30) for _ in range(3)]
        )
        assert len(batch.transactions) == 3

    def test_prediction_response_schema(self):

        from src.serve import PredictionResponse

        resp = PredictionResponse(
            prediction=0,
            probability=0.05,
            label="Legitimate",
            timestamp="2026-05-05T12:00:00",
        )
        assert resp.prediction == 0
        assert resp.label == "Legitimate"


class TestMLflowIntegration:

    def test_mlflow_tracking_uri(self):

        import mlflow

        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        assert mlflow.get_tracking_uri() == config["mlflow"]["tracking_uri"]

    def test_mlflow_experiment_exists(self):

        import mlflow

        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        exp = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
        assert exp is not None, "Experiment not found in MLflow"

    def test_mlflow_has_runs(self):

        import mlflow
        from mlflow.tracking import MlflowClient

        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        exp = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.`mlflow.parentRunId` IS NULL",
        )
        assert len(runs) >= 3, f"Expected >= 3 runs, found {len(runs)}"

    def test_runs_have_metrics(self):

        import mlflow
        from mlflow.tracking import MlflowClient

        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        exp = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.`mlflow.parentRunId` IS NULL AND metrics.f1_score > 0",
        )
        assert len(runs) >= 3, f"Expected >= 3 scored runs, found {len(runs)}"
        for run in runs[:5]:
            assert (
                "f1_score" in run.data.metrics
            ), f"Run {run.info.run_id} missing f1_score metric"

    def test_model_registry_has_model(self):

        import mlflow
        from mlflow.tracking import MlflowClient

        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        model_name = config["mlflow"]["registry_model_name"]
        try:
            model = client.get_registered_model(model_name)
            assert model is not None
        except Exception as e:
            pytest.fail(f"Model '{model_name}' not found in registry: {e}")

    def test_production_model_exists(self):

        import mlflow
        from mlflow.tracking import MlflowClient

        config = load_config()
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        model_name = config["mlflow"]["registry_model_name"]
        versions = client.get_latest_versions(model_name, stages=["Production"])
        assert len(versions) > 0, "No Production model version found"


class TestProcessedData:

    def test_processed_files_exist(self):

        config = load_config()
        d = config["data"]["processed_dir"]
        for fname in [
            "X_train.csv",
            "X_val.csv",
            "X_test.csv",
            "y_train.csv",
            "y_val.csv",
            "y_test.csv",
        ]:
            path = os.path.join(d, fname)
            assert os.path.exists(path), f"Missing: {path}"

    def test_processed_feature_count(self):

        config = load_config()
        d = config["data"]["processed_dir"]
        X_test = pd.read_csv(os.path.join(d, "X_test.csv"))
        assert X_test.shape[1] == 30, f"Expected 30 features, got {X_test.shape[1]}"

    def test_processed_no_nulls(self):

        config = load_config()
        d = config["data"]["processed_dir"]
        for fname in ["X_train.csv", "X_val.csv", "X_test.csv"]:
            df = pd.read_csv(os.path.join(d, fname))
            assert df.isnull().sum().sum() == 0, f"NaN found in {fname}"

    def test_processed_target_is_binary(self):

        config = load_config()
        d = config["data"]["processed_dir"]
        for fname in ["y_train.csv", "y_val.csv", "y_test.csv"]:
            y = pd.read_csv(os.path.join(d, fname)).squeeze()
            assert set(y.unique()).issubset(
                {0, 1}
            ), f"Non-binary values in {fname}: {y.unique()}"
