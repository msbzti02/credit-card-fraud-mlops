# MLOps Capstone: Presentation Cheat Sheet
*Use this guide to prepare for your 5-8 minute presentation. It covers the narrative, the challenges, and a complete breakdown of your architecture.*

---

## 🎯 1. The Main Goal
The core objective of this project is to design and implement an **Enterprise-Grade Machine Learning Operations (MLOps) Lifecycle**. 

Instead of just training a model in a Jupyter Notebook and stopping there, this project builds a fully automated, end-to-end pipeline using **MLflow** as the central orchestrator. It covers the entire lifecycle of a **Credit Card Fraud Detection** system—from raw data ingestion and hyperparameter tuning to real-time deployment and automated monitoring.

---

## 🚨 2. Key Problems Faced & Our Solutions

During your presentation, professors love hearing about the "challenges" you encountered and how you engineered your way out of them. Talk about these three major problems:

### Problem 1: Extreme Class Imbalance
* **The Issue:** In the real world, fraud is rare. Our dataset has 284,807 transactions, but only ~0.17% are actually fraudulent. If a model simply predicts "Not Fraud" every single time, it gets 99.8% accuracy, but it is completely useless.
* **The Solution:** 
  1. We implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to artificially balance the training dataset.
  2. We discarded "Accuracy" as a metric. Instead, we forced MLflow to track **MCC (Matthews Correlation Coefficient)**, **PR-AUC**, and **F1-Score**, which are much stricter metrics for imbalanced data.

### Problem 2: Concept Drift (Fraudsters Evolve)
* **The Issue:** Fraudsters constantly change their tactics. A model trained in January might be completely obsolete by June.
* **The Solution:** We implemented an automated monitoring system using **Evidently AI**. The system continuously checks incoming live batches of transactions against our original training data. If it detects a "drift" (a change in data distribution), it triggers an **auto-retraining pipeline** to learn the new fraud patterns.

### Problem 3: The "Black Box" Problem in Finance
* **The Issue:** In the financial sector, you cannot just block a customer's credit card and say "the AI told me to." You need interpretability.
* **The Solution:** We implemented **SHAP (SHapley Additive exPlanations)** to break down exactly which features (e.g., Transaction Amount, Time) contributed to the fraud flag, providing transparency.

---

## 📁 3. File-by-File Breakdown (The Architecture)

If the professor asks, "How does your codebase work?", use this file-by-file explanation to walk them through the pipeline:

### ⚙️ The Core Orchestrators
* **`configs/config.yaml`**: The "brain" of the project. Instead of hardcoding variables throughout the code, everything (hyperparameters, database paths, test splits) is controlled from this single file.
* **`run_pipeline.py`**: The master execution script. It acts as the pipeline orchestrator, sequentially running the data preprocessing, training, tuning, registering, and monitoring stages.

### 🧪 Data & Training Layer
* **`src/data_preprocessing.py`**: Handles loading the raw CSV, applying `StandardScaler` to normalize the data, splitting it into Train/Validation/Test sets, and applying SMOTE to balance the fraud classes.
* **`src/evaluate.py`**: A specialized script for calculating advanced metrics (like Brier Score and Log Loss) and generating visual artifacts (Confusion Matrices, ROC curves).
* **`src/train.py`**: Trains **6 different baseline models** (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and MLP Neural Network). It integrates deeply with MLflow to log all parameters, metrics, system hardware usage (CPU/RAM), and input schemas.

### 🚀 Tuning & Governance Layer
* **`src/tune.py`**: Instead of a basic Grid Search, this file uses **Optuna** to perform advanced Bayesian hyperparameter optimization. It runs 50 distinct trials on the XGBoost model, logging every single attempt into MLflow as a nested run.
* **`src/register_model.py`**: The model governance script. It takes the absolute best model from the Optuna tuning phase and registers it into the **MLflow Model Registry**. It handles the lifecycle transition, moving the model safely into the `Staging` and `Production` stages.

### 🌐 Deployment & Monitoring Layer
* **`src/serve.py`**: The deployment layer. It launches a **FastAPI** high-performance REST server. Crucially, it dynamically pulls the active model directly from the MLflow Registry URI (`models:/FraudDetector/Production`), meaning the server auto-updates whenever a new model is promoted.
* **`src/explainability.py`**: Generates SHAP values to explain the model's decision-making process.
* **`src/monitor.py`**: Simulates incoming real-world transaction batches and compares them to the training baseline to detect statistical data drift.
* **`src/auto_retrain.py`**: The automated feedback loop. If `monitor.py` detects too much drift, this script automatically wakes up, retrains the model on the fresh data, and promotes the new version to Production.

---

## 💡 Presentation Tips
1. **Show, Don't Just Tell:** Keep MLflow open (`http://localhost:5000`) during your presentation. When talking about Experiment Tracking, physically show them the 6 models and the nested Optuna runs.
2. **Highlight the Registry:** Click on the "Models" tab in MLflow and show them how the model smoothly transitions into the "Production" stage.
3. **Show the API:** Have the FastAPI Swagger UI (`http://localhost:8000/docs`) open to prove that your model isn't just sitting in a file—it is actively deployed and ready to accept JSON transaction data.
