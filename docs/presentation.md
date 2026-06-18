# Slide 1: Title Slide
**Title:** Development and Evaluation of a Machine Learning Lifecycle Management System using MLflow
**Course:** AIN-3009 MLOPS
**Name:** [Your Name/Surname]
**Date:** May 2026

---

# Slide 2: Project Overview & Domain
*   **Domain:** Financial Sector (Credit Card Fraud Detection).
*   **Dataset:** Kaggle Credit Card Fraud (284,807 transactions).
*   **Challenge:** Extreme class imbalance (~0.17% fraud).
*   **Solution:** SMOTE, synthetic data augmentation, and rigorous MLOps lifecycle management.

---

# Slide 3: MLOps Architecture
*   A 7-stage automated pipeline orchestrated by Python and MLflow.
*   **Stages:** Data Preprocessing -> Model Training -> Stacking/Tuning -> Registry -> Optimization -> Explainability -> Monitoring.
*   Everything tracked, reproducible, and ready for production.

---

# Slide 4: Objective 1 - Experiment Tracking
*   **Models Trained:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, MLP.
*   **Metrics Logged:** Accuracy, Precision, Recall, F1, MCC, Brier Score, Log Loss.
*   **Artifacts:** Confusion Matrices, ROC/PR Curves, Feature Importances.
*   *Demo/Screenshot Idea:* Show the MLflow dashboard with 6 models and System Metrics (CPU/RAM).

---

# Slide 5: Objective 2 - Training & Tuning
*   Used **Optuna** for Bayesian Hyperparameter Optimization.
*   Ran 50 trials on the XGBoost classifier.
*   Each trial is logged as a nested run inside MLflow to find the absolute best F1/MCC score.
*   *Demo/Screenshot Idea:* Show the nested Optuna trials in MLflow.

---

# Slide 6: Objective 5 - Model Registry
*   The best model is registered as `FraudDetector`.
*   Logged with a complete **Model Signature** (inputs/outputs schema) and Input Example.
*   Managed lifecycle transitions from `None` -> `Staging` -> `Production`.
*   *Demo/Screenshot Idea:* Show the Model Registry UI and the Schema tab.

---

# Slide 7: Objective 3 - Model Deployment
*   Deployed the production model using **FastAPI**.
*   The API dynamically loads the model using `models:/FraudDetector/Production`.
*   Allows seamless updates without changing the server code.
*   *Demo/Screenshot Idea:* Show the FastAPI Swagger UI (`/docs`) running.

---

# Slide 8: Objective 4 - Monitoring & Drift
*   Simulated real-time transaction batches to monitor performance.
*   Integrated **Evidently AI** concepts to detect data drift.
*   When drift is detected (metrics drop), it triggers an auto-retraining pipeline to update the model.
*   *Demo/Screenshot Idea:* Show the performance over time dropping, triggering retraining.

---

# Slide 9: Conclusion
*   Successfully built an end-to-end, enterprise-grade MLOps system.
*   Combines advanced ML techniques (SMOTE, Optuna) with strict governance (MLflow).
*   Ready for real-world financial deployment.
*   **Thank You! Any questions?**
