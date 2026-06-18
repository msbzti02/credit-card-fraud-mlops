# Master Presentation Script: MLOps Credit Card Fraud Detection
*This is your complete, word-for-word master script. It combines the technical explanations with the intuitive analogies (like the Drive-Thru) so you can explain complex MLOps concepts simply and confidently.*

---

## **Slide 1: Title Slide**
**What to say:**
"Hello everyone, and welcome to my presentation. My name is [Your Name]. Today, I will be presenting my term project for the MLOps course. My project focuses on the development of a complete Machine Learning Operations Lifecycle, specifically applied to Credit Card Fraud Detection using MLflow."

---

## **Slide 2: Domain & Problem Statement**
**What to say:**
"To build this MLOps system, I selected the financial sector as my domain, utilizing the Kaggle Credit Card Fraud dataset which contains over 280,000 transactions. 

The biggest challenge with this dataset is extreme class imbalance—only 0.17% of the transactions are actually fraudulent. If a model simply guesses 'Not Fraud' every time, it gets 99% accuracy, which is totally useless in the real world. To solve this, I applied SMOTE (Synthetic Minority Over-sampling Technique) during the preprocessing stage to artificially generate fraud examples, balancing the dataset so the models could learn the fraud patterns properly."

---

## **Slide 3: Architecture Overview**
**What to say:**
"Instead of just training a model in a simple Jupyter notebook, I built a fully automated, end-to-end pipeline. 

As you can see in this flowchart, the raw data first goes through scaling and SMOTE balancing. Then, it enters the training phase where 6 different models are evaluated. After that, the system uses Bayesian optimization for hyperparameter tuning. The winning model is pushed to the MLflow Registry, deployed via a FastAPI endpoint, and finally monitored continuously for data drift."

---

## **Slide 4: Experiment Tracking (MLflow)**
**What to say:**
"Let's look at the first requirement: Experiment Tracking. I trained 6 baseline models, including XGBoost, Random Forest, and a Neural Network. 

Because of the class imbalance, I threw away basic 'Accuracy' and forced MLflow to track strict metrics like the Matthews Correlation Coefficient, PR-AUC, and Brier Score. As you can see in the screenshot on the right, MLflow perfectly logged all my runs, generated confusion matrices, and even tracked live system hardware utilization, like CPU and RAM usage, while the models were training."

---

## **Slide 5: Hyperparameter Tuning**
**What to say:**
"For the next stage, Model Tuning, I didn't want to use a basic Grid Search. Instead, I integrated Optuna directly into MLflow. 

The system executed 50 separate Bayesian optimization trials on the XGBoost model to find the absolute perfect parameters. Every single trial was automatically captured by MLflow as a nested run, making it incredibly easy to identify the winning combination."

---

## **Slide 6: Real-Time Deployment (The Drive-Thru)**
**What to say:**
"For deployment, I built a high-performance REST API using FastAPI. You can think of this as building a 'drive-thru window' so the outside world can actually access the model.

What makes this architecture special is dynamic loading. The server doesn't use a hardcoded model file. Instead, it reaches directly into the MLflow registry URI and asks for whichever model currently holds the 'Production' title. 

The biggest benefit of this is zero-downtime updates: If I train a better model tomorrow and promote it in MLflow, my live API will automatically update to use the new model without me having to change a single line of server code!"

---

## **Slide 7: Performance Monitoring (The Safety Net)**
**What to say:**
"Finally, no MLOps pipeline is complete without monitoring. In the real world, fraud models do not stay accurate forever because hackers constantly invent new tricks. This degradation is called *Concept Drift*. 

To test if my system could handle this, I simulated the real world by feeding 5 live batches of transactions into the model. On Batch 5, I purposefully injected heavily mutated, completely new fraud patterns. As you can see by the sharp drop in the red and blue lines on the chart, the monitoring system successfully caught the failure in real-time.

But the system didn't just crash. Because the drift threshold was breached, the system sounded an alarm and automatically triggered a retraining script. It essentially hit the reset button, retrained itself to learn the hackers' new tricks, and pushed a fresh model into production. It is a completely self-healing system!"

---

## **Slide 8: Conclusion**
**What to say:**
"In conclusion, this project successfully demonstrates a true enterprise-grade machine learning lifecycle. It overcomes extreme class imbalance, ensures 100% reproducibility through MLflow tracking, provides dynamic FastAPI deployment, and closes the loop with automated drift detection and retraining. 

Thank you so much for listening. I would now be happy to answer any questions or show a quick live demo of the dashboard!"
