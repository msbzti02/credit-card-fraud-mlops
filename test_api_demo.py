                                                    
import requests, json, pandas as pd

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

legit_idx = y_test[y_test == 0].index[0]
legit_features = X_test.iloc[legit_idx].tolist()

fraud_idx = y_test[y_test == 1].index[0]
fraud_features = X_test.iloc[fraud_idx].tolist()

print("=" * 60)
print("  MODEL OUTPUT DEMO")
print("=" * 60)

print("\n--- Test 1: Real LEGITIMATE Transaction ---")
resp = requests.post("http://localhost:8000/predict", json={"features": legit_features})
print(json.dumps(resp.json(), indent=2))

print("\n--- Test 2: Real FRAUD Transaction ---")
resp = requests.post("http://localhost:8000/predict", json={"features": fraud_features})
print(json.dumps(resp.json(), indent=2))

print("\n--- Test 3: Health Check ---")
resp = requests.get("http://localhost:8000/health")
print(json.dumps(resp.json(), indent=2))

print("\n--- Test 4: Model Info ---")
resp = requests.get("http://localhost:8000/model-info")
print(json.dumps(resp.json(), indent=2))

print("\n--- Test 5: Batch Prediction (3 transactions) ---")
batch = {
    "transactions": [
        {"features": legit_features},
        {"features": fraud_features},
        {"features": X_test.iloc[y_test[y_test == 0].index[5]].tolist()},
    ]
}
resp = requests.post("http://localhost:8000/predict_batch", json=batch)
result = resp.json()
print("  Total transactions:", result["total"])
print("  Fraud detected:    ", result["fraud_count"])
for i, pred in enumerate(result["predictions"]):
    print(f"  Transaction {i+1}: {pred['label']:>12s}  (prob: {pred['probability']:.6f})")

print("\n" + "=" * 60)