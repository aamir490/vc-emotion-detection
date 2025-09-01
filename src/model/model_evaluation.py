import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os

# Choose feature type
FEATURE_TYPE = "tfidf"  # change to "bow" if needed

# Load trained model
model_path = os.path.join("model", "model.pkl")
clf = pickle.load(open(model_path, 'rb'))

# Load test features
test_data = pd.read_csv(f'./data/features/test_{FEATURE_TYPE}.csv')

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]  # probability of positive class

# Calculate metrics
metrics_dict = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'auc': roc_auc_score(y_test, y_pred_proba)
}

# Save metrics
os.makedirs("metrics", exist_ok=True)
metrics_path = os.path.join("metrics", "metrics.json")
with open(metrics_path, 'w') as file:
    json.dump(metrics_dict, file, indent=4)

print(f"Metrics calculated and saved at {metrics_path}")
