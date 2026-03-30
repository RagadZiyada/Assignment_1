import json
import os
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ENDPOINT_URL = os.environ["ENDPOINT_URL"]
API_KEY = os.environ["API_KEY"]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def main():
    print("Loading deployment dataset...")
    df = pd.read_parquet("local_data/deploy/data.parquet")

    y_true = (df["overall"] >= 4).astype(int)

    drop_cols = ["asin", "reviewerID", "overall"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = X.select_dtypes(include=[np.number]).fillna(0)

    payload = {"data": X.values.tolist()}

    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=120
    )

    print("Status:", response.status_code)

    preds = response.json()["predictions"]
    y_pred = np.array(preds)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))

if __name__ == "__main__":
    main()
