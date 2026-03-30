import argparse
import os
import time
import joblib
import mlflow
import azureml.mlflow
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--C", type=float, default=5.724181499928872)
    parser.add_argument("--max_iter", type=int, default=300)
    return parser.parse_args()


def resolve_parquet_path(path: str) -> str:
    if os.path.isfile(path):
        return path

    candidate = os.path.join(path, "data.parquet")
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(f"Could not find parquet file at: {path}")


def load_data(path: str) -> pd.DataFrame:
    parquet_path = resolve_parquet_path(path)
    return pd.read_parquet(parquet_path)


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' is missing.")
    df = df.copy()
    df["label"] = (df["overall"] >= 4).astype(int)
    return df


def expand_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands list/array-like object columns into numeric columns if they exist.
    Leaves scalar columns unchanged.
    """
    df = df.copy()
    expanded_parts = []
    drop_cols = []

    for col in df.columns:
        if df[col].dtype != "object":
            continue

        first_valid = None
        for val in df[col]:
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                first_valid = val
                break

        if first_valid is None:
            continue

        if isinstance(first_valid, (list, tuple, np.ndarray)):
            arr = pd.DataFrame(df[col].apply(lambda x: x if isinstance(x, (list, tuple, np.ndarray)) else []).tolist())
            arr = arr.add_prefix(f"{col}_")
            expanded_parts.append(arr)
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    if expanded_parts:
        df = pd.concat([df.reset_index(drop=True)] + [part.reset_index(drop=True) for part in expanded_parts], axis=1)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the merged Lab 4/Assignment 1 features directly.
    No feature engineering should happen here.
    """
    df = expand_object_columns(df)

    drop_cols = [c for c in ["asin", "reviewerID", "overall", "label"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Keep only numeric columns from the merged dataset
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.shape[1] == 0:
        raise RuntimeError("Feature matrix is empty after selecting numeric columns.")

    return X


def evaluate(model, X, y, split: str):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        f"{split}_accuracy": accuracy_score(y, preds),
        f"{split}_precision": precision_score(y, preds, zero_division=0),
        f"{split}_recall": recall_score(y, preds, zero_division=0),
        f"{split}_f1": f1_score(y, preds, zero_division=0),
    }

    if len(np.unique(y)) > 1:
        metrics[f"{split}_auc"] = roc_auc_score(y, probs)

    for name, value in metrics.items():
        mlflow.log_metric(name, float(value))
        print(f"{name}: {value}")


def main():
    args = parse_args()
    start_time = time.time()

    mlflow.log_param("model_name", "logistic_regression")
    mlflow.log_param("C", args.C)
    mlflow.log_param("max_iter", args.max_iter)

    print("Loading data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)

    print("Creating labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)

    print("Building features...")
    X_train = build_features(train_df)
    y_train = train_df["label"]

    X_val = build_features(val_df)
    y_val = val_df["label"]

    X_test = build_features(test_df)
    y_test = test_df["label"]

    # Ensure identical feature columns across all splits
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    if len(X_train) == 0:
        raise RuntimeError("Training feature matrix is empty.")

    print("Training model...")
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        random_state=42,
        n_jobs=None
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)
    print(f"training_runtime_seconds: {runtime}")
    print("Done.")


if __name__ == "__main__":
    main()

