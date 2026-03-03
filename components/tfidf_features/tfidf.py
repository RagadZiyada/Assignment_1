import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    return parser.parse_args()

def resolve_parquet_path(input_path: str) -> str:
    if os.path.isfile(input_path):
        return input_path

    candidate = os.path.join(input_path, "data.parquet")
    if os.path.exists(candidate):
        return candidate

    parquet_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.endswith(".parquet")
    ]
    if parquet_files:
        return parquet_files[0]

    raise FileNotFoundError(f"No parquet file found in: {input_path}")

def save_dense_df(matrix, keys_df, feature_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=feature_names)
    final_df = pd.concat([keys_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    final_df.to_parquet(os.path.join(out_dir, "data.parquet"), index=False)

def main():
    args = parse_args()

    train_path = resolve_parquet_path(args.train)
    val_path = resolve_parquet_path(args.val)
    test_path = resolve_parquet_path(args.test)

    print(f"Reading train from: {train_path}")
    print(f"Reading val from: {val_path}")
    print(f"Reading test from: {test_path}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    for df in [train_df, val_df, test_df]:
        df["reviewText"] = df["reviewText"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df["reviewText"])
    X_val = vectorizer.transform(val_df["reviewText"])
    X_test = vectorizer.transform(test_df["reviewText"])

    feature_names = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]

    save_dense_df(X_train, train_df[["asin", "reviewerID"]], feature_names, args.train_out)
    save_dense_df(X_val, val_df[["asin", "reviewerID"]], feature_names, args.val_out)
    save_dense_df(X_test, test_df[["asin", "reviewerID"]], feature_names, args.test_out)

    print("TF-IDF completed successfully")

if __name__ == "__main__":
    main()