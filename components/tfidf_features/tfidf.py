import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--deploy", type=str, required=True)
    parser.add_argument("--max_features", type=int, default=1000)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    return parser.parse_args()


def load_df(path):
    return pd.read_parquet(path)


def get_text_column(df):
    candidates = ["reviewText", "review_text", "text", "normalized_text"]
    for col in candidates:
        if col in df.columns:
            return col
    raise RuntimeError(
        f"Could not find a text column. Tried: {candidates}. Found columns: {list(df.columns)}"
    )


def transform_split(df, vectorizer, text_col):
    transformed = vectorizer.transform(df[text_col].fillna("").astype(str))
    tfidf_df = pd.DataFrame(
        transformed.toarray(),
        columns=[f"tfidf_{i}" for i in range(transformed.shape[1])],
        index=df.index
    )

    base_cols = [c for c in ["asin", "reviewerID", "overall"] if c in df.columns]
    result = pd.concat([df[base_cols].reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return result


def save_df(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "data.parquet"), index=False)


def main():
    args = parse_args()

    train_df = load_df(args.train)
    val_df = load_df(args.val)
    test_df = load_df(args.test)
    deploy_df = load_df(args.deploy)

    text_col = get_text_column(train_df)

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectorizer.fit(train_df[text_col].fillna("").astype(str))

    train_out_df = transform_split(train_df, vectorizer, text_col)
    val_out_df = transform_split(val_df, vectorizer, text_col)
    test_out_df = transform_split(test_df, vectorizer, text_col)
    deploy_out_df = transform_split(deploy_df, vectorizer, text_col)

    save_df(train_out_df, args.train_out)
    save_df(val_out_df, args.val_out)
    save_df(test_out_df, args.test_out)
    save_df(deploy_out_df, args.deploy_out)

    print("TF-IDF features created successfully.")
    print("Train rows:", len(train_out_df))
    print("Val rows:", len(val_out_df))
    print("Test rows:", len(test_out_df))
    print("Deploy rows:", len(deploy_out_df))


if __name__ == "__main__":
    main()