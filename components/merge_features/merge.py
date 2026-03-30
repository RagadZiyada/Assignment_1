import argparse
import os
import pandas as pd


KEY_COLUMNS = ["asin", "reviewerID"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def load_df(path):
    return pd.read_parquet(path)


def safe_merge(left, right, name):
    for col in KEY_COLUMNS:
        if col not in left.columns or col not in right.columns:
            raise RuntimeError(f"Missing key column '{col}' while merging {name}")
    return left.merge(right, on=KEY_COLUMNS, how="inner")


def drop_duplicate_non_key_columns(df, protected_cols):
    seen = set()
    keep_cols = []
    for col in df.columns:
        if col in protected_cols:
            keep_cols.append(col)
        elif col not in seen:
            keep_cols.append(col)
            seen.add(col)
    return df[keep_cols]


def main():
    args = parse_args()

    length_df = load_df(args.length)
    sentiment_df = load_df(args.sentiment)
    tfidf_df = load_df(args.tfidf)
    sbert_df = load_df(args.sbert)

    # preserve one copy of overall if present
    protected_cols = set(KEY_COLUMNS + ["overall"])

    merged = safe_merge(length_df, sentiment_df, "length + sentiment")
    merged = drop_duplicate_non_key_columns(merged, protected_cols)

    merged = safe_merge(merged, tfidf_df, "previous + tfidf")
    merged = drop_duplicate_non_key_columns(merged, protected_cols)

    merged = safe_merge(merged, sbert_df, "previous + sbert")
    merged = drop_duplicate_non_key_columns(merged, protected_cols)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "data.parquet")
    merged.to_parquet(out_path, index=False)

    print("Merged rows:", len(merged))
    print("Merged columns:", len(merged.columns))
    print("Output written to:", out_path)
    print("Contains keys:", [c for c in KEY_COLUMNS if c in merged.columns])
    print("Contains overall:", "overall" in merged.columns)


if __name__ == "__main__":
    main()