import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
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

def main():
    args = parse_args()

    length_path = resolve_parquet_path(args.length)
    sentiment_path = resolve_parquet_path(args.sentiment)
    tfidf_path = resolve_parquet_path(args.tfidf)
    sbert_path = resolve_parquet_path(args.sbert)

    print(f"Reading length from: {length_path}")
    print(f"Reading sentiment from: {sentiment_path}")
    print(f"Reading tfidf from: {tfidf_path}")
    print(f"Reading sbert from: {sbert_path}")

    length_df = pd.read_parquet(length_path)
    sentiment_df = pd.read_parquet(sentiment_path)
    tfidf_df = pd.read_parquet(tfidf_path)
    sbert_df = pd.read_parquet(sbert_path)

    merged = length_df.merge(sentiment_df, on=["asin", "reviewerID"], how="inner")
    merged = merged.merge(tfidf_df, on=["asin", "reviewerID"], how="inner")
    merged = merged.merge(sbert_df, on=["asin", "reviewerID"], how="inner")

    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")
    merged.to_parquet(output_path, index=False)

    print(f"Saved merged features to: {output_path}")
    print("Merged rows:", len(merged))
    print("Merged columns:", len(merged.columns))

if __name__ == "__main__":
    main()