import argparse
import os
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
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

    parquet_path = resolve_parquet_path(args.data)
    print(f"Reading parquet from: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df["reviewText"] = df["reviewText"].fillna("").astype(str)

    sia = SentimentIntensityAnalyzer()
    scores = df["reviewText"].apply(sia.polarity_scores).apply(pd.Series)

    out_df = df[["asin", "reviewerID"]].copy()
    out_df["sentiment_pos"] = scores["pos"]
    out_df["sentiment_neg"] = scores["neg"]
    out_df["sentiment_neu"] = scores["neu"]
    out_df["sentiment_compound"] = scores["compound"]

    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")
    out_df.to_parquet(output_path, index=False)

    print(f"Saved sentiment features to: {output_path}")
    print("Sentiment rows:", len(out_df))

if __name__ == "__main__":
    main()