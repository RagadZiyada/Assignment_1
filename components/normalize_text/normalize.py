import argparse
import os
import re
import pandas as pd

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

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    args = parse_args()

    parquet_path = resolve_parquet_path(args.data)
    print(f"Reading parquet from: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df["reviewText"] = df["reviewText"].fillna("").astype(str)
    df["reviewText"] = df["reviewText"].apply(normalize_text)
    df = df[df["reviewText"].str.len() >= 10].copy()

    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(output_path, index=False)

    print(f"Saved normalized data to: {output_path}")
    print("Normalized rows:", len(df))

if __name__ == "__main__":
    main()