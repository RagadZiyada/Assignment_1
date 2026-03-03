import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
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

def main():
    args = parse_args()

    parquet_path = resolve_parquet_path(args.data)
    print(f"Reading parquet from: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - args.train_ratio),
        random_state=args.seed,
        shuffle=True
    )

    val_size = args.val_ratio / (1 - args.train_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=args.seed,
        shuffle=True
    )

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"), index=False)

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))

if __name__ == "__main__":
    main()