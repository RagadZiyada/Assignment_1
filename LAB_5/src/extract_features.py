from pathlib import Path
import argparse
import json
import time
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute


def get_targets(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("unit_number")["RUL"]
        .last()
        .reset_index()
        .rename(columns={"RUL": "target_RUL"})
    )


def extract_tsfresh(df: pd.DataFrame) -> pd.DataFrame:
    features_input = df.drop(columns=["RUL"], errors="ignore").copy()

    features = extract_features(
        features_input,
        column_id="unit_number",
        column_sort="time_in_cycles",
        default_fc_parameters=MinimalFCParameters(),
        disable_progressbar=False,
        n_jobs=1
    )

    impute(features)
    features = features.reset_index().rename(columns={"index": "unit_number"})
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / "train_prepared.csv")
    test_df = pd.read_csv(input_dir / "test_prepared.csv")

    start = time.time()

    print("Feature extraction started")

    train_features = extract_tsfresh(train_df)
    test_features = extract_tsfresh(test_df)

    y_train = get_targets(train_df)
    y_test = get_targets(test_df)

    train_final = train_features.merge(y_train, on="unit_number", how="left")
    test_final = test_features.merge(y_test, on="unit_number", how="left")

    train_final.to_csv(output_dir / "train_features.csv", index=False)
    test_final.to_csv(output_dir / "test_features.csv", index=False)

    runtime = round(time.time() - start, 2)

    with open(output_dir / "feature_runtime.json", "w", encoding="utf-8") as f:
        json.dump({"feature_extraction_seconds": runtime}, f, indent=2)

    print("Feature extraction completed")
    print("Feature extraction runtime (seconds):", runtime)


if __name__ == "__main__":
    main()