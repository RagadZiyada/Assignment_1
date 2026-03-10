from pathlib import Path
import argparse
import pandas as pd


COLUMN_NAMES = (
    ["unit_number", "time_in_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_fd001_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    while df.shape[1] > len(COLUMN_NAMES):
        df = df.iloc[:, :-1]
    df.columns = COLUMN_NAMES
    return df


def load_rul_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    df.columns = ["final_rul"]
    df["unit_number"] = range(1, len(df) + 1)
    return df


def add_train_rul(train_df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = train_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    df = train_df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    return df.drop(columns=["max_cycle"])


def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = test_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]

    df = test_df.merge(max_cycles, on="unit_number", how="left")
    df = df.merge(rul_df, on="unit_number", how="left")
    df["RUL"] = df["final_rul"] + (df["max_cycle"] - df["time_in_cycles"])
    return df.drop(columns=["max_cycle", "final_rul"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_fd001_file(data_dir / "train_FD001.txt")
    test_df = load_fd001_file(data_dir / "test_FD001.txt")
    rul_df = load_rul_file(data_dir / "RUL_FD001.txt")

    train_df = add_train_rul(train_df)
    test_df = add_test_rul(test_df, rul_df)

    train_df.to_csv(output_dir / "train_prepared.csv", index=False)
    test_df.to_csv(output_dir / "test_prepared.csv", index=False)

    print("prepare_data finished successfully")


if __name__ == "__main__":
    main()