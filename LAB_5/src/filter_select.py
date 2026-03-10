from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression


def correlation_filter(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return [c for c in df.columns if c not in to_drop]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--variance_threshold", type=float, default=0.0)
    parser.add_argument("--corr_threshold", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / "train_features.csv")
    test_df = pd.read_csv(input_dir / "test_features.csv")

    id_col = "unit_number"
    target_col = "target_RUL"

    feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df[target_col]

    vt = VarianceThreshold(threshold=args.variance_threshold)
    X_train_vt = vt.fit_transform(X_train)
    kept_vt = X_train.columns[vt.get_support()].tolist()

    X_train_vt_df = pd.DataFrame(X_train_vt, columns=kept_vt)
    X_test_vt_df = X_test[kept_vt].copy()

    kept_corr = correlation_filter(X_train_vt_df, threshold=args.corr_threshold)
    X_train_corr = X_train_vt_df[kept_corr].copy()
    X_test_corr = X_test_vt_df[kept_corr].copy()

    mi = mutual_info_regression(X_train_corr, y_train, random_state=42)
    mi_df = pd.DataFrame({"feature": X_train_corr.columns, "mi_score": mi}).sort_values(
        "mi_score", ascending=False
    )

    top_k = min(args.top_k, len(mi_df))
    selected_features = mi_df.head(top_k)["feature"].tolist()

    train_selected = pd.concat(
        [train_df[[id_col]], X_train_corr[selected_features], train_df[[target_col]]],
        axis=1
    )
    test_selected = pd.concat(
        [test_df[[id_col]], X_test_corr[selected_features], test_df[[target_col]]],
        axis=1
    )

    train_selected.to_csv(output_dir / "train_filtered.csv", index=False)
    test_selected.to_csv(output_dir / "test_filtered.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(output_dir / "filtered_feature_list.csv", index=False)

    summary = {
        "original_feature_count": len(feature_cols),
        "after_variance_count": len(kept_vt),
        "after_correlation_count": len(kept_corr),
        "after_mi_top_k_count": len(selected_features)
    }

    with open(output_dir / "filter_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("filter_select finished successfully")
    print("Features after filtering:", len(selected_features))


if __name__ == "__main__":
    main()