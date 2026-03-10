from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / "train_ga_selected.csv")
    test_df = pd.read_csv(input_dir / "test_ga_selected.csv")

    id_col = "unit_number"
    target_col = "target_RUL"
    feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )

    print("Train_Model_started")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "feature_count": len(feature_cols),
        "samples_used": int(len(X_train))
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "unit_number": test_df["unit_number"],
        "actual_RUL": y_test,
        "predicted_RUL": preds
    }).to_csv(output_dir / "predictions.csv", index=False)

    joblib.dump(model, output_dir / "model.joblib")

    print("Samples used:", len(X_train))
    print("Selected features:", len(feature_cols))
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)
    print("train_model finished successfully")


if __name__ == "__main__":
    main()