from pathlib import Path
import argparse
import json
import random
import numpy as np
import pandas as pd

from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


def evaluate_individual(individual, X, y, feature_names, alpha=0.01):
    selected = [feature_names[i] for i, bit in enumerate(individual) if bit == 1]

    if len(selected) == 0:
        return (1e9,)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=1
    )

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        X[selected],
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1
    )

    rmse = -scores.mean()
    penalty = alpha * len(selected)
    return (rmse + penalty,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--population_size", type=int, default=8)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--cxpb", type=float, default=0.5)
    parser.add_argument("--mutpb", type=float, default=0.2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(input_dir / "train_filtered.csv")
    test_df = pd.read_csv(input_dir / "test_filtered.csv")

    id_col = "unit_number"
    target_col = "target_RUL"
    feature_names = [c for c in train_df.columns if c not in [id_col, target_col]]

    X_train = train_df[feature_names]
    y_train = train_df[target_col]

    random.seed(42)
    np.random.seed(42)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(feature_names))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, X=X_train, y=y_train, feature_names=feature_names, alpha=0.01)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=args.population_size)
    hall_of_fame = tools.HallOfFame(1)

    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        ngen=args.generations,
        halloffame=hall_of_fame,
        verbose=True
    )

    best = hall_of_fame[0]
    selected_features = [feature_names[i] for i, bit in enumerate(best) if bit == 1]

    if not selected_features:
        raise ValueError("GA selected zero features")

    train_selected = pd.concat(
        [train_df[[id_col]], train_df[selected_features], train_df[[target_col]]],
        axis=1
    )
    test_selected = pd.concat(
        [test_df[[id_col]], test_df[selected_features], test_df[[target_col]]],
        axis=1
    )

    train_selected.to_csv(output_dir / "train_ga_selected.csv", index=False)
    test_selected.to_csv(output_dir / "test_ga_selected.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(output_dir / "selected_features.csv", index=False)

    with open(output_dir / "ga_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_feature_count": len(selected_features),
                "best_fitness": float(best.fitness.values[0])
            },
            f,
            indent=2
        )

    print("ga_select finished successfully")
    print("Selected features:", len(selected_features))


if __name__ == "__main__":
    main()