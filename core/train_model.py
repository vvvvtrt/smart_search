#!/usr/bin/env python3
import argparse
import os
import pandas as pd

from catboost import CatBoostClassifier, Pool

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.environ.get("DATASET_PATH", os.path.join(ROOT_DIR, "data", "train.csv"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(ROOT_DIR, "data", "model.cbm"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(DATASET_PATH)
    if df.empty:
        raise SystemExit("Dataset is empty. Run build_dataset.py after building search.db.")
    y = df["label"]
    X = df.drop(columns=["label"])

    cat_features = ["query_text", "item_id", "item_name", "item_category", "company_name"]

    pool = Pool(X, y, cat_features=cat_features)
    model = CatBoostClassifier(
        iterations=args.iters,
        depth=args.depth,
        learning_rate=args.lr,
        loss_function="Logloss",
        verbose=50,
    )
    model.fit(pool)
    model.save_model(MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
