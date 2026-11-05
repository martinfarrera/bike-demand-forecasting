"""Compare baseline and tree variants on a chronological holdout; print JSON only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src.bike_data import load_hourly_data

CUTOFF = "2012-10-01"
CALENDAR_FEATURES = (
    "hr", "weekday", "mnth", "yr", "holiday", "workingday",
)
FEATURES = (
    *CALENDAR_FEATURES, "weathersit", "temp", "atemp", "hum", "windspeed",
)


def chronological_split(frame: pd.DataFrame, cutoff: str = CUTOFF):
    """Validate and split unique observed hours; never shuffle or fill missing hours."""
    required = {"timestamp", "cnt", *FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing evaluation columns: {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise ValueError("timestamps must be valid and unique")
    numeric = frame.loc[:, [*FEATURES, "cnt"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric["cnt"] < 0).any():
        raise ValueError("features and target must be finite numeric values; target must be non-negative")
    frame[numeric.columns] = numeric
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    boundary = pd.Timestamp(cutoff)
    if pd.isna(boundary):
        raise ValueError("cutoff must be a valid timestamp")
    train = frame.loc[frame["timestamp"] < boundary].copy()
    test = frame.loc[frame["timestamp"] >= boundary].copy()
    if train.empty or test.empty:
        raise ValueError("chronological split requires nonempty training and test sets")
    return train, test


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit on training labels only. Test labels are never read here."""
    means = train.groupby(["hr", "workingday"])["cnt"].mean()
    keys = pd.MultiIndex.from_frame(test[["hr", "workingday"]])
    baseline = means.reindex(keys).fillna(train["cnt"].mean()).to_numpy()
    calendar_model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=0)
    calendar_model.fit(train.loc[:, CALENDAR_FEATURES], train["cnt"])
    weather_model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=0)
    weather_model.fit(train.loc[:, FEATURES], train["cnt"])
    return {
        "hour_workday_mean": baseline,
        "calendar_only_decision_tree": calendar_model.predict(test.loc[:, CALENDAR_FEATURES]),
        "decision_tree": weather_model.predict(test.loc[:, FEATURES]),
    }


def evaluate(frame: pd.DataFrame, cutoff: str = CUTOFF) -> dict:
    train, test = chronological_split(frame, cutoff)
    predictions = fit_predict(train, test)
    metrics = {}
    for name, prediction in predictions.items():
        error = test["cnt"].to_numpy() - prediction
        metrics[name] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
        }
    return {
        "cutoff": cutoff,
        "split": {
            name: {
                "rows": len(part),
                "start": part["timestamp"].min().isoformat(),
                "end": part["timestamp"].max().isoformat(),
            }
            for name, part in (("train", train), ("test", test))
        },
        "features": list(FEATURES),
        "calendar_features": list(CALENDAR_FEATURES),
        "tree_parameters": {"max_depth": 8, "min_samples_leaf": 20, "random_state": 0},
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/hour.csv"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"local data file not found: {args.data}; no download is attempted")
    print(json.dumps(evaluate(load_hourly_data(args.data)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
