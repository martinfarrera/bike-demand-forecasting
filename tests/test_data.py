from pathlib import Path

import pandas as pd
import pytest

from src.bike_data import load_hourly_data, summarize_by_hour


def sample_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "instant": [1, 2, 3],
            "dteday": ["2011-01-01"] * 3,
            "season": [1, 1, 1],
            "yr": [0, 0, 0],
            "mnth": [1, 1, 1],
            "hr": [6, 7, 6],
            "holiday": [0, 0, 0],
            "weekday": [6, 6, 6],
            "workingday": [0, 0, 0],
            "weathersit": [1, 2, 1],
            "temp": [0.2, 0.3, 0.2],
            "atemp": [0.2, 0.3, 0.2],
            "hum": [0.5, 0.6, 0.5],
            "windspeed": [0.1, 0.1, 0.1],
            "casual": [1, 2, 3],
            "registered": [4, 5, 6],
            "cnt": [5, 7, 9],
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_loader_parses_timestamps_and_preserves_target(tmp_path):
    frame = load_hourly_data(sample_csv(tmp_path / "hour.csv"))

    assert frame["timestamp"].tolist() == list(pd.to_datetime([
        "2011-01-01 06:00", "2011-01-01 06:00", "2011-01-01 07:00"
    ]))
    assert frame["cnt"].tolist() == [5, 9, 7]


def test_loader_rejects_negative_target(tmp_path):
    path = sample_csv(tmp_path / "hour.csv")
    frame = pd.read_csv(path)
    frame.loc[0, "cnt"] = -1
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="cnt must be non-negative"):
        load_hourly_data(path)


def test_hourly_summary_aggregates_counts_and_keeps_all_hours():
    result = summarize_by_hour(
        pd.DataFrame({"timestamp": pd.to_datetime(["2011-01-01 06:00", "2011-01-02 06:00"]), "cnt": [5, 9]})
    )

    assert len(result) == 24
    assert result.loc[result["hour"] == 6, "mean_rentals"].item() == 7
