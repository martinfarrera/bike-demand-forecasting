"""Download, validate, and summarize the UCI Bike Sharing hourly data."""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
REQUIRED_COLUMNS = {
    "dteday", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
}


def download_hourly_data(destination: str | Path) -> Path:
    """Download UCI's archive and extract only hour.csv to destination."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "bike-demand-forecasting/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        member = next((name for name in bundle.namelist() if Path(name).name == "hour.csv"), None)
        if member is None:
            raise ValueError("UCI archive does not contain hour.csv")
        destination.write_bytes(bundle.read(member))
    return destination


def load_hourly_data(path: str | Path) -> pd.DataFrame:
    """Load hourly observations, validate the public schema, and add timestamp."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("hourly dataset must not be empty")

    frame["timestamp"] = pd.to_datetime(frame["dteday"], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(frame["hr"], errors="coerce"), unit="h"
    )
    if frame["timestamp"].isna().any():
        raise ValueError("dteday and hr must form valid timestamps")
    for column in ("casual", "registered", "cnt"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{column} must contain numeric values")
        if (frame[column] < 0).any():
            raise ValueError(f"{column} must be non-negative")
    if not (frame["casual"] + frame["registered"] == frame["cnt"]).all():
        raise ValueError("cnt must equal casual + registered")
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def summarize_by_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mean and observation count for each clock hour (0–23)."""
    if not {"timestamp", "cnt"}.issubset(frame.columns):
        raise ValueError("frame must contain timestamp and cnt columns")
    hourly = frame.assign(hour=pd.to_datetime(frame["timestamp"]).dt.hour).groupby("hour")["cnt"].agg(
        mean_rentals="mean", observations="count"
    )
    result = hourly.reindex(range(24)).rename_axis("hour").reset_index()
    result["observations"] = result["observations"].fillna(0).astype(int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and summarize UCI bike-sharing data")
    parser.add_argument("--data", type=Path, default=Path("data/raw/hour.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/hourly-summary.csv"))
    args = parser.parse_args()
    source = args.data if args.data.exists() else download_hourly_data(args.data)
    frame = load_hourly_data(source)
    summary = summarize_by_hour(frame)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(f"Validated {len(frame)} hourly observations; wrote {args.summary}")


if __name__ == "__main__":
    main()
