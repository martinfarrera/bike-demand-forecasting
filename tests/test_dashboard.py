import json

import pandas as pd
import pytest

from src.bike_dashboard import build_dashboard_data, render_dashboard


def sample_data():
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2011-01-01 00:00", "2011-01-01 01:00", "2011-07-02 12:00",
            "2012-10-01 00:00", "2012-10-02 08:00",
        ]),
        "hr": [0, 1, 12, 0, 8],
        "weekday": [5, 5, 6, 0, 1],
        "workingday": [0, 0, 0, 1, 1],
        "season": [1, 1, 3, 4, 4],
        "yr": [0, 0, 0, 1, 1],
        "mnth": [1, 1, 7, 10, 10],
        "holiday": [0, 0, 0, 0, 0],
        "weathersit": [1, 1, 1, 1, 1],
        "temp": [0.1, 0.2, 0.5, 0.1, 0.2],
        "atemp": [0.1, 0.2, 0.5, 0.1, 0.2],
        "hum": [0.5, 0.5, 0.5, 0.5, 0.5],
        "windspeed": [0.1, 0.1, 0.1, 0.1, 0.1],
        "cnt": [10, 20, 50, 30, 40],
    })


def test_dashboard_data_includes_patterns_and_model_comparison():
    payload = build_dashboard_data(sample_data())
    assert len(payload["hourly"]) == 24
    assert payload["hourly"][0]["mean"] == 20
    assert len(payload["weekdays"]) == 7
    assert payload["seasons"][2]["mean"] == 50
    assert payload["models"]["hour_workday_mean"]["mae"] == pytest.approx(25 / 3)
    assert payload["models"]["decision_tree"]["mae"] == pytest.approx(25 / 3)
    assert payload["holdout_rows"] == 2


def test_dashboard_is_self_contained_and_discloses_limitations():
    html = render_dashboard(sample_data())
    assert "Bike-sharing demand dashboard" in html
    assert "Observed historical weather" in html
    assert "hourly-mean-chart" in html
    embedded = html.split('<script id="dashboard-data" type="application/json">', 1)[1]
    embedded = embedded.split("</script>", 1)[0]
    assert json.loads(embedded)["holdout_rows"] == 2
    assert "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset" in html
