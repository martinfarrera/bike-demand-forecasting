import numpy as np
import pandas as pd
import pytest

from src.bike_model import FEATURES, chronological_split, evaluate, fit_predict


def observations():
    frame = pd.DataFrame({name: [0, 0, 0, 0] for name in FEATURES})
    frame["timestamp"] = pd.to_datetime([
        "2012-09-29", "2012-09-30", "2012-10-01", "2012-10-02"
    ])
    frame["cnt"] = [10.0, 30.0, 50.0, 70.0]
    return frame


def test_split_sorts_and_assigns_boundary_to_test():
    train, test = chronological_split(observations().iloc[::-1])
    assert train["cnt"].tolist() == [10, 30]
    assert test["cnt"].tolist() == [50, 70]
    assert train["timestamp"].max() < test["timestamp"].min()


@pytest.mark.parametrize("defect", ["duplicate", "invalid_time", "nan", "infinity", "negative", "missing"])
def test_split_rejects_invalid_data(defect):
    frame = observations()
    if defect == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif defect == "invalid_time":
        frame.loc[0, "timestamp"] = pd.NaT
    elif defect == "nan":
        frame.loc[0, "temp"] = np.nan
    elif defect == "infinity":
        frame.loc[0, "cnt"] = np.inf
    elif defect == "negative":
        frame.loc[0, "cnt"] = -1
    else:
        frame = frame.drop(columns="hr")
    with pytest.raises(ValueError):
        chronological_split(frame)


@pytest.mark.parametrize("cutoff", ["2010-01-01", "2013-01-01"])
def test_split_rejects_empty_partition(cutoff):
    with pytest.raises(ValueError, match="nonempty"):
        chronological_split(observations(), cutoff)


def test_predictions_do_not_use_test_targets_or_target_components():
    train, test = chronological_split(observations())
    expected = fit_predict(train, test)
    train = train.assign(casual=train["cnt"], registered=0, instant=range(len(train)))
    test = test.drop(columns="cnt").assign(casual=999999, registered=999999, instant=-1)
    actual = fit_predict(train, test)
    assert not {"cnt", "casual", "registered", "instant", "timestamp"}.intersection(FEATURES)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_array_equal(actual[name], [20, 20])


def test_baseline_uses_group_mean_and_global_fallback():
    train, test = chronological_split(observations())
    train.loc[1, "hr"] = 1
    test.loc[test.index[1], "hr"] = 23
    predictions = fit_predict(train, test)
    np.testing.assert_array_equal(predictions["hour_workday_mean"], [10, 20])


def test_metrics_are_per_observed_hour_in_count_units():
    result = evaluate(observations())
    assert result["split"]["train"]["rows"] == 2
    assert result["split"]["test"]["rows"] == 2
    for metrics in result["metrics"].values():
        assert metrics["mae"] == 40
        assert metrics["rmse"] == pytest.approx(np.sqrt(1700))
