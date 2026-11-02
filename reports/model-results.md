# Chronological bike-demand evaluation

A modest decision tree beat a train-only hour/workday mean baseline on the final three months of the local UCI hourly data: MAE fell from **79.95 to 70.20 rentals per observed hour** (about 12.2%). This is a retrospective, observed-weather comparison—not evidence of operational future-weather forecasting performance.

## Actual results

Target: `cnt`, the total rentals recorded in each hourly observation. The local `hour.csv` contains 17,379 records, all used once in the split below.

| Partition | First observed timestamp | Last observed timestamp | Rows |
|---|---|---|---:|
| Training | 2011-01-01 00:00 | 2012-09-30 23:00 | 15,211 |
| Holdout | 2012-10-01 00:00 | 2012-12-31 23:00 | 2,168 |

| Method | MAE (rentals/hour) | RMSE (rentals/hour) |
|---|---:|---:|
| Hour/workday training mean | 79.950435 | 119.589266 |
| Decision tree | 70.195990 | 108.350930 |

MAE is mean absolute prediction error; RMSE is the square root of mean squared prediction error. Every observed holdout row has equal weight. These are hourly count errors, not errors on daily aggregates or percentages.

## Exact evaluation semantics

- Sort by the dataset's date plus hour, interpreted as timezone-naive source timestamps. Train on timestamps strictly before **2012-10-01**; test on timestamps at or after it. Reject missing/invalid or duplicate timestamps, nonfinite model inputs/targets, negative targets, and empty partitions.
- Keep missing hours absent: do not interpolate, manufacture zero-demand observations, or randomly split rows. The holdout has 2,168 observed hours, rather than all 2,208 calendar hours in that quarter.
- Fit the baseline using training `cnt` means grouped by `hr` and `workingday`. For an unseen group, use the overall training mean. No holdout labels enter baseline fitting.
- Fit one `DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=0)` on training data only. Parameters and cutoff were fixed before this evaluation; there is no hyperparameter search, validation-based selection, scaling, imputation, or subsequent refit on holdout labels.
- Use only `hr`, `weekday`, `mnth`, `yr`, `holiday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, and `windspeed`. Numeric codes are passed directly to the tree, not one-hot encoded. Weather inputs retain UCI's supplied normalization.
- Exclude `cnt` and its exact components `casual` and `registered` from predictors, as well as row identifier `instant` and raw timestamps. Test targets are accessed only for scoring. No target lags or rolling demand statistics are used.
- Both predictors stay frozen throughout the holdout. This is one chronological holdout, not rolling-origin retraining, a recursive multi-step forecast, or a specified lead-time forecast.

## Limitations

**Weather availability is the main boundary.** The tree receives weather measurements from each evaluated historical hour. These observations would not generally be known at a forecast issuance time. Calendar features can be known in advance, but future weather would require forecasts with their own errors. The baseline uses only calendar information, so this comparison does not isolate model capacity from the added information available to the tree. A calendar-only tree and forecast-weather evaluation would be useful follow-ups, not claims established here.

The data cover one historical bike-sharing system during 2011–2012. Demand growth, seasonal variation, missing hours, extreme weather, and later operating conditions limit generalization. Only the last quarter is tested; no cross-season backtesting, uncertainty intervals, subgroup error analysis, or statistical significance claims are provided. Missing hours are not assumed to mean zero rentals. Count labels measure observed rentals rather than unconstrained demand or unmet requests. Numeric category ordering is a simplification of this small tree.

There was no holdout-driven parameter tuning in this implementation. Now that these results have been inspected, future tuning should use earlier chronological validation windows and reserve a new untouched evaluation period where possible.

## Reproduce locally

From the project root, with the existing dependencies installed and the already-downloaded UCI file at `data/raw/hour.csv`:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tests/test_model.py
PYTHONDONTWRITEBYTECODE=1 python -m src.bike_model --data data/raw/hour.csv
```

The model command reads local data and prints JSON only. It neither downloads data nor writes results; a missing input file is an explicit error. Raw data remain ignored by Git; this report is explicitly allowed through the reports ignore rule. Dependency versions are not pinned, so exact cross-version reproducibility is not guaranteed.

Focused validation: **12 tests passed**, covering chronological sorting and cutoff assignment, invalid input rejection, empty partitions, training-only predictions without test targets, exclusion of target components, baseline fallback, and metric arithmetic.

Source and attribution: [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), CC BY 4.0; see [raw-data provenance](../data/raw/README.md). This implements T2's S1–S2 model-comparison work only; the static dashboard and final project-wide checks remain T3–T4.
