# Bike Demand Forecasting

A reproducible, aggregate-focused exploration of hourly bike-sharing demand using the [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (CC BY 4.0). The analysis compares an hour/workday historical-mean baseline with a small decision tree using a chronological holdout. It is retrospective analysis, not a production forecasting service.

## Dashboard

With Python dependencies from `requirements.txt` installed and the UCI `hour.csv` available locally at `data/raw/hour.csv`, run from the project root:

```sh
python -m src.bike_dashboard
```

Open `reports/dashboard.html` in a browser. The self-contained page shows average rentals by hour, weekday, and season, compares hourly demand on workdays versus non-working days (weekends and holidays), and reports MAE/RMSE for the holdout models. It embeds only aggregate chart data and evaluation scores; raw data remain local and ignored by Git. The page explains the observed-weather limitation and chronological split.

To choose paths explicitly:

```sh
python -m src.bike_dashboard --data data/raw/hour.csv --output reports/dashboard.html
```

The command never downloads data. If the local input is absent, it exits with an error. See [`reports/model-results.md`](reports/model-results.md) for evaluation details and caveats.
