# Source data

The project downloads `hour.csv` from the UCI Bike Sharing dataset (ID 275):
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

The source also provides `day.csv`; this project uses the hourly observations,
which include rental counts, weather, calendar, and season fields. The dataset
contains aggregate counts, not individual trip records. It is licensed CC BY 4.0.

Run `python -m src.bike_data` from the project root to download the official ZIP
archive and extract only `hour.csv` here. Downloaded source files are excluded
from Git by `.gitignore`. Cite the source as: Fanaee-T, H. (2013). *Bike
Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.
