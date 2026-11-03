"""Build a self-contained static dashboard from validated local bike data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.bike_data import load_hourly_data
from src.bike_model import chronological_split, fit_predict

SEASONS = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
WEEKDAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")


def _group_means(frame: pd.DataFrame, key: str, labels: list[str]) -> list[dict]:
    means = frame.groupby(key)["cnt"].mean()
    first_index = 1 if key == "season" else 0
    return [
        {"label": label, "mean": float(means.get(index, 0.0))}
        for index, label in enumerate(labels, start=first_index)
    ]


def build_dashboard_data(frame: pd.DataFrame) -> dict:
    """Aggregate demand patterns and calculate holdout model metrics."""
    train, test = chronological_split(frame)
    predictions = fit_predict(train, test)
    metrics = {}
    for name, values in predictions.items():
        errors = test["cnt"].to_numpy() - values
        metrics[name] = {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean())),
        }

    ordered = frame.sort_values("timestamp")
    hourly = _group_means(ordered, "hr", [str(hour) for hour in range(24)])
    weekdays = _group_means(ordered, "weekday", list(WEEKDAYS))
    seasons = _group_means(ordered, "season", [SEASONS[index] for index in range(1, 5)])
    return {
        "hourly": hourly,
        "weekdays": weekdays,
        "seasons": seasons,
        "models": metrics,
        "holdout_rows": len(test),
        "holdout_start": test["timestamp"].min().strftime("%Y-%m-%d"),
        "holdout_end": test["timestamp"].max().strftime("%Y-%m-%d"),
        "cutoff": "2012-10-01",
    }


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bike-sharing demand dashboard</title>
<style>
:root{color-scheme:light;--ink:#18312b;--muted:#536b64;--accent:#147d64;--paper:#f4f7f4;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:auto;padding:2rem 1.25rem}h1,h2{line-height:1.2}h1{font-size:clamp(2rem,5vw,3.4rem);margin:.3rem 0}h2{font-size:1.25rem}.eyebrow,.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1rem;margin:1.4rem 0}
section,.metric,.note{background:var(--card);border-radius:14px;padding:1.1rem;box-shadow:0 2px 12px #16382a0d}.chart{width:100%;height:230px;overflow:visible}.bars{display:flex;align-items:end;gap:3px;height:170px;border-bottom:1px solid #cad7d0;padding-top:10px}.bar{flex:1;min-width:3px;background:var(--accent);border-radius:3px 3px 0 0;position:relative}.bar:hover{background:#e28b35}.axis{display:flex;justify-content:space-between;color:var(--muted);font-size:.75rem;margin-top:.35rem}.metric-row{display:flex;gap:.8rem;flex-wrap:wrap}.metric{flex:1;min-width:180px}.value{font-size:1.8rem;font-weight:700}.note{border-left:5px solid #e28b35}.table{width:100%;border-collapse:collapse}.table td,.table th{text-align:left;padding:.5rem;border-bottom:1px solid #e5ece7}footer{margin-top:1.5rem;color:var(--muted);font-size:.9rem}
</style></head><body><main>
<p class="eyebrow">UCI BIKE SHARING · HISTORICAL DATA</p><h1>Bike-sharing demand dashboard</h1>
<p class="muted">Explore observed hourly rental patterns and a chronological model comparison.</p>
<div class="metric-row" id="metrics"></div>
<div class="grid"><section><h2>Average rentals by hour</h2><div id="hourly-mean-chart" class="chart"></div><p class="muted">Mean rentals per observed hour across the dataset.</p></section>
<section><h2>Average rentals by weekday</h2><div id="weekday-chart" class="chart"></div><p class="muted">UCI weekday codes: Sunday (0) through Saturday (6).</p></section>
<section><h2>Average rentals by season</h2><div id="season-chart" class="chart"></div><p class="muted">Winter, spring, summer and fall.</p></section>
<section><h2>Holdout model comparison</h2><table class="table"><thead><tr><th>Method</th><th>MAE</th><th>RMSE</th></tr></thead><tbody id="model-table"></tbody></table><p class="muted">Errors are rentals per observed hour; lower is better.</p></section></div>
<aside class="note"><h2>How to interpret this</h2><p>Observed historical weather is a limitation: the decision tree uses weather measurements from each evaluated hour. That observed historical weather would not generally be known when making a real forecast. These results are retrospective and do not demonstrate operational future-weather forecasting performance.</p><p>The evaluation trains before <strong>2012-10-01</strong> and tests on the later observed quarter. Missing hours are not filled with zero demand. The data represent one bike-sharing system in 2011–2012.</p></aside>
<footer>Source: <a href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset">UCI Bike Sharing dataset</a>, CC BY 4.0. Aggregates and scores are embedded in this static page; raw data are not included.</footer>
<script id="dashboard-data" type="application/json">__DATA__</script>
<script>
const d=JSON.parse(document.getElementById('dashboard-data').textContent);
function chart(id,rows){const el=document.getElementById(id),max=Math.max(...rows.map(x=>x.mean),1);const bars=document.createElement('div');bars.className='bars';rows.forEach(x=>{const b=document.createElement('div');b.className='bar';b.style.height=Math.max(2,x.mean/max*100)+'%';b.title=x.label+': '+x.mean.toFixed(1)+' rentals';bars.appendChild(b)});const axis=document.createElement('div');axis.className='axis';axis.innerHTML='<span>'+rows[0].label+'</span><span>'+rows[Math.floor(rows.length/2)].label+'</span><span>'+rows.at(-1).label+'</span>';el.append(bars,axis)}
chart('hourly-mean-chart',d.hourly);chart('weekday-chart',d.weekdays);chart('season-chart',d.seasons);
const metrics=document.getElementById('metrics');[['Holdout period',d.holdout_start+' to '+d.holdout_end],['Observed holdout hours',d.holdout_rows.toLocaleString()]].forEach(([label,value])=>{const card=document.createElement('div');card.className='metric';card.innerHTML='<div class="muted">'+label+'</div><div class="value">'+value+'</div>';metrics.appendChild(card)});
const names={hour_workday_mean:'Hour/workday mean',decision_tree:'Decision tree'};document.getElementById('model-table').innerHTML=Object.entries(d.models).map(([key,m])=>'<tr><td>'+names[key]+'</td><td>'+m.mae.toFixed(2)+'</td><td>'+m.rmse.toFixed(2)+'</td></tr>').join('');
</script></main></body></html>'''


def render_dashboard(frame: pd.DataFrame) -> str:
    """Render a self-contained HTML dashboard; does not write raw data."""
    payload = json.dumps(build_dashboard_data(frame), separators=(",", ":"), allow_nan=False)
    # Prevent embedded data from terminating its JSON script element.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return _PAGE.replace("__DATA__", payload)


def write_dashboard(frame: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(frame), encoding="utf-8")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/hour.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/dashboard.html"))
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"local data file not found: {args.data}; no download is attempted")
    write_dashboard(load_hourly_data(args.data), args.output)
    print(f"Wrote static dashboard to {args.output}")


if __name__ == "__main__":
    main()
''