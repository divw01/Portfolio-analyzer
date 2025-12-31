# Portfolio Look-Through (Dash)

Beginner-friendly app that shows your portfolio's look-through exposures for stocks and ETFs. It expands ETF holdings using Alpha Vantage and enriches results with sector/country/industry from a Nasdaq security master.

## Quick start (first time)

You have two options to run this app. Either use the app fully hosted here (https://portfolio-analyzer-n23k.onrender.com), which will be slow but require no downloads, or run the app locally, as described below:

1) Install Python 3.10+ from https://www.python.org/downloads/
2) Open Terminal (macOS) or Command Prompt (Windows) in this folder.
3) Run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app_dash.py
```

The app prints a local URL (like `http://127.0.0.1:8050`). Open it in your browser.

## What you need to provide

### 1) Portfolio CSV
Create a CSV with a `ticker` column and one of these dollar inputs:
- `dollars`
- `shares` + `price_per_share`
- `percent` (0..1) + total portfolio value (entered in the app)

Optional ETF hints (useful if you already know):
- `is_etf` (True/False)
- `asset_type` (ETF/Stock)

You can upload the CSV or paste it directly into the app.

### 2) Security master (already included)
This repo already includes a default security master:
- `data/security_master/security_master.parquet`

If you want a fresh copy, you can download a new Nasdaq screener CSV here:
- https://www.nasdaq.com/market-activity/stocks/screener

Then upload it in the app and click "Refresh security master".

Required columns for a refresh:
- `Symbol`, `Name`, `Country`, `Sector`, `Industry`

### 3) Alpha Vantage API key (only needed for ETFs)
ETF holdings come from Alpha Vantage:
- Get a free API key here: https://www.alphavantage.co/support/#api-key

As a normal user, you can simply paste the key into the app UI (Step 2). It will be stored in your browser's local storage for convenience.

The environment variable / secrets file are only needed for non-interactive or deployed use cases (for example: running on Render, a server, or CI where there is no browser to paste the key):
- `ALPHAVANTAGE_API_KEY` environment variable, or
- `.streamlit/secrets.toml` with `ALPHAVANTAGE_API_KEY="..."` (Dash reads this file too)

## Where files are stored (on your machine)

Security master:
- `data/security_master/security_master.parquet` (the active master)
- `data/security_master/meta.json` (last refresh info)
- `data/security_master/raw_*.csv` (raw audit copies after refresh)

ETF holdings cache:
- `data/etf_holdings/<ETF>/holdings_YYYY-MM-DD.parquet`

ETF/stock overrides you save in the app:
- `data/overrides/asset_overrides.csv`

Exports (reserved for future features):
- `data/exports/`

## How to use the app (step-by-step)

When you open the app, the left side has 4 steps. Do them in order:

1) Portfolio
- Upload or paste your portfolio CSV.
- If you use `percent`, enter the total portfolio value so dollars can be computed.

2) ETF holdings
- If your portfolio has ETFs, paste your Alpha Vantage API key.
- Choose a refresh policy and an as-of date if you want to control caching.

3) Security master
- The default security master is already loaded.
- Optional: upload a fresh Nasdaq screener CSV and click "Refresh security master".

4) Run
- Click "Run look-through".
- The dashboard on the right will fill in with KPIs, charts, and a drill table.

How the dashboard works:
- Each chart is interactive. Clicking a bar filters the other charts and the table.
- Click the same bar again to clear that filter.
- Use "Clear all filters" at the top to reset everything.

## What the app does (behind the scenes)

1) Normalize your portfolio to dollar exposure.
2) Classify each holding as stock or ETF.
3) Expand ETFs into their constituents and compute look-through exposure.
4) Enrich results with sector/country/industry from the security master.
5) Render KPIs, charts, and a drill table with cross-filtering.

## Notes

- If you have ETFs but no API key, the app still runs, but ETF holdings will be missing.
- Unknown tickers are shown in the dashboard so you can refresh the security master or fix symbols.
