# Portfolio Look-Through (Dash)

Interactive Dash app to compute look-through exposures for portfolios that mix stocks and ETFs. It normalizes inputs, expands ETF holdings (via Alpha Vantage), and enriches exposures with sector/country/industry from a Nasdaq security master.

## Quickstart

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app_dash.py

The app prints a local URL on startup.

## Inputs

### 1) Portfolio CSV
Required columns:
- `ticker`
- One of:
  - `dollars`
  - `shares` + `price_per_share`
  - `percent` (0..1) + total portfolio value in the UI

Optional ETF hints:
- `is_etf` (True/False)
- `asset_type` (ETF/Stock)

### 2) Security Master (Nasdaq screener)
A default security master is bundled at:
- `data/security_master/security_master.parquet`

You can optionally upload a fresh Nasdaq screener CSV in the app and click "Refresh security master" to regenerate:
- `data/security_master/security_master.parquet`
- `data/security_master/meta.json`
- `data/security_master/raw_*.csv` (audit trail)

Required columns for a refresh:
- `Symbol`, `Name`, `Country`, `Sector`, `Industry`

### 3) Alpha Vantage API key (ETF holdings)
ETF holdings are fetched from Alpha Vantage using the `ETF_PROFILE` endpoint. Set a key via:
- `ALPHAVANTAGE_API_KEY` environment variable, or
- `.streamlit/secrets.toml` with `ALPHAVANTAGE_API_KEY="..."` (the Dash app reads this file too)

Holdings are cached locally under:
- `data/etf_holdings/<ETF>/holdings_YYYY-MM-DD.parquet`

The UI lets you choose a refresh policy (missing, older_than, all) and an as-of date.

## What the app does

1) Normalize portfolio inputs to dollar exposure.
2) Classify ETF vs Stock (explicit column, overrides, or default stock).
3) Expand ETFs into constituent holdings and compute look-through exposure.
4) Enrich exposures with sector/country/industry from the security master.
5) Render dashboard KPIs, bar charts, and a drill table with cross-filtering.

## Local files created

- `data/security_master/` (default security master, optional refresh copies)
- `data/etf_holdings/` (ETF holdings cache)
- `data/overrides/asset_overrides.csv` (manual ETF/stock overrides saved by the app)
- `data/exports/` (reserved for export outputs)

## Notes

- If you have ETFs but no API key, the app can still run but ETF holdings will be missing.
- Unknown tickers are surfaced in the dashboard to help diagnose gaps in the security master.
