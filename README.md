# Portfolio Look-Through (Updated)

This app computes look-through exposures for a portfolio of stocks + ETFs.

## What you need

1) Nasdaq Stock Screener CSV (Security Master)
- A default security master is bundled in the repo under `data/security_master/security_master.parquet`
- Optional: download a fresh Nasdaq Stock Screener CSV and upload it in the app to refresh
- Required columns: Symbol, Name, Country, Sector, Industry
- Stored locally under `data/security_master/`

2) Portfolio CSV
- Required: `ticker`
- Provide ONE of:
  - `dollars`
  - `shares` + `price_per_share`
  - `percent` (0..1) + Total portfolio value in the sidebar
- Optional: `is_etf` (True/False)
  - If missing, you can set ETF overrides in the app

3) Alpha Vantage API key (for ETF holdings)
- Used only for ETF holdings via function `ETF_PROFILE`
- Set it as an environment variable:

  export ALPHAVANTAGE_API_KEY="YOUR_KEY"

  or put it in Streamlit secrets as:
  ALPHAVANTAGE_API_KEY="YOUR_KEY"

ETF holdings snapshots are cached daily under `data/etf_holdings/<ETF>/holdings_YYYY-MM-DD.parquet`.

## Install & run

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
