import re
import json
import time
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import requests


# =========================
# Config + Paths
# =========================

DATA_DIR = Path("data")
SECURITY_MASTER_DIR = DATA_DIR / "security_master"
ETF_HOLDINGS_DIR = DATA_DIR / "etf_holdings"
OVERRIDES_DIR = DATA_DIR / "overrides"
EXPORTS_DIR = DATA_DIR / "exports"

SECURITY_MASTER_PARQUET = SECURITY_MASTER_DIR / "security_master.parquet"
SECURITY_MASTER_META = SECURITY_MASTER_DIR / "meta.json"
ASSET_OVERRIDES_CSV = OVERRIDES_DIR / "asset_overrides.csv"


def ensure_dirs() -> None:
    SECURITY_MASTER_DIR.mkdir(parents=True, exist_ok=True)
    ETF_HOLDINGS_DIR.mkdir(parents=True, exist_ok=True)
    OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def today_str() -> str:
    return dt.date.today().isoformat()


def symbol_norm(symbol: str) -> str:
    if symbol is None:
        return ""
    s = str(symbol).upper().strip()
    s = s.replace(".", "-").replace("/", "-")
    s = re.sub(r"-{2,}", "-", s)
    return s


def safe_float(x) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        if isinstance(x, str):
            x = x.replace("$", "").replace(",", "").strip()
        return float(x)
    except Exception:
        return None


def parse_market_cap(x) -> Optional[float]:
    return safe_float(x)


def parse_last_sale(x) -> Optional[float]:
    return safe_float(x)


# =========================
# Alpha Vantage client
# =========================

AV_BASE_URL = "https://www.alphavantage.co/query"


def alpha_vantage_get(
    api_key: str,
    params: Dict[str, Any],
    timeout: int = 30,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Alpha Vantage GET wrapper with basic retry/backoff.

    Raises RuntimeError on rate limit / API errors.
    """
    if not api_key:
        raise ValueError("Alpha Vantage API key missing. Set ALPHAVANTAGE_API_KEY in Streamlit secrets or env.")

    params = dict(params)
    params["apikey"] = api_key

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(AV_BASE_URL, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            if isinstance(data, dict) and ("Note" in data or "Information" in data or "Error Message" in data):
                raise RuntimeError(f"Alpha Vantage response indicates an issue: {data}")

            return data
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)
            else:
                raise RuntimeError(f"Alpha Vantage request failed after {max_retries} retries: {e}") from e

    raise RuntimeError(f"Alpha Vantage request failed: {last_err}")


# =========================
# Overrides (Stock vs ETF)
# =========================

def load_asset_overrides() -> pd.DataFrame:
    ensure_dirs()
    if ASSET_OVERRIDES_CSV.exists():
        df = pd.read_csv(ASSET_OVERRIDES_CSV)
        if "ticker_norm" not in df.columns and "ticker" in df.columns:
            df["ticker_norm"] = df["ticker"].map(symbol_norm)
        if "is_etf" not in df.columns:
            df["is_etf"] = False

        df["ticker_norm"] = df["ticker_norm"].astype(str).map(symbol_norm)
        df["is_etf"] = df["is_etf"].astype(bool)
        df = df.drop_duplicates(subset=["ticker_norm"], keep="last")
        return df[["ticker_norm", "is_etf"]]

    return pd.DataFrame(columns=["ticker_norm", "is_etf"])


def save_asset_overrides(overrides_df: pd.DataFrame) -> None:
    ensure_dirs()
    out = overrides_df.copy()
    if "ticker_norm" not in out.columns:
        raise ValueError("Overrides must include ticker_norm")
    if "is_etf" not in out.columns:
        raise ValueError("Overrides must include is_etf")

    out["ticker_norm"] = out["ticker_norm"].astype(str).map(symbol_norm)
    out["is_etf"] = out["is_etf"].astype(bool)
    out = out.drop_duplicates(subset=["ticker_norm"], keep="last")
    out.to_csv(ASSET_OVERRIDES_CSV, index=False)


# =========================
# Security Master (Nasdaq Screener CSV)
# =========================

REQUIRED_MASTER_COLS = {"Symbol", "Name", "Country", "Sector", "Industry"}


def load_security_master() -> pd.DataFrame:
    ensure_dirs()
    if SECURITY_MASTER_PARQUET.exists():
        df = pd.read_parquet(SECURITY_MASTER_PARQUET)
        if "symbol_norm" not in df.columns and "Symbol" in df.columns:
            df["symbol_norm"] = df["Symbol"].map(symbol_norm)
        return df
    return pd.DataFrame(columns=["Symbol", "Name", "Country", "Sector", "Industry", "symbol_norm"])


def security_master_meta() -> Optional[Dict[str, Any]]:
    if SECURITY_MASTER_META.exists():
        try:
            return json.loads(SECURITY_MASTER_META.read_text())
        except Exception:
            return None
    return None


def refresh_security_master_from_csv_bytes(file_bytes: bytes, filename: str = "nasdaq_screener.csv") -> pd.DataFrame:
    """
    Reads the Nasdaq Stock Screener CSV bytes and persists:
      - raw timestamped copy under data/security_master/raw_*.csv
      - cleaned parquet under data/security_master/security_master.parquet
      - meta.json

    Best practice: keep raw for audit/rollback.
    """
    ensure_dirs()
    raw_copy = SECURITY_MASTER_DIR / f"raw_{now_ts()}_{Path(filename).name}"
    raw_copy.write_bytes(file_bytes)

    df = pd.read_csv(raw_copy)

    missing = REQUIRED_MASTER_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Security master CSV missing required columns: {sorted(list(missing))}")

    keep_cols = [c for c in ["Symbol", "Name", "Last Sale", "Market Cap", "Country", "IPO Year", "Volume", "Sector", "Industry"] if c in df.columns]
    df = df[keep_cols].copy()

    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["symbol_norm"] = df["Symbol"].map(symbol_norm)

    if "Last Sale" in df.columns:
        df["last_sale_num"] = df["Last Sale"].map(parse_last_sale)
    if "Market Cap" in df.columns:
        df["market_cap_num"] = df["Market Cap"].map(parse_market_cap)

    df = df[df["symbol_norm"] != ""].copy()

    if "market_cap_num" in df.columns:
        df = df.sort_values(["symbol_norm", "market_cap_num"], ascending=[True, False])
    df = df.drop_duplicates(subset=["symbol_norm"], keep="first")

    df.to_parquet(SECURITY_MASTER_PARQUET, index=False)

    meta = {
        "uploaded_at": dt.datetime.now().isoformat(),
        "raw_copy": str(raw_copy),
        "row_count": int(len(df)),
        "columns": list(df.columns),
    }
    SECURITY_MASTER_META.write_text(json.dumps(meta, indent=2))

    return df


# =========================
# ETF Holdings Cache
# =========================

def holdings_cache_path(etf_symbol: str, asof_date: Optional[str] = None) -> Path:
    ensure_dirs()
    etf = symbol_norm(etf_symbol)
    d = asof_date or today_str()
    folder = ETF_HOLDINGS_DIR / etf
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"holdings_{d}.parquet"


def load_cached_etf_holdings(etf_symbol: str, asof_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    path = holdings_cache_path(etf_symbol, asof_date)
    if path.exists():
        return pd.read_parquet(path)
    return None


def latest_holdings_cache_date(etf_symbol: str) -> Optional[str]:
    etf = symbol_norm(etf_symbol)
    folder = ETF_HOLDINGS_DIR / etf
    if not folder.exists():
        return None
    dates = []
    for p in folder.glob("holdings_*.parquet"):
        name = p.stem.replace("holdings_", "")
        try:
            d = dt.date.fromisoformat(name)
        except Exception:
            continue
        dates.append(d)
    if not dates:
        return None
    return max(dates).isoformat()


def load_latest_cached_etf_holdings(etf_symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    latest = latest_holdings_cache_date(etf_symbol)
    if not latest:
        return None, None
    try:
        return load_cached_etf_holdings(etf_symbol, latest), latest
    except Exception:
        return None, None


def refresh_etf_holdings(
    api_key: str,
    etf_symbol: str,
    asof_date: Optional[str] = None,
    sleep_seconds: float = 0.0
) -> pd.DataFrame:
    """
    Fetch ETF holdings from Alpha Vantage ETF_PROFILE and cache daily snapshot.
    """
    etf_symbol = symbol_norm(etf_symbol)
    d = asof_date or today_str()

    data = alpha_vantage_get(api_key, {"function": "ETF_PROFILE", "symbol": etf_symbol})

    holdings = None
    for k in ["holdings", "Holdings", "constituents", "Constituents"]:
        if k in data and isinstance(data[k], list):
            holdings = data[k]
            break

    if holdings is None:
        for v in data.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                keys = {kk.lower() for kk in v[0].keys()}
                if any(k in keys for k in ["symbol", "ticker", "holding", "asset", "constituent"]) and any(k in keys for k in ["weight", "allocation", "percentage", "pct"]):
                    holdings = v
                    break

    if holdings is None:
        raise ValueError(f"Could not locate holdings list in ETF_PROFILE response for {etf_symbol}. Keys: {list(data.keys())}")

    hdf = pd.DataFrame(holdings).copy()

    lower_cols = {c: c.lower() for c in hdf.columns}
    sym_candidates = [c for c in hdf.columns if lower_cols[c] in ["symbol", "ticker", "holding", "asset", "constituent"]]
    wt_candidates = [c for c in hdf.columns if any(x in lower_cols[c] for x in ["weight", "allocation", "percentage", "pct"])]

    if not sym_candidates:
        sym_candidates = [c for c in hdf.columns if "sym" in lower_cols[c]]
    if not wt_candidates:
        wt_candidates = [c for c in hdf.columns if "weight" in lower_cols[c] or "alloc" in lower_cols[c] or "pct" in lower_cols[c]]

    if not sym_candidates or not wt_candidates:
        raise ValueError(f"Could not infer holdings symbol/weight columns for {etf_symbol}. Columns: {list(hdf.columns)}")

    sym_col = sym_candidates[0]
    wt_col = wt_candidates[0]

    out = pd.DataFrame({
        "etf_symbol": etf_symbol,
        "constituent_symbol_raw": hdf[sym_col].astype(str).str.strip(),
        "constituent_symbol_norm": hdf[sym_col].astype(str).map(symbol_norm),
        "weight_raw": hdf[wt_col],
    })

    out["weight"] = out["weight_raw"].map(safe_float)
    if out["weight"].notna().any():
        if out["weight"].dropna().median() > 1.0:
            out["weight"] = out["weight"] / 100.0

    out = out.dropna(subset=["constituent_symbol_norm", "weight"])
    out = out[out["constituent_symbol_norm"] != ""].copy()

    path = holdings_cache_path(etf_symbol, d)
    out.to_parquet(path, index=False)

    if sleep_seconds > 0:
        time.sleep(float(sleep_seconds))

    return out


# =========================
# Portfolio normalization + look-through
# =========================

def normalize_portfolio_inputs(df: pd.DataFrame, total_portfolio_value: Optional[float] = None) -> pd.DataFrame:
    """
    Supported inputs:
      - ticker (required)
      - percent (optional; 0..1)
      - shares (optional)
      - price_per_share (optional)
      - dollars (optional)

    Compute position_value:
      - dollars if provided
      - else shares * price_per_share if available
      - else percent * total_portfolio_value if provided
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("Portfolio input must contain a 'ticker' column.")

    # Ensure optional columns exist to keep downstream validation and error reporting robust.
    optional_cols = ["percent", "shares", "price_per_share", "dollars"]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df["ticker_raw"] = df["ticker"].astype(str)
    df["ticker_norm"] = df["ticker_raw"].map(symbol_norm)

    # Drop rows that are completely empty (e.g., trailing blank line in pasted CSV).
    df = df.dropna(how="all")

    def compute_value(row):
        if pd.notna(row.get("dollars")):
            return float(row["dollars"])
        if pd.notna(row.get("shares")) and pd.notna(row.get("price_per_share")):
            return float(row["shares"]) * float(row["price_per_share"])
        if pd.notna(row.get("percent")):
            if total_portfolio_value is None or total_portfolio_value <= 0:
                raise ValueError("total_portfolio_value must be provided (>0) if any row uses 'percent'.")
            return float(row["percent"]) * float(total_portfolio_value)
        return None

    df["position_value"] = df.apply(compute_value, axis=1)

    bad = df[df["position_value"].isna()]
    if len(bad) > 0:
        raise ValueError(
            "Some rows are missing enough info to compute dollars exposure. "
            "Provide dollars OR shares+price_per_share OR percent+total_portfolio_value.\n"
            f"{bad[['ticker_raw','percent','shares','price_per_share','dollars']].to_string(index=False)}"
        )

    keep = ["ticker_raw", "ticker_norm", "percent", "shares", "price_per_share", "dollars", "position_value"]
    if "is_etf" in df.columns:
        keep.append("is_etf")

    return df[keep]


def apply_etf_classification(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines asset_type based on:
    1) explicit portfolio_df.is_etf if present, else
    2) saved overrides in data/overrides/asset_overrides.csv, else
    3) default False
    """
    df = portfolio_df.copy()
    overrides = load_asset_overrides()

    if "is_etf" not in df.columns:
        df["is_etf"] = pd.NA

    df = df.merge(overrides, on="ticker_norm", how="left", suffixes=("", "_override"))

    # priority: explicit -> override -> False
    df["is_etf_final"] = df["is_etf"]
    df.loc[df["is_etf_final"].isna(), "is_etf_final"] = df.loc[df["is_etf_final"].isna(), "is_etf_override"]
    df["is_etf_final"] = df["is_etf_final"].fillna(False).astype(bool)

    df["asset_type"] = df["is_etf_final"].map(lambda x: "ETF" if bool(x) else "Stock")

    drop_cols = [c for c in ["is_etf_override"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    return df


def enrich_with_security_master(df: pd.DataFrame, master_df: pd.DataFrame, join_col: str) -> pd.DataFrame:
    master_cols = ["symbol_norm", "Name", "Country", "Sector", "Industry"]
    master_min = master_df[master_cols].drop_duplicates("symbol_norm") if len(master_df) else pd.DataFrame(columns=master_cols)
    out = df.merge(master_min, left_on=join_col, right_on="symbol_norm", how="left")
    out = out.drop(columns=["symbol_norm"])
    return out


def build_lookthrough_exposures(
    api_key: str,
    portfolio_df: pd.DataFrame,
    master_df: pd.DataFrame,
    refresh_missing_etf_holdings: bool = True,
    asof_date: Optional[str] = None,
    refresh_policy: str = "missing",
    refresh_cutoff_date: Optional[str] = None,
    per_etf_sleep_seconds: float = 0.0,
    progress_cb=None,   # optional callback(str message, float progress 0..1)
) -> pd.DataFrame:
    rows = []
    d = asof_date or today_str()
    cutoff = refresh_cutoff_date or d

    n = len(portfolio_df)
    for i, (_, r) in enumerate(portfolio_df.iterrows(), start=1):
        src = r["ticker_norm"]
        src_type = r["asset_type"]
        pv = float(r["position_value"])

        if progress_cb:
            progress_cb(f"Processing {src} ({i}/{n}) ...", i / max(n, 1))

        if src_type == "Stock":
            rows.append({
                "source_ticker_norm": src,
                "source_type": src_type,
                "underlying_symbol_norm": src,
                "exposure_value": pv,
            })
        else:
            cached = load_cached_etf_holdings(src, d)
            latest_cached, latest_date = load_latest_cached_etf_holdings(src)

            if refresh_policy == "all":
                needs_refresh = True
            elif refresh_policy == "older_than":
                if latest_date is None:
                    needs_refresh = True
                else:
                    try:
                        needs_refresh = dt.date.fromisoformat(latest_date) < dt.date.fromisoformat(cutoff)
                    except Exception:
                        needs_refresh = cached is None
            else:
                needs_refresh = latest_date is None

            if needs_refresh and refresh_missing_etf_holdings:
                cached = refresh_etf_holdings(api_key, src, d, sleep_seconds=per_etf_sleep_seconds)
            elif cached is None and latest_cached is not None:
                cached = latest_cached

            if cached is None or len(cached) == 0:
                rows.append({
                    "source_ticker_norm": src,
                    "source_type": src_type,
                    "underlying_symbol_norm": None,
                    "exposure_value": pv,
                })
            else:
                for _, h in cached.iterrows():
                    rows.append({
                        "source_ticker_norm": src,
                        "source_type": src_type,
                        "underlying_symbol_norm": h["constituent_symbol_norm"],
                        "exposure_value": pv * float(h["weight"]),
                    })

    exp = pd.DataFrame(rows)
    exp = enrich_with_security_master(exp, master_df, join_col="underlying_symbol_norm")

    exp["company_name"] = exp.get("Name")
    exp["country"] = exp.get("Country")
    exp["sector"] = exp.get("Sector")
    exp["industry"] = exp.get("Industry")

    return exp


def build_slices(exposures: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_company = (
        exposures.dropna(subset=["underlying_symbol_norm"])
        .groupby(["underlying_symbol_norm", "company_name"], dropna=False)
        .agg(total_exposure=("exposure_value", "sum"))
        .sort_values("total_exposure", ascending=False)
        .reset_index()
    )

    by_sector = (
        exposures.groupby("sector", dropna=False)
        .agg(total_exposure=("exposure_value", "sum"))
        .sort_values("total_exposure", ascending=False)
        .reset_index()
    )

    by_country = (
        exposures.groupby("country", dropna=False)
        .agg(total_exposure=("exposure_value", "sum"))
        .sort_values("total_exposure", ascending=False)
        .reset_index()
    )

    by_source_vehicle = (
        exposures.groupby(["source_ticker_norm", "source_type"], dropna=False)
        .agg(total_exposure=("exposure_value", "sum"))
        .sort_values("total_exposure", ascending=False)
        .reset_index()
    )

    return by_company, by_sector, by_country, by_source_vehicle


def find_unknown_underlyings(exposures: pd.DataFrame) -> pd.DataFrame:
    df = exposures.copy()
    df = df.dropna(subset=["underlying_symbol_norm"])
    unknown = df[df["company_name"].isna()][["underlying_symbol_norm"]].drop_duplicates()
    unknown = unknown.sort_values("underlying_symbol_norm").reset_index(drop=True)
    return unknown
