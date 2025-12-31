# app.py
"""
Portfolio Look-Through (Streamlit) — Dashboard Reimagined

Design goals
- Professional dashboard layout: sidebar for inputs + controls, main area for KPIs + visuals + detail table
- No cramped columns / truncation: format KPI values explicitly
- Cross-filtering: click bars in charts (Sector/Country/Source) to filter the rest
- Visible filters: show active filters and allow reset/clear

Notes on Streamlit interactions
- Altair chart selections can trigger reruns and return selection data back to Python. We use that
  to implement crossfilter-like behavior. :contentReference[oaicite:2]{index=2}
- Dataframe row selections are also supported via on_select if you want to extend drilldowns later. :contentReference[oaicite:3]{index=3}
"""

import os
from pathlib import Path
from io import StringIO
import datetime as dt

import pandas as pd
import streamlit as st

# Altair for interactive chart selections
import altair as alt

from core import (
    ensure_dirs,
    load_security_master,
    security_master_meta,
    refresh_security_master_from_csv_bytes,
    normalize_portfolio_inputs,
    apply_etf_classification,
    enrich_with_security_master,
    build_lookthrough_exposures,
    build_slices,
    find_unknown_underlyings,
    load_asset_overrides,
    save_asset_overrides,
)

# -------------------------
# Setup
# -------------------------
ensure_dirs()
st.set_page_config(page_title="Portfolio Look-Through", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# UI polish (lightweight)
# -------------------------
st.markdown(
    """
<style>
/* tighten the top padding a bit */
.block-container { padding-top: 1.25rem; padding-bottom: 2.25rem; }

/* make sidebar feel less cramped */
section[data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }

/* nicer KPI cards spacing */
.kpi-row { margin-top: 0.25rem; margin-bottom: 0.75rem; }

/* “chip” style for active filters */
.chip {
  display:inline-block; padding: 0.20rem 0.55rem; border-radius: 999px;
  border: 1px solid rgba(49,51,63,0.20);
  margin-right: 0.35rem; margin-bottom: 0.35rem;
  font-size: 0.85rem;
}
.chip strong { font-weight: 600; }

/* subtle section headings */
.section-title { font-size: 1.05rem; font-weight: 650; margin: 0.25rem 0 0.5rem 0; }

/* keep charts from looking “boxed weirdly” */
[data-testid="stVegaLiteChart"] > div { border-radius: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
def is_local_run() -> bool:
    headless = os.environ.get("STREAMLIT_SERVER_HEADLESS", "")
    return headless.lower() != "true"


def get_av_api_key() -> str | None:
    secrets_key = None
    try:
        secrets_key = st.secrets.get("ALPHAVANTAGE_API_KEY", None)
    except Exception:
        secrets_key = None
    return (
        st.session_state.get("ALPHAVANTAGE_API_KEY")
        or secrets_key
        or os.environ.get("ALPHAVANTAGE_API_KEY")
    )


def persist_av_api_key_locally(key: str) -> None:
    Path(".streamlit").mkdir(exist_ok=True)
    secrets_path = Path(".streamlit/secrets.toml")

    existing = secrets_path.read_text() if secrets_path.exists() else ""
    lines = existing.splitlines()

    out_lines, found = [], False
    for line in lines:
        if line.strip().startswith("ALPHAVANTAGE_API_KEY"):
            out_lines.append(f'ALPHAVANTAGE_API_KEY = "{key}"')
            found = True
        else:
            out_lines.append(line)

    if not found:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        out_lines.append(f'ALPHAVANTAGE_API_KEY = "{key}"')

    secrets_path.write_text("\n".join(out_lines) + "\n")


def human_currency(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e12:
        return f"{sign}${x/1e12:,.2f}T"
    if x >= 1e9:
        return f"{sign}${x/1e9:,.2f}B"
    if x >= 1e6:
        return f"{sign}${x/1e6:,.2f}M"
    if x >= 1e3:
        return f"{sign}${x/1e3:,.1f}K"
    return f"{sign}${x:,.0f}"


def human_number(x: float) -> str:
    try:
        x = float(x)
    except Exception:
        return "—"
    if abs(x) >= 1e9:
        return f"{x/1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"{x/1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:,.1f}K"
    return f"{x:,.0f}"


def safe_str(s) -> str:
    if s is None:
        return ""
    out = str(s).strip()
    return "" if out.lower() in ("nan", "none") else out


def init_state():
    defaults = {
        # inputs / intermediate
        "portfolio_raw": None,
        "portfolio_norm": None,
        "portfolio_classified": None,
        "exposures": None,
        # dashboard filters (global)
        "flt_sector": [],
        "flt_country": [],
        "flt_source": [],
        "flt_ticker": [],
        "flt_min_exposure": 0.0,
        # chart selection memory (so clicks persist)
        "sel_sector": [],
        "sel_country": [],
        "sel_source": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_filters():
    st.session_state["flt_sector"] = []
    st.session_state["flt_country"] = []
    st.session_state["flt_source"] = []
    st.session_state["flt_ticker"] = []
    st.session_state["flt_min_exposure"] = 0.0
    st.session_state["sel_sector"] = []
    st.session_state["sel_country"] = []
    st.session_state["sel_source"] = []


init_state()

# -------------------------
# Sidebar: Inputs + Controls
# -------------------------
st.sidebar.markdown("## Controls")

asof = st.sidebar.date_input("As-of date", value=dt.date.today())
refresh_missing = st.sidebar.checkbox("Fetch missing ETF holdings (if not cached)", value=True)
per_etf_sleep = st.sidebar.slider("Sleep seconds per ETF (avoid AV throttling)", 0.0, 3.0, 0.0, 0.25)
total_value = st.sidebar.number_input(
    "Total portfolio value (only needed if using percent)",
    min_value=0.0,
    value=0.0,
    step=1000.0,
)

st.sidebar.divider()

# API key section (keep it where you need it: near “Run”)
st.sidebar.markdown("### Alpha Vantage Key")
current_key = st.session_state.get("ALPHAVANTAGE_API_KEY") or ""
if not current_key:
    try:
        current_key = st.secrets.get("ALPHAVANTAGE_API_KEY", "")
    except Exception:
        current_key = ""

key_in = st.sidebar.text_input("API Key", value=current_key, type="password")

kcol1, kcol2 = st.sidebar.columns(2)
with kcol1:
    if st.button("Use this session", key="btn_key_session"):
        if key_in.strip():
            st.session_state["ALPHAVANTAGE_API_KEY"] = key_in.strip()
            st.sidebar.success("Key set for session.")
        else:
            st.sidebar.error("Paste a key first.")
with kcol2:
    if is_local_run():
        if st.button("Remember", key="btn_key_remember"):
            if key_in.strip():
                persist_av_api_key_locally(key_in.strip())
                st.session_state["ALPHAVANTAGE_API_KEY"] = key_in.strip()
                st.sidebar.success("Saved locally.")
            else:
                st.sidebar.error("Paste a key first.")
    else:
        st.sidebar.caption("Local save disabled.")

active_key = get_av_api_key()
st.sidebar.caption(f"Key status: {'set' if active_key else 'not set'}")

st.sidebar.divider()

# Data sources
st.sidebar.markdown("### Data Sources")

# Security master
master = load_security_master()
meta = security_master_meta()

sm_file = st.sidebar.file_uploader("Upload Nasdaq Screener CSV", type=["csv"], key="sm_upload")
if st.sidebar.button("Refresh security master", type="primary"):
    if sm_file is None:
        st.sidebar.error("Upload a screener CSV first.")
    else:
        with st.spinner("Refreshing security master..."):
            master = refresh_security_master_from_csv_bytes(sm_file.getvalue(), sm_file.name)
        st.sidebar.success(f"Refreshed ({len(master):,} rows).")
        meta = security_master_meta()

if meta:
    st.sidebar.caption(f"Master rows: {len(master):,}")
    st.sidebar.caption(f"Last refreshed: {meta.get('uploaded_at', '—')}")
else:
    st.sidebar.caption(f"Master rows: {len(master):,}")

# Portfolio input
st.sidebar.markdown("### Portfolio Input")
pf_file = st.sidebar.file_uploader("Upload portfolio CSV", type=["csv"], key="pf_upload")
default_pf = "ticker,dollars,is_etf\nAAPL,5000,False\nQQQ,8000,True\n"
pf_text = st.sidebar.text_area("...or paste CSV", value=default_pf, height=140)

if st.sidebar.button("Load portfolio", type="secondary"):
    try:
        if pf_file is not None:
            df = pd.read_csv(pf_file)
        else:
            raw = pf_text.strip()
            if not raw:
                raise ValueError("Paste area is empty and no CSV was uploaded.")
            df = pd.read_csv(StringIO(raw))

        st.session_state["portfolio_raw"] = df

        tmp = df.copy()
        tmp.columns = [c.strip().lower() for c in tmp.columns]
        if "percent" in tmp.columns and tmp["percent"].notna().any():
            if total_value <= 0:
                raise ValueError("Your portfolio uses 'percent'. Set Total portfolio value in the sidebar (> 0).")

        pf_norm = normalize_portfolio_inputs(df, total_portfolio_value=(total_value if total_value > 0 else None))
        st.session_state["portfolio_norm"] = pf_norm

        pf_class = apply_etf_classification(pf_norm)
        pf_class = enrich_with_security_master(pf_class, master, join_col="ticker_norm")
        st.session_state["portfolio_classified"] = pf_class

        st.sidebar.success(f"Loaded ({len(df):,} rows).")
    except Exception as e:
        st.sidebar.error(str(e))

# Overrides (compact, in sidebar)
with st.sidebar.expander("ETF Overrides (optional)", expanded=False):
    pf_norm = st.session_state.get("portfolio_norm")
    overrides = load_asset_overrides()
    override_map = dict(zip(overrides["ticker_norm"], overrides["is_etf"])) if len(overrides) else {}

    if pf_norm is None or len(pf_norm) == 0:
        st.info("Load a portfolio to edit overrides.")
    else:
        tickers = sorted(pf_norm["ticker_norm"].unique().tolist())
        edit_rows = [{"ticker_norm": t, "is_etf": bool(override_map.get(t, False))} for t in tickers]
        overrides_df = pd.DataFrame(edit_rows)

        edited = st.data_editor(
            overrides_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ticker_norm": st.column_config.TextColumn(disabled=True),
                "is_etf": st.column_config.CheckboxColumn(),
            },
            key="override_editor",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save overrides", key="btn_save_overrides"):
                try:
                    save_asset_overrides(edited)
                    st.success("Saved.")
                except Exception as e:
                    st.error(str(e))
        with c2:
            st.caption("Used only when portfolio `is_etf` is missing.")

st.sidebar.divider()

# Run
run = st.sidebar.button("Run look-through", type="primary", use_container_width=True)

if run:
    try:
        if len(master) == 0:
            raise ValueError("Security master is empty. Upload and refresh it first.")
        pf_norm = st.session_state.get("portfolio_norm")
        if pf_norm is None or len(pf_norm) == 0:
            raise ValueError("No portfolio loaded. Load a portfolio first.")

        api_key = get_av_api_key()
        if not api_key:
            st.sidebar.warning("No API key set. Stock-only works; ETFs may fail to fetch holdings.")

        pf_class = apply_etf_classification(pf_norm)
        pf_class = enrich_with_security_master(pf_class, master, join_col="ticker_norm")
        st.session_state["portfolio_classified"] = pf_class

        progress_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)

        def progress_cb(msg: str, p: float):
            progress_text.write(msg)
            pct = min(max(int(p * 100), 0), 100)
            progress_bar.progress(pct)

        with st.spinner("Computing exposures..."):
            exp = build_lookthrough_exposures(
                api_key=api_key,
                portfolio_df=pf_class,
                master_df=master,
                refresh_missing_etf_holdings=bool(refresh_missing),
                asof_date=str(asof),
                per_etf_sleep_seconds=float(per_etf_sleep),
                progress_cb=progress_cb,
            )

        progress_text.empty()
        progress_bar.empty()

        st.session_state["exposures"] = exp
        reset_filters()
        st.sidebar.success(f"Done ({len(exp):,} exposure rows).")

    except Exception as e:
        st.sidebar.error(str(e))

# -------------------------
# Main: Dashboard
# -------------------------
st.markdown("## Portfolio Look-Through")

exp = st.session_state.get("exposures")
pf_classified = st.session_state.get("portfolio_classified")

if exp is None or len(exp) == 0:
    st.info("Load data and run look-through from the sidebar to see the dashboard.")
    if pf_classified is not None and len(pf_classified) > 0:
        with st.expander("Preview: portfolio (normalized + classified)", expanded=False):
            st.dataframe(pf_classified, use_container_width=True, hide_index=True)
    if len(master) > 0:
        with st.expander("Preview: security master (top 30)", expanded=False):
            st.dataframe(master.head(30), use_container_width=True, hide_index=True)
    st.stop()

# Clean some columns expected downstream
for col in ["sector", "country", "industry", "source_vehicle", "underlying_ticker"]:
    if col not in exp.columns:
        # Keep app resilient if core changes column names later
        pass

# Build slices once (unfiltered) for KPI context
by_company_all, by_sector_all, by_country_all, by_source_all = build_slices(exp)
unknown_all = find_unknown_underlyings(exp)

# -------------------------
# Global filters (sidebar-driven + chart-driven)
# -------------------------
# Options
sectors = sorted({safe_str(x) for x in exp.get("sector", pd.Series(dtype=object)).dropna().tolist()} - {""})
countries = sorted({safe_str(x) for x in exp.get("country", pd.Series(dtype=object)).dropna().tolist()} - {""})
sources = sorted({safe_str(x) for x in exp.get("source_vehicle", pd.Series(dtype=object)).dropna().tolist()} - {""})
tickers = sorted({safe_str(x) for x in exp.get("underlying_ticker", pd.Series(dtype=object)).dropna().tolist()} - {""})

# Filter strip UI
top = st.container()
with top:
    tcol1, tcol2, tcol3, tcol4, tcol5 = st.columns([1.2, 1.2, 1.2, 1.2, 0.8])
    with tcol1:
        st.session_state["flt_sector"] = st.multiselect(
            "Sector",
            options=sectors,
            default=st.session_state.get("flt_sector", []),
        )
    with tcol2:
        st.session_state["flt_country"] = st.multiselect(
            "Country",
            options=countries,
            default=st.session_state.get("flt_country", []),
        )
    with tcol3:
        st.session_state["flt_source"] = st.multiselect(
            "Source",
            options=sources,
            default=st.session_state.get("flt_source", []),
        )
    with tcol4:
        st.session_state["flt_ticker"] = st.multiselect(
            "Underlying",
            options=tickers,
            default=st.session_state.get("flt_ticker", []),
        )
    with tcol5:
        st.session_state["flt_min_exposure"] = st.number_input(
            "Min $",
            min_value=0.0,
            value=float(st.session_state.get("flt_min_exposure", 0.0) or 0.0),
            step=1000.0,
        )

    b1, b2 = st.columns([0.75, 0.25])
    with b1:
        # Active filters display
        chips = []
        if st.session_state["flt_sector"]:
            chips.append(("Sector", ", ".join(st.session_state["flt_sector"][:4]) + ("…" if len(st.session_state["flt_sector"]) > 4 else "")))
        if st.session_state["flt_country"]:
            chips.append(("Country", ", ".join(st.session_state["flt_country"][:4]) + ("…" if len(st.session_state["flt_country"]) > 4 else "")))
        if st.session_state["flt_source"]:
            chips.append(("Source", ", ".join(st.session_state["flt_source"][:4]) + ("…" if len(st.session_state["flt_source"]) > 4 else "")))
        if st.session_state["flt_ticker"]:
            chips.append(("Underlying", ", ".join(st.session_state["flt_ticker"][:4]) + ("…" if len(st.session_state["flt_ticker"]) > 4 else "")))
        if float(st.session_state["flt_min_exposure"] or 0.0) > 0:
            chips.append(("Min $", human_currency(st.session_state["flt_min_exposure"])))

        if chips:
            st.markdown(
                "<div style='margin-top:0.25rem; margin-bottom:0.25rem;'>"
                + "".join([f"<span class='chip'><strong>{k}:</strong> {v}</span>" for k, v in chips])
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No filters applied.")
    with b2:
        if st.button("Reset", use_container_width=True):
            reset_filters()
            st.rerun()

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if st.session_state["flt_sector"]:
        out = out[out["sector"].isin(st.session_state["flt_sector"])]
    if st.session_state["flt_country"]:
        out = out[out["country"].isin(st.session_state["flt_country"])]
    if st.session_state["flt_source"]:
        out = out[out["source_vehicle"].isin(st.session_state["flt_source"])]
    if st.session_state["flt_ticker"]:
        out = out[out["underlying_ticker"].isin(st.session_state["flt_ticker"])]
    minv = float(st.session_state["flt_min_exposure"] or 0.0)
    if minv > 0:
        out = out[out["exposure_value"] >= minv]
    return out

filtered = apply_filters(exp)

# -------------------------
# KPI row (no truncation)
# -------------------------
total_exp = float(filtered["exposure_value"].sum()) if "exposure_value" in filtered.columns else 0.0
n_underlyings = filtered["underlying_ticker"].nunique() if "underlying_ticker" in filtered.columns else 0
n_sources = filtered["source_vehicle"].nunique() if "source_vehicle" in filtered.columns else 0

unknown_filtered = find_unknown_underlyings(filtered)
unknown_cnt = len(unknown_filtered) if unknown_filtered is not None else 0

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total exposure ($)", human_currency(total_exp))
with k2:
    st.metric("Unique underlyings", f"{n_underlyings:,}")
with k3:
    st.metric("Source vehicles", f"{n_sources:,}")
with k4:
    st.metric("Unknown tickers", f"{unknown_cnt:,}")

st.divider()

# -------------------------
# Crossfilter charts (Altair selections)
# -------------------------
# We’ll let chart clicks ADD/SET a filter list for that dimension.
# If you want multi-select-with-shift behavior later, you can extend this logic.

def bar_with_selection(df_slice: pd.DataFrame, dim_col: str, value_col: str, title: str, key_prefix: str):
    # Ensure consistent types
    d = df_slice.copy()
    d[dim_col] = d[dim_col].fillna("(Unknown)").astype(str)

    sel = alt.selection_point(fields=[dim_col], empty=True, toggle=False, name=f"sel_{key_prefix}")

    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            y=alt.Y(f"{dim_col}:N", sort="-x", title=None),
            x=alt.X(f"{value_col}:Q", title=None),
            tooltip=[alt.Tooltip(f"{dim_col}:N", title=dim_col), alt.Tooltip(f"{value_col}:Q", format=",.0f", title="Exposure $")],
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)),
        )
        .add_params(sel)
        .properties(height=min(360, max(180, 22 * min(len(d), 18))), title=title)
    )

    # Streamlit returns selection data when on_select="rerun". :contentReference[oaicite:4]{index=4}
    event = st.altair_chart(chart, use_container_width=True, on_select="rerun")
    # event is either None or a dict of selections depending on version / selection state
    return event

# Build chart data from filtered (so charts reflect current filters)
_, by_sector, by_country, by_source = build_slices(filtered)

# Normalize slice column names for safety
# Expect build_slices returns columns: sector/country/source_vehicle + total_exposure
def ensure_cols(df, dim, total_col="total_exposure"):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[dim, total_col])
    if total_col not in df.columns:
        # try common alt
        for c in ["total_exposure", "exposure", "exposure_value", "total"]:
            if c in df.columns:
                df = df.rename(columns={c: total_col})
                break
    return df

by_sector = ensure_cols(by_sector, "sector")
by_country = ensure_cols(by_country, "country")
by_source = ensure_cols(by_source, "source_vehicle")

c1, c2, c3 = st.columns([1.0, 1.0, 1.0])

with c1:
    st.markdown("<div class='section-title'>Exposure by Sector</div>", unsafe_allow_html=True)
    evt = bar_with_selection(by_sector.head(25), "sector", "total_exposure", "", "sector")
    # Try to extract selected sector(s) robustly
    try:
        # event structure can vary; handle common case
        sel_data = evt.get("selection", {}) if isinstance(evt, dict) else {}
        pts = sel_data.get("sel_sector", {}).get("points", []) or sel_data.get("points", [])
        chosen = [p.get("sector") for p in pts if isinstance(p, dict) and "sector" in p]
        chosen = [x for x in chosen if safe_str(x)]
        if chosen:
            st.session_state["flt_sector"] = chosen
    except Exception:
        pass

with c2:
    st.markdown("<div class='section-title'>Exposure by Country</div>", unsafe_allow_html=True)
    evt = bar_with_selection(by_country.head(25), "country", "total_exposure", "", "country")
    try:
        sel_data = evt.get("selection", {}) if isinstance(evt, dict) else {}
        pts = sel_data.get("sel_country", {}).get("points", []) or sel_data.get("points", [])
        chosen = [p.get("country") for p in pts if isinstance(p, dict) and "country" in p]
        chosen = [x for x in chosen if safe_str(x)]
        if chosen:
            st.session_state["flt_country"] = chosen
    except Exception:
        pass

with c3:
    st.markdown("<div class='section-title'>Exposure by Source</div>", unsafe_allow_html=True)
    evt = bar_with_selection(by_source, "source_vehicle", "total_exposure", "", "source")
    try:
        sel_data = evt.get("selection", {}) if isinstance(evt, dict) else {}
        pts = sel_data.get("sel_source", {}).get("points", []) or sel_data.get("points", [])
        chosen = [p.get("source_vehicle") for p in pts if isinstance(p, dict) and "source_vehicle" in p]
        chosen = [x for x in chosen if safe_str(x)]
        if chosen:
            st.session_state["flt_source"] = chosen
    except Exception:
        pass

st.divider()

# -------------------------
# Details: Top companies + exposure table + downloads
# -------------------------
by_company, _, _, _ = build_slices(filtered)
unknown = find_unknown_underlyings(filtered)

left, right = st.columns([1.0, 1.0])

with left:
    st.markdown("<div class='section-title'>Top Underlyings</div>", unsafe_allow_html=True)
    if by_company is not None and len(by_company) > 0:
        st.dataframe(by_company.head(30), use_container_width=True, hide_index=True, height=380)
    else:
        st.caption("No data after filters.")

with right:
    st.markdown("<div class='section-title'>Exposure Rows (drill)</div>", unsafe_allow_html=True)
    show_cols = [c for c in ["underlying_ticker", "name", "sector", "country", "industry", "source_vehicle", "exposure_value"] if c in filtered.columns]
    df_show = filtered[show_cols].copy() if show_cols else filtered.copy()
    if "exposure_value" in df_show.columns:
        df_show = df_show.sort_values("exposure_value", ascending=False)

    st.dataframe(df_show.head(200), use_container_width=True, hide_index=True, height=380)

# Unknowns callout
if unknown is not None and len(unknown) > 0:
    st.warning(
        "Some tickers were not found in your security master (or mapped to unknown attributes). "
        "Refreshing the Nasdaq screener CSV often fixes this (symbol formatting can matter, e.g., BRK.B → BRK-B)."
    )
    st.dataframe(unknown, use_container_width=True, hide_index=True)

# Download filtered exposures
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered exposures CSV",
    data=csv_bytes,
    file_name=f"exposures_filtered_{asof}.csv",
    mime="text/csv",
)