# dash_app.py
"""
Portfolio Look-Through — Dash UI (professional dashboard)

Assumptions about your core outputs:
- build_lookthrough_exposures(...) returns a DataFrame with at least:
  underlying_ticker, exposure_value, sector, country, industry, source_vehicle
- build_slices(df) returns: by_company, by_sector, by_country, by_source with total_exposure-like columns
  (we normalize column names defensively)

Run:
  python dash_app.py
Then open the local URL it prints.

This file intentionally keeps ALL business logic in core.py.
"""

import os
import datetime as dt
from pathlib import Path
from io import BytesIO, StringIO

import pandas as pd

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
    load_cached_etf_holdings,
    symbol_norm,
    ETF_HOLDINGS_DIR,
    load_asset_overrides,
    save_asset_overrides,
)

# Dash
from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

ensure_dirs()

SOURCE_TYPE_COLORS = {"Stock": "#1f77b4", "ETF": "#ff7f0e"}
ETF_CACHE_GLOB = "holdings_*.parquet"

# -------------------------
# Helpers
# -------------------------
def is_local_run() -> bool:
    headless = os.environ.get("STREAMLIT_SERVER_HEADLESS", "")
    return headless.lower() != "true"


def get_av_api_key_from_env_or_localfile() -> str | None:
    """
    Dash doesn’t have st.secrets. Simplest approach:
    1) env var ALPHAVANTAGE_API_KEY
    2) optional local file .streamlit/secrets.toml from your prior Streamlit setup
    """
    k = os.environ.get("ALPHAVANTAGE_API_KEY")
    if k:
        return k.strip()

    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        txt = secrets_path.read_text(errors="ignore")
        for line in txt.splitlines():
            if line.strip().startswith("ALPHAVANTAGE_API_KEY"):
                # naive parse: ALPHAVANTAGE_API_KEY = "..."
                parts = line.split("=", 1)
                if len(parts) == 2:
                    v = parts[1].strip().strip('"').strip("'")
                    return v or None
    return None


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


def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in ("nan", "none") else s


def normalize_slice(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    """
    Ensure slice tables have columns: dim, total_exposure
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[dim, "total_exposure"])

    out = df.copy()
    if "total_exposure" not in out.columns:
        for c in ["total_exposure", "exposure", "exposure_value", "total"]:
            if c in out.columns:
                out = out.rename(columns={c: "total_exposure"})
                break

    if dim not in out.columns:
        # attempt fallback: first column is the dim
        out = out.rename(columns={out.columns[0]: dim})

    out[dim] = out[dim].fillna("(Unknown)").astype(str)
    out["total_exposure"] = pd.to_numeric(out["total_exposure"], errors="coerce").fillna(0.0)
    out = out.sort_values("total_exposure", ascending=False)
    return out


def load_portfolio_df(contents, filename, text_value):
    import base64

    if contents and filename:
        try:
            b64 = contents.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64)
            return pd.read_csv(BytesIO(raw_bytes))
        except Exception:
            return None
    raw = (text_value or "").strip()
    if not raw:
        return None
    try:
        return pd.read_csv(StringIO(raw))
    except Exception:
        return None


def parse_cache_date(s: str):
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        return None


def latest_cache_date_for_etf(etf_symbol: str):
    etf = symbol_norm(etf_symbol)
    folder = ETF_HOLDINGS_DIR / etf
    if not folder.exists():
        return None
    dates = []
    for p in folder.glob(ETF_CACHE_GLOB):
        name = p.stem.replace("holdings_", "")
        d = parse_cache_date(name)
        if d:
            dates.append(d)
    if not dates:
        return None
    return max(dates).isoformat()


def has_cache_for_date(etf_symbol: str, asof_date: str):
    if not asof_date:
        return False
    etf = symbol_norm(etf_symbol)
    path = ETF_HOLDINGS_DIR / etf / f"holdings_{asof_date}.parquet"
    return path.exists()


def apply_filters(exp: pd.DataFrame, flt: dict) -> pd.DataFrame:
    out = exp
    if flt.get("sector") and "sector" in out.columns:
        out = out[out["sector"].isin(flt["sector"])]
    if flt.get("country") and "country" in out.columns:
        out = out[out["country"].isin(flt["country"])]
    if flt.get("source") and "source_vehicle" in out.columns:
        out = out[out["source_vehicle"].isin(flt["source"])]
    if flt.get("underlying") and "underlying_ticker" in out.columns:
        out = out[out["underlying_ticker"].isin(flt["underlying"])]
    minv = float(flt.get("min_exposure") or 0.0)
    if minv > 0:
        out = out[out["exposure_value"] >= minv]
    return out


def make_bar(df_slice: pd.DataFrame, dim: str, title: str):
    fig = px.bar(
        df_slice.head(25),
        x="total_exposure",
        y=dim,
        orientation="h",
        title=title,
        labels={"total_exposure": "Exposure ($)", dim: ""},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=45, b=10),
        height=380,
        title=dict(font=dict(size=16)),
    )
    fig.update_traces(hovertemplate=f"{dim}: %{{y}}<br>Exposure: %{{x:,.0f}}<extra></extra>")
    return fig


def make_kpi_cards(exp_filtered: pd.DataFrame):
    total = float(exp_filtered["exposure_value"].sum()) if len(exp_filtered) else 0.0
    if len(exp_filtered) and "underlying_ticker" in exp_filtered.columns:
        n_under = exp_filtered["underlying_ticker"].nunique()
    elif len(exp_filtered) and "ticker" in exp_filtered.columns:
        n_under = exp_filtered["ticker"].nunique()
    else:
        n_under = 0
    n_src = exp_filtered["source_vehicle"].nunique() if len(exp_filtered) and "source_vehicle" in exp_filtered.columns else 0
    unknown = find_unknown_underlyings(exp_filtered) if "underlying_ticker" in exp_filtered.columns else None
    unk = len(unknown) if unknown is not None else 0

    def card(label, value):
        return dbc.Card(
            dbc.CardBody(
                [
                    html.Div(label, className="text-muted", style={"fontSize": "0.9rem"}),
                    html.Div(value, style={"fontSize": "1.6rem", "fontWeight": "700"}),
                ]
            ),
            className="shadow-sm",
        )

    return dbc.Row(
        [
            dbc.Col(card("Total exposure", human_currency(total)), md=3),
            dbc.Col(card("Unique underlyings", f"{n_under:,}"), md=3),
            dbc.Col(card("Source vehicles", f"{n_src:,}"), md=3),
            dbc.Col(card("Unknown tickers", f"{unk:,}"), md=3),
        ],
        className="g-3",
    )


def source_type_legend():
    def swatch(label, color_hex):
        return html.Span(
            label,
            className="me-2",
            style={
                "backgroundColor": color_hex,
                "color": "white",
                "padding": "0.15rem 0.45rem",
                "borderRadius": "6px",
                "fontSize": "0.75rem",
                "fontWeight": "600",
            },
        )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Source type", className="text-muted", style={"fontSize": "0.8rem", "marginBottom": "0.25rem"}),
                html.Div([swatch("Stock", SOURCE_TYPE_COLORS["Stock"]), swatch("ETF", SOURCE_TYPE_COLORS["ETF"])]),
            ],
            style={"padding": "0.4rem 0.6rem"},
        ),
        className="shadow-sm",
    )


def chips_from_filters(flt: dict):
    chips = []
    def chip(k, v):
        return dbc.Badge(
            [html.Strong(f"{k}: "), v],
            color="light",
            text_color="dark",
            className="me-2 mb-2",
            style={"border": "1px solid rgba(0,0,0,0.15)", "borderRadius": "999px", "padding": "0.45rem 0.6rem"},
        )

    if flt.get("sector"):
        chips.append(chip("Sector", ", ".join(flt["sector"][:4]) + ("…" if len(flt["sector"]) > 4 else "")))
    if flt.get("country"):
        chips.append(chip("Country", ", ".join(flt["country"][:4]) + ("…" if len(flt["country"]) > 4 else "")))
    if flt.get("source"):
        chips.append(chip("Source", ", ".join(flt["source"][:4]) + ("…" if len(flt["source"]) > 4 else "")))
    if flt.get("underlying"):
        chips.append(chip("Underlying", ", ".join(flt["underlying"][:4]) + ("…" if len(flt["underlying"]) > 4 else "")))
    if float(flt.get("min_exposure") or 0.0) > 0:
        chips.append(chip("Min $", human_currency(flt["min_exposure"])))

    if not chips:
        chips = [html.Span("No filters applied.", className="text-muted")]
    return chips


# -------------------------
# App
# -------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Portfolio Look-Through"

app.layout = dbc.Container(
    fluid=True,
    children=[
        # Stores
        dcc.Store(id="store-exposures"),          # json
        dcc.Store(id="store-filters", data={"sector": [], "country": [], "source": [], "underlying": [], "min_exposure": 0.0}),
        dcc.Store(id="store-master-meta"),
        dcc.Store(id="store-status-refresh", data=None),
        dcc.Store(id="store-status-run", data=None),
        dcc.Store(id="store-status", data={"message": "", "kind": "info"}),

        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Portfolio Look-Through", className="mb-0"),
                        html.Div("Look-through exposures for stocks + ETFs with linked dashboard filtering.", className="text-muted"),
                    ],
                    md=8,
                ),
                dbc.Col(
                    dbc.Button("Clear all filters", id="btn-clear-filters", color="secondary", outline=True, className="mt-2", n_clicks=0),
                    md=4,
                    className="text-end",
                ),
            ],
            className="mt-3 mb-2",
        ),

        dbc.Row(
            [
                # Left panel (controls)
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Get started", className="h5 mb-1"),
                                html.Div("Follow the steps below. Your inputs can be saved locally for next time.", className="text-muted mb-3"),

                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            [
                                                html.Div("First describe your portfolio. Upload a CSV or paste it in.", className="text-muted mb-2"),
                                                html.Div("Required: ticker plus dollars or percent. Optional: is_etf or asset_type to identify ETFs.", className="text-muted mb-2", style={"fontSize": "0.9rem"}),
                                                html.Div("Examples:", className="text-muted", style={"fontSize": "0.85rem"}),
                                                html.Ul(
                                                    [
                                                        html.Li("ticker,dollars,is_etf"),
                                                        html.Li("ticker,percent"),
                                                        html.Li("ticker,percent,asset_type"),
                                                    ],
                                                    className="text-muted",
                                                    style={"fontSize": "0.85rem", "marginBottom": "0.75rem"},
                                                ),
                                                dbc.Label("Portfolio CSV (upload)"),
                                                dcc.Upload(
                                                    id="upload-portfolio",
                                                    children=html.Div(["Drag & drop or ", html.A("select CSV")]),
                                                    style={
                                                        "width": "100%",
                                                        "height": "60px",
                                                        "lineHeight": "60px",
                                                        "borderWidth": "1px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "10px",
                                                        "textAlign": "center",
                                                    },
                                                ),
                                                dbc.Label("…or paste CSV", className="mt-2"),
                                                dbc.Textarea(
                                                    id="portfolio-text",
                                                    value="ticker,dollars,is_etf\nAAPL,5000,False\nQQQ,8000,True\n",
                                                    style={"height": "140px", "fontFamily": "ui-monospace, SFMono-Regular, Menlo, monospace"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                    persisted_props=["value"],
                                                ),
                                                dbc.FormText("Columns: ticker plus dollars or percent. Optional: is_etf.", className="mt-1"),
                                                dbc.Collapse(
                                                    [
                                                        dbc.Label("Total portfolio value", className="mt-3"),
                                                        dbc.Input(
                                                            id="total-value",
                                                            type="number",
                                                            min=0,
                                                            step=1000,
                                                            value=0,
                                                            persistence=True,
                                                            persistence_type="local",
                                                            persisted_props=["value"],
                                                        ),
                                                        html.Div(id="total-value-help", className="text-muted mt-1", style={"fontSize": "0.85rem"}),
                                                    ],
                                                    id="total-value-wrap",
                                                    is_open=False,
                                                ),
                                                html.Div(id="portfolio-check", className="mt-2"),
                                            ],
                                            label="1) Portfolio",
                                            tab_id="tab-portfolio",
                                        ),
                                        dbc.Tab(
                                            [
                                                html.Div("Now determine the constituent holdings for the ETFs in your portfolio.", className="text-muted mb-2"),
                                                html.Div(id="etf-cache-status", className="mb-2"),
                                                dbc.Label("Refresh policy"),
                                                dbc.RadioItems(
                                                    id="refresh-policy",
                                                    options=[
                                                        {"label": "Get ETFs that are not cached (recommended)", "value": "missing"},
                                                        {"label": "Get/Refresh ETFs older than the chosen date", "value": "older_than"},
                                                        {"label": "Get/Refresh all ETFs", "value": "all"},
                                                    ],
                                                    value="missing",
                                                    className="mb-2",
                                                    persistence=True,
                                                    persistence_type="local",
                                                    persisted_props=["value"],
                                                ),
                                                dbc.Label("Refresh as-of date"),
                                                dcc.DatePickerSingle(
                                                    id="asof-date",
                                                    date=dt.date.today(),
                                                    display_format="YYYY-MM-DD",
                                                    style={"width": "100%"},
                                                    persistence=True,
                                                    persistence_type="local",
                                                    persisted_props=["date"],
                                                ),
                                                dbc.FormText("Used to decide which ETFs are stale and should be refreshed.", className="mt-1"),
                                                dbc.Label("Sleep seconds per ETF (throttling)", className="mt-2"),
                                                dcc.Slider(
                                                    id="per-etf-sleep",
                                                    min=0.0,
                                                    max=3.0,
                                                    step=0.25,
                                                    value=0.0,
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                    persistence=True,
                                                    persistence_type="local",
                                                    persisted_props=["value"],
                                                ),
                                                html.Hr(),
                                                dbc.Label("Alpha Vantage API Key"),
                                                dbc.Input(
                                                    id="api-key",
                                                    type="password",
                                                    value=get_av_api_key_from_env_or_localfile() or "",
                                                    placeholder="Paste key",
                                                    persistence=True,
                                                    persistence_type="local",
                                                    persisted_props=["value"],
                                                ),
                                                dbc.FormText(
                                                    [
                                                        "Needed to fetch ETF holdings if they are not already cached. A free API key can be generated ",
                                                        html.A("here", href="https://www.alphavantage.co/support/#api-key", target="_blank", rel="noreferrer"),
                                                        ".",
                                                    ],
                                                    className="mt-1",
                                                ),
                                                html.Div(id="api-key-indicator", className="text-muted mt-1", style={"fontSize": "0.85rem"}),
                                            ],
                                            label="2) ETF holdings",
                                            tab_id="tab-etf",
                                        ),
                                        dbc.Tab(
                                            [
                                                html.Div(
                                                    [
                                                        "Now specify equity characteristics. We use the Nasdaq Screener security master for sector/country/industry. A CSV can be downloaded ",
                                                        html.A("here", href="https://www.nasdaq.com/market-activity/stocks/screener", target="_blank", rel="noreferrer"),
                                                        ".",
                                                    ],
                                                    className="text-muted mb-2",
                                                ),
                                                dbc.Label("Security Master (Nasdaq screener CSV)"),
                                                dcc.Upload(
                                                    id="upload-security-master",
                                                    children=html.Div(["Drag & drop or ", html.A("select CSV")]),
                                                    style={
                                                        "width": "100%",
                                                        "height": "60px",
                                                        "lineHeight": "60px",
                                                        "borderWidth": "1px",
                                                        "borderStyle": "dashed",
                                                        "borderRadius": "10px",
                                                        "textAlign": "center",
                                                    },
                                                ),
                                                dbc.Button("Refresh security master", id="btn-refresh-master", color="primary", className="mt-2", n_clicks=0),
                                                html.Div(id="master-meta", className="text-muted mt-2", style={"fontSize": "0.9rem"}),
                                            ],
                                            label="3) Security master",
                                            tab_id="tab-master",
                                        ),
                                        dbc.Tab(
                                            [
                                                html.Div("Run the look-through. Your inputs are saved in this browser.", className="text-muted mb-2"),
                                                dbc.Button("Run look-through", id="btn-run", color="success", className="mt-2", n_clicks=0, style={"width": "100%"}),
                                                html.Div(id="status-banner", className="mt-3"),
                                            ],
                                            label="4) Run",
                                            tab_id="tab-run",
                                        ),
                                    ],
                                    id="tabs-steps",
                                    active_tab="tab-portfolio",
                                ),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    md=3,
                ),

                # Main dashboard
                dbc.Col(
                    [
                        dbc.Card(dbc.CardBody([html.Div(id="active-filters", className="mb-2")]), className="shadow-sm mb-3"),

                        html.Div(id="kpi-row", className="mb-2"),
                        html.Div(id="source-legend-row", className="mb-3"),
                        html.Div(id="etf-status-panel", className="mb-3"),

                        dbc.Row(
                            [
                                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="chart-sector", config={"displayModeBar": False})), className="shadow-sm"), md=4),
                                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="chart-country", config={"displayModeBar": False})), className="shadow-sm"), md=4),
                                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="chart-source", config={"displayModeBar": False})), className="shadow-sm"), md=4),
                            ],
                            className="g-3 mb-3",
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.Div("Top underlyings", className="h6"),
                                                dcc.Graph(id="chart-top-underlyings", config={"displayModeBar": False}),
                                            ]
                                        ),
                                        className="shadow-sm",
                                    ),
                                    md=12,
                                ),
                            ],
                            className="g-3 mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.Div("Exposure rows (drill)", className="h6"),
                                                dash_table.DataTable(
                                                    id="table-exposures",
                                                    page_size=25,
                                                    sort_action="native",
                                                    filter_action="none",
                                                    style_table={"height": "520px", "overflowY": "auto"},
                                                    style_cell={"fontFamily": "system-ui", "fontSize": "0.9rem", "padding": "8px"},
                                                    style_header={"fontWeight": "700"},
                                                ),
                                            ]
                                        ),
                                        className="shadow-sm",
                                    ),
                                    md=12,
                                ),
                            ],
                            className="g-3",
                        ),

                        html.Div(id="unknown-panel", className="mt-3"),
                    ],
                    md=9,
                ),
            ],
            className="g-3",
        ),

        html.Div(style={"height": "24px"}),
    ],
)

# -------------------------
# Callbacks: status banner
# -------------------------
@app.callback(
    Output("status-banner", "children"),
    Input("store-status", "data"),
)
def render_status(st_data):
    if not st_data:
        return ""
    msg = st_data.get("message", "")
    kind = st_data.get("kind", "info")
    if not msg:
        return ""
    color = {"info": "info", "success": "success", "warning": "warning", "error": "danger"}.get(kind, "info")
    return dbc.Alert(msg, color=color, className="py-2 mb-0")


@app.callback(
    Output("store-status", "data"),
    Input("store-status-refresh", "data"),
    Input("store-status-run", "data"),
    prevent_initial_call=True,
)
def mux_status(refresh_status, run_status):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    if trig == "store-status-refresh.data":
        return refresh_status
    if trig == "store-status-run.data":
        return run_status
    return no_update


@app.callback(
    Output("total-value-wrap", "is_open"),
    Output("total-value-help", "children"),
    Output("portfolio-check", "children"),
    Input("upload-portfolio", "contents"),
    Input("upload-portfolio", "filename"),
    Input("portfolio-text", "value"),
)
def update_portfolio_guidance(pf_contents, pf_filename, pf_text):
    df = load_portfolio_df(pf_contents, pf_filename, pf_text)
    if df is None:
        help_text = "Only required if your CSV includes a percent column."
        msg = "Upload or paste a portfolio to continue."
        return False, help_text, dbc.Alert(msg, color="secondary", className="py-2 mb-0")

    cols = [c.strip().lower() for c in df.columns]
    if "percent" in cols:
        pct_col = df.columns[cols.index("percent")]
        has_percent = df[pct_col].notna().any()
    else:
        has_percent = False
    total_help = "Percent column detected. Enter total portfolio value to compute dollars." if has_percent else "Not required unless using percent."

    n_rows = len(df)
    n_etf = None
    if "is_etf" in cols:
        s = df[df.columns[cols.index("is_etf")]]
        s_norm = s.fillna("").astype(str).str.strip().str.lower()
        n_etf = int(s_norm.isin(["true", "1", "yes", "y", "etf"]).sum())
    elif "asset_type" in cols:
        s = df[df.columns[cols.index("asset_type")]].astype(str).str.upper()
        n_etf = int((s == "ETF").sum())

    if n_etf is None:
        base_msg = f"Loaded {n_rows:,} rows. ETF count unknown (no is_etf/asset_type column)."
    else:
        base_msg = f"Loaded {n_rows:,} rows. ETFs: {n_etf:,}."
    return has_percent, total_help, dbc.Alert(base_msg, color="info", className="py-2 mb-0")


@app.callback(
    Output("api-key-indicator", "children"),
    Input("api-key", "value"),
)
def render_api_key_indicator(api_key):
    if api_key and str(api_key).strip():
        return "Saved locally in this browser."
    return "Not saved yet."


@app.callback(
    Output("etf-cache-status", "children"),
    Output("etf-status-panel", "children"),
    Input("upload-portfolio", "contents"),
    Input("upload-portfolio", "filename"),
    Input("portfolio-text", "value"),
    Input("asof-date", "date"),
    Input("refresh-policy", "value"),
    Input("api-key", "value"),
    Input("tabs-steps", "active_tab"),
)
def render_etf_cache_status(pf_contents, pf_filename, pf_text, asof_date, refresh_policy, api_key, active_tab):
    df = load_portfolio_df(pf_contents, pf_filename, pf_text)
    if df is None:
        summary = dbc.Alert("Add your portfolio in Step 1 to check ETF holdings status.", color="secondary", className="py-2 mb-0")
        panel = "" if active_tab != "tab-etf" else summary
        return summary, panel

    cols = [c.strip().lower() for c in df.columns]
    etf_tickers = []
    if "is_etf" in cols:
        s = df[df.columns[cols.index("is_etf")]]
        s_norm = s.fillna("").astype(str).str.strip().str.lower()
        etf_mask = s_norm.isin(["true", "1", "yes", "y", "etf"])
        tick_col = df.columns[cols.index("ticker")] if "ticker" in cols else None
        if tick_col:
            etf_tickers = df.loc[etf_mask, tick_col].dropna().astype(str).tolist()
    elif "asset_type" in cols:
        s = df[df.columns[cols.index("asset_type")]].astype(str).str.upper()
        tick_col = df.columns[cols.index("ticker")] if "ticker" in cols else None
        if tick_col:
            etf_tickers = df.loc[s == "ETF", tick_col].dropna().astype(str).tolist()

    if not etf_tickers:
        summary = dbc.Alert(
            "ETF list unknown. Add an is_etf or asset_type column to check holdings status.",
            color="secondary",
            className="py-2 mb-0",
        )
        panel = "" if active_tab != "tab-etf" else summary
        return summary, panel

    etf_tickers = sorted({t.strip() for t in etf_tickers if str(t).strip()})
    asof_date = asof_date or dt.date.today().isoformat()
    policy = refresh_policy or "missing"

    rows = []
    api_calls = 0
    for t in etf_tickers:
        latest_date = latest_cache_date_for_etf(t)
        has_asof = has_cache_for_date(t, asof_date)

        if policy == "all":
            will_refresh = True
        elif policy == "older_than":
            if latest_date is None:
                will_refresh = True
            else:
                will_refresh = latest_date < asof_date
        else:
            will_refresh = latest_date is None

        if will_refresh:
            api_calls += 1

        rows.append(
            {
                "ETF": t,
                "Latest cache": latest_date or "Missing",
                "Cached for as-of": "Yes" if has_asof else "No",
                "Will refresh": "Yes" if will_refresh else "No",
            }
        )

    api_key_set = bool((api_key or "").strip())
    header = f"ETFs detected: {len(etf_tickers)}. Estimated API calls: {api_calls}."
    if api_calls and not api_key_set:
        header += " API key required to refresh."

    summary = dbc.Alert(header, color="info", className="py-2 mb-0")

    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["ETF", "Latest cache", "Cached for as-of", "Will refresh"]],
        style_table={"maxHeight": "360px", "overflowY": "auto", "width": "100%"},
        style_cell={"fontFamily": "system-ui", "fontSize": "0.9rem", "padding": "8px"},
        style_header={"fontWeight": "700"},
        page_size=12,
    )

    panel = "" if active_tab != "tab-etf" else dbc.Card(
        dbc.CardBody(
            [
                html.Div("ETF holdings status", className="h6 mb-2"),
                html.Div(header, className="text-muted", style={"fontSize": "0.85rem"}),
                html.Div(table, className="mt-2"),
            ]
        ),
        className="shadow-sm",
    )
    return summary, panel


# -------------------------
# Callbacks: refresh security master
# -------------------------
@app.callback(
    Output("store-master-meta", "data"),
    Output("store-status-refresh", "data"),
    Input("btn-refresh-master", "n_clicks"),
    State("upload-security-master", "contents"),
    State("upload-security-master", "filename"),
    prevent_initial_call=True,
)
def refresh_master(n, contents, filename):
    if not contents or not filename:
        return no_update, {"message": "Upload a Nasdaq screener CSV first.", "kind": "error"}

    # contents is "data:...;base64,<...>"
    import base64
    b64 = contents.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64)

    try:
        df = refresh_security_master_from_csv_bytes(raw_bytes, filename)
        meta = security_master_meta() or {}
        meta["rows"] = len(df)
        return meta, {"message": f"Security master refreshed ({len(df):,} rows).", "kind": "success"}
    except Exception as e:
        return no_update, {"message": str(e), "kind": "error"}


@app.callback(
    Output("master-meta", "children"),
    Input("store-master-meta", "data"),
)
def render_master_meta(meta):
    if not meta:
        # show current cached meta if exists
        meta = security_master_meta() or {}
    rows = meta.get("rows", None)
    if rows is None:
        # fallback to current cache
        try:
            rows = len(load_security_master())
        except Exception:
            rows = 0
    refreshed = meta.get("uploaded_at", "—")
    raw_copy = meta.get("raw_copy", "—")
    return html.Div([f"Rows: {rows:,} • Last refreshed: {refreshed}", html.Br(), f"Raw copy: {raw_copy}"])


# -------------------------
# Callback: run look-through
# -------------------------
@app.callback(
    Output("store-exposures", "data"),
    Output("store-status-run", "data"),
    Input("btn-run", "n_clicks"),
    State("api-key", "value"),
    State("upload-portfolio", "contents"),
    State("upload-portfolio", "filename"),
    State("portfolio-text", "value"),
    State("total-value", "value"),
    State("refresh-policy", "value"),
    State("asof-date", "date"),
    State("per-etf-sleep", "value"),
    prevent_initial_call=True,
)
def run_lookthrough(n, api_key, pf_contents, pf_filename, pf_text, total_value, refresh_policy, asof_date, per_etf_sleep):
    try:
        master = load_security_master()
        if master is None or len(master) == 0:
            return no_update, {"message": "Security master is empty. Upload + refresh it first.", "kind": "error"}

        # Load portfolio
        df = None
        import base64
        if pf_contents and pf_filename:
            b64 = pf_contents.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64)
            df = pd.read_csv(BytesIO(raw_bytes))
        else:
            raw = (pf_text or "").strip()
            if not raw:
                return no_update, {"message": "No portfolio provided (upload or paste).", "kind": "error"}
            df = pd.read_csv(StringIO(raw))

        # Validate percent total value early
        tmp = df.copy()
        tmp.columns = [c.strip().lower() for c in tmp.columns]
        if "percent" in tmp.columns and tmp["percent"].notna().any():
            if not total_value or float(total_value) <= 0:
                return no_update, {"message": "Portfolio uses 'percent'. Set Total portfolio value (>0).", "kind": "error"}

        pf_norm = normalize_portfolio_inputs(df, total_portfolio_value=(float(total_value) if total_value and float(total_value) > 0 else None))
        pf_class = apply_etf_classification(pf_norm)
        pf_class = enrich_with_security_master(pf_class, master, join_col="ticker_norm")

        # key may be blank if stock-only; we’ll allow run but warn in status
        key = (api_key or "").strip()
        if not key:
            warn = "No API key set. Stock-only works; ETFs may fail to fetch holdings."
        else:
            warn = ""

        exp = build_lookthrough_exposures(
            api_key=key if key else None,
            portfolio_df=pf_class,
            master_df=master,
            refresh_missing_etf_holdings=True,
            asof_date=str(asof_date),
            refresh_policy=refresh_policy or "missing",
            refresh_cutoff_date=str(asof_date),
            per_etf_sleep_seconds=float(per_etf_sleep or 0.0),
            progress_cb=None,  # optional: wire up later with a progress component
        )

        payload = exp.to_json(date_format="iso", orient="split")
        msg = f"Done. Exposures rows: {len(exp):,}."
        kind = "success"
        if warn:
            msg = f"{msg} {warn}"
            kind = "warning"
        return payload, {"message": msg, "kind": kind}

    except Exception as e:
        return no_update, {"message": str(e), "kind": "error"}


# -------------------------
# Callbacks: chart clicks -> update filters (true crossfilter)
# -------------------------

@app.callback(
    Output("store-filters", "data"),
    Input("btn-clear-filters", "n_clicks"),
    Input("chart-sector", "clickData"),
    Input("chart-country", "clickData"),
    Input("chart-source", "clickData"),
    Input("chart-top-underlyings", "clickData"),
    State("store-filters", "data"),
    prevent_initial_call=True,
)
def update_filters(btn_clear, sector_click, country_click, source_click, top_click, flt):
    flt = flt or {"sector": [], "country": [], "source": [], "underlying": [], "min_exposure": 0.0}

    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    # Clear all
    if trig == "btn-clear-filters.n_clicks":
        return {"sector": [], "country": [], "source": [], "underlying": [], "min_exposure": 0.0}

    def get_click_y(cd):
        if not cd:
            return None
        pts = cd.get("points") or []
        if not pts:
            return None
        return pts[0].get("y")

    def get_click_x(cd):
        if not cd:
            return None
        pts = cd.get("points") or []
        if not pts:
            return None
        return pts[0].get("x")

    # Toggle behavior: click same value again clears that filter dimension
    if trig == "chart-sector.clickData":
        v = get_click_y(sector_click)
        if v:
            flt["sector"] = [v] if flt.get("sector") != [v] else []
    elif trig == "chart-country.clickData":
        v = get_click_y(country_click)
        if v:
            flt["country"] = [v] if flt.get("country") != [v] else []
    elif trig == "chart-source.clickData":
        v = get_click_y(source_click)
        if v:
            flt["source"] = [v] if flt.get("source") != [v] else []
    elif trig == "chart-top-underlyings.clickData":
        v = get_click_x(top_click)
        if v:
            flt["underlying"] = [v] if flt.get("underlying") != [v] else []

    return flt

# -------------------------
# Callback: render dashboard from exposures + filters
# -------------------------
@app.callback(
    Output("active-filters", "children"),
    Output("kpi-row", "children"),
    Output("source-legend-row", "children"),
    Output("chart-sector", "figure"),
    Output("chart-country", "figure"),
    Output("chart-source", "figure"),
    Output("chart-top-underlyings", "figure"),
    Output("table-exposures", "data"),
    Output("table-exposures", "columns"),
    Output("unknown-panel", "children"),
    Input("store-exposures", "data"),
    Input("store-filters", "data"),
)
def render_dashboard(exp_json, flt):
    empty_fig = px.bar(pd.DataFrame({"x": [], "y": []}))
    empty_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=300)

    if not exp_json:
        return (
            chips_from_filters(flt or {}),
            dbc.Alert("Run look-through to see the dashboard.", color="info", className="mb-0"),
            "",
            empty_fig, empty_fig, empty_fig, empty_fig,
            [], [],
            "",
        )

    exp = pd.read_json(exp_json, orient="split")
    flt = flt or {"sector": [], "country": [], "source": [], "underlying": [], "min_exposure": 0.0}

    # Normalize core columns for consistent filtering/labels.
    if "underlying_ticker" not in exp.columns:
        if "underlying_symbol_norm" in exp.columns:
            exp["underlying_ticker"] = exp["underlying_symbol_norm"]
        elif "ticker" in exp.columns:
            exp["underlying_ticker"] = exp["ticker"]
    if "name" not in exp.columns:
        if "company_name" in exp.columns:
            exp["name"] = exp["company_name"]
        elif "Name" in exp.columns:
            exp["name"] = exp["Name"]
    if "source_vehicle" not in exp.columns and "source_ticker_norm" in exp.columns:
        exp["source_vehicle"] = exp["source_ticker_norm"]

    # Defensive cleanup
    for c in ["sector", "country", "industry", "source_vehicle", "underlying_ticker"]:
        if c in exp.columns:
            exp[c] = exp[c].fillna("(Unknown)").astype(str)

    if "exposure_value" in exp.columns:
        exp["exposure_value"] = pd.to_numeric(exp["exposure_value"], errors="coerce").fillna(0.0)

    exp_f = apply_filters(exp, flt)

    # KPI cards
    kpis = make_kpi_cards(exp_f)

    # slices
    by_company, by_sector, by_country, _by_source = build_slices(exp_f)
    if "source_type" in exp_f.columns:
        by_sector = (
            exp_f.groupby(["sector", "source_type"], dropna=False)
            .agg(total_exposure=("exposure_value", "sum"))
            .sort_values("total_exposure", ascending=False)
            .reset_index()
        )
        by_country = (
            exp_f.groupby(["country", "source_type"], dropna=False)
            .agg(total_exposure=("exposure_value", "sum"))
            .sort_values("total_exposure", ascending=False)
            .reset_index()
        )
    by_sector = normalize_slice(by_sector, "sector")
    by_country = normalize_slice(by_country, "country")
    if "source_vehicle" in exp_f.columns and "source_type" in exp_f.columns:
        by_source = (
            exp_f.groupby(["source_vehicle", "source_type"], dropna=False)
            .agg(total_exposure=("exposure_value", "sum"))
            .sort_values("total_exposure", ascending=False)
            .reset_index()
        )
    elif "source_vehicle" in exp_f.columns:
        by_source = (
            exp_f.groupby("source_vehicle", dropna=False)
            .agg(total_exposure=("exposure_value", "sum"))
            .sort_values("total_exposure", ascending=False)
            .reset_index()
        )
    else:
        by_source = _by_source
    by_source = normalize_slice(by_source, "source_vehicle")

    if "source_type" in by_sector.columns:
        fig_sector = px.bar(
            by_sector.head(25),
            x="total_exposure",
            y="sector",
            color="source_type",
            orientation="h",
            title="Exposure by Sector",
            labels={"total_exposure": "Exposure ($)", "sector": ""},
            color_discrete_map=SOURCE_TYPE_COLORS,
        )
        fig_sector.update_layout(
            margin=dict(l=10, r=10, t=45, b=10),
            height=380,
            title=dict(font=dict(size=16)),
            showlegend=False,
        )
        fig_sector.update_traces(hovertemplate="Sector: %{y}<br>Exposure: %{x:,.0f}<extra></extra>")
    else:
        fig_sector = make_bar(by_sector, "sector", "Exposure by Sector")

    if "source_type" in by_country.columns:
        fig_country = px.bar(
            by_country.head(25),
            x="total_exposure",
            y="country",
            color="source_type",
            orientation="h",
            title="Exposure by Country",
            labels={"total_exposure": "Exposure ($)", "country": ""},
            color_discrete_map=SOURCE_TYPE_COLORS,
        )
        fig_country.update_layout(
            margin=dict(l=10, r=10, t=45, b=10),
            height=380,
            title=dict(font=dict(size=16)),
            showlegend=False,
        )
        fig_country.update_traces(hovertemplate="Country: %{y}<br>Exposure: %{x:,.0f}<extra></extra>")
    else:
        fig_country = make_bar(by_country, "country", "Exposure by Country")

    if "source_type" in by_source.columns:
        fig_source = px.bar(
            by_source.head(25),
            x="total_exposure",
            y="source_vehicle",
            color="source_type",
            orientation="h",
            title="Exposure by Source",
            labels={"total_exposure": "Exposure ($)", "source_vehicle": ""},
            color_discrete_map=SOURCE_TYPE_COLORS,
        )
        fig_source.update_layout(
            margin=dict(l=10, r=10, t=45, b=10),
            height=380,
            title=dict(font=dict(size=16)),
            showlegend=False,
        )
        fig_source.update_traces(hovertemplate="Source: %{y}<br>Exposure: %{x:,.0f}<extra></extra>")
    else:
        fig_source = make_bar(by_source, "source_vehicle", "Exposure by Source")

    # Top underlyings chart
    if "source_type" in exp_f.columns:
        by_company = (
            exp_f.dropna(subset=["underlying_ticker"])
            .groupby(["underlying_ticker", "source_type"], dropna=False)
            .agg(total_exposure=("exposure_value", "sum"))
            .sort_values("total_exposure", ascending=False)
            .reset_index()
        )
    if by_company is None or len(by_company) == 0:
        top_fig = empty_fig
    else:
        bc = by_company.copy()
        # try normalize total column
        if "total_exposure" not in bc.columns:
            for c in ["total_exposure", "exposure", "exposure_value", "total"]:
                if c in bc.columns:
                    bc = bc.rename(columns={c: "total_exposure"})
                    break
        # try normalize dim column
        if "underlying_ticker" not in bc.columns:
            bc = bc.rename(columns={bc.columns[0]: "underlying_ticker"})
        bc = bc.sort_values("total_exposure", ascending=False).head(15)
        if "source_type" in bc.columns:
            top_fig = px.bar(
                bc,
                x="underlying_ticker",
                y="total_exposure",
                color="source_type",
                title="Top Underlyings",
                labels={"underlying_ticker": "", "total_exposure": "Exposure ($)"},
                color_discrete_map=SOURCE_TYPE_COLORS,
            )
            top_fig.update_layout(
                margin=dict(l=10, r=10, t=45, b=10),
                height=380,
                title=dict(font=dict(size=16)),
                showlegend=False,
            )
            top_fig.update_traces(hovertemplate="Ticker: %{x}<br>Exposure: %{y:,.0f}<extra></extra>")
        else:
            top_fig = px.bar(
                bc,
                x="underlying_ticker",
                y="total_exposure",
                title="Top Underlyings",
                labels={"underlying_ticker": "", "total_exposure": "Exposure ($)"},
            )
            top_fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=380, title=dict(font=dict(size=16)))
            top_fig.update_traces(hovertemplate="Ticker: %{x}<br>Exposure: %{y:,.0f}<extra></extra>")

    # Table
    show_cols = [c for c in ["underlying_ticker", "name", "sector", "country", "industry", "source_vehicle", "exposure_value"] if c in exp_f.columns]
    df_show = exp_f[show_cols].copy() if show_cols else exp_f.copy()
    if "exposure_value" in df_show.columns:
        df_show = df_show.sort_values("exposure_value", ascending=False)

    data = df_show.head(500).to_dict("records")
    cols = [{"name": c, "id": c} for c in df_show.columns]

    # Unknown panel
    unknown = find_unknown_underlyings(exp_f)
    if unknown is not None and len(unknown) > 0:
        unk = unknown.head(200).to_dict("records")
        unk_cols = [{"name": c, "id": c} for c in unknown.columns]
        unknown_panel = dbc.Card(
            dbc.CardBody(
                [
                    html.Div("Unknown tickers (not found in security master)", className="h6"),
                    dash_table.DataTable(
                        data=unk,
                        columns=unk_cols,
                        page_size=10,
                        style_cell={"fontFamily": "system-ui", "fontSize": "0.9rem", "padding": "8px"},
                        style_header={"fontWeight": "700"},
                    ),
                    html.Div("Fix: refresh screener CSV and/or confirm symbol normalization.", className="text-muted mt-2"),
                ]
            ),
            className="shadow-sm",
        )
    else:
        unknown_panel = ""

    # Active filters chips
    chips = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.Div("Active filters", className="text-muted", style={"fontSize": "0.9rem"}),
                            html.Div(chips_from_filters(flt), className="mt-2"),
                        ]
                    ),
                    className="shadow-sm",
                )
            )
        ]
    )

    if "source_type" in exp_f.columns:
        legend_row = dbc.Row([dbc.Col(source_type_legend(), md=12)])
    else:
        legend_row = ""

    return chips, kpis, legend_row, fig_sector, fig_country, fig_source, top_fig, data, cols, unknown_panel


if __name__ == "__main__":
    # Production: use gunicorn, etc. Locally this is fine.
    app.run(debug=True)
