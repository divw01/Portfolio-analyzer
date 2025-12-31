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
import json
import datetime as dt
from pathlib import Path
from io import BytesIO, StringIO

import pandas as pd

from archive.core_updated import (
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
    load_asset_overrides,
    save_asset_overrides,
)

# Dash
from dash import Dash, html, dcc, Input, Output, State, callback_context, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.express as px

ensure_dirs()

SOURCE_TYPE_COLORS = {"Stock": "#1f77b4", "ETF": "#ff7f0e"}
UI_STATE_PATH = Path("data/ui_state.json")

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


def load_ui_state() -> dict:
    try:
        if UI_STATE_PATH.exists():
            return json.loads(UI_STATE_PATH.read_text())
    except Exception:
        return {}
    return {}


def save_ui_state(state: dict) -> None:
    try:
        UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        UI_STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


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
INITIAL_UI_STATE = load_ui_state()

app.layout = dbc.Container(
    fluid=True,
    children=[
        # Stores
        dcc.Store(id="store-exposures"),          # json
        dcc.Store(id="store-filters", data={"sector": [], "country": [], "source": [], "underlying": [], "min_exposure": 0.0}),
        dcc.Store(id="store-master-meta"),
        dcc.Store(id="store-status-refresh", data=None),
        dcc.Store(id="store-status-run", data=None),
        dcc.Store(id="store-status-inputs", data=None),
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
                                                html.Div(
                                                    "Step 1: Describe your portfolio. Upload a CSV or paste a small table.",
                                                    className="text-muted mb-2",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div("Required column:", className="fw-semibold"),
                                                        html.Ul(
                                                            [html.Li("ticker — the security symbol (equities and ETFs)")],
                                                            className="text-muted",
                                                            style={"fontSize": "0.9rem", "marginBottom": "0.5rem"},
                                                        ),
                                                        html.Div("You must provide ONE of these value inputs:", className="fw-semibold"),
                                                        html.Ul(
                                                            [
                                                                html.Li("dollars  (position value in USD)"),
                                                                html.Li("shares AND price_per_share"),
                                                                html.Li("percent  (weights). If any row relies on percent, we will ask for total_portfolio_value."),
                                                            ],
                                                            className="text-muted",
                                                            style={"fontSize": "0.9rem", "marginBottom": "0.5rem"},
                                                        ),
                                                        html.Div("Optional helpers:", className="fw-semibold"),
                                                        html.Ul(
                                                            [
                                                                html.Li("asset_type (EQUITY / ETF). If omitted we try to infer; you can also override later."),
                                                                html.Li("is_etf (true/false). If present, it wins."),
                                                                html.Li("name, currency, notes — ignored by the engine but useful for you."),
                                                            ],
                                                            className="text-muted",
                                                            style={"fontSize": "0.9rem", "marginBottom": "0.75rem"},
                                                        ),
                                                        html.Div("Examples:", className="text-muted", style={"fontSize": "0.85rem"}),
                                                        html.Ul(
                                                            [
                                                                html.Li("ticker,dollars,is_etf"),
                                                                html.Li("ticker,shares,price_per_share,asset_type"),
                                                                html.Li("ticker,percent,asset_type"),
                                                            ],
                                                            className="text-muted",
                                                            style={"fontSize": "0.85rem", "marginBottom": "0.75rem"},
                                                        ),
                                                    ]
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
                                                        "marginBottom": "0.5rem",
                                                    },
                                                ),
                                                dbc.Label("Or paste CSV content"),
                                                dbc.Textarea(
                                                    id="portfolio-text",
                                                    value=INITIAL_UI_STATE.get("portfolio_text", ""),
                                                    placeholder="ticker,dollars,is_etf\nAAPL,15000,false\nVOO,25000,true",
                                                    style={"height": "140px"},
                                                ),
                                                dbc.Collapse(
                                                    [
                                                        dbc.Label("Total portfolio value (only used when needed)"),
                                                        dbc.Input(
                                                            id="total-value",
                                                            type="number",
                                                            min=0,
                                                            step=1000,
                                                            value=INITIAL_UI_STATE.get("total_value", 0),
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
                                                html.Div(
                                                    "Step 2: ETF look-through. We determine the constituent holdings for any ETFs in your portfolio.",
                                                    className="text-muted mb-2",
                                                ),
                                                html.Div(id="etf-cache-status", className="mb-2"),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("ETF snapshot date (optional)"),
                                                                dbc.Input(
                                                                    id="etf-refresh-date",
                                                                    type="text",
                                                                    value=dt.date.today().isoformat(),
                                                                    placeholder="YYYY-MM-DD (defaults to today when fetching)",
                                                                ),
                                                                dbc.FormText(
                                                                    "Leave as-is for normal use. This only matters when fetching new holdings snapshots.",
                                                                    className="mt-1",
                                                                ),
                                                            ],
                                                            md=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Alpha Vantage API key (required only if fetching ETF holdings)"),
                                                                dbc.Input(
                                                                    id="api-key",
                                                                    type="password",
                                                                    value=INITIAL_UI_STATE.get("api_key", ""),
                                                                    placeholder="Paste your Alpha Vantage API key",
                                                                ),
                                                                dbc.FormText(
                                                                    "We only use the key client-side during this run to call Alpha Vantage ETF_PROFILE.",
                                                                    className="mt-1",
                                                                ),
                                                            ],
                                                            md=6,
                                                        ),
                                                    ],
                                                    className="g-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            dbc.Button("Fetch missing ETF holdings", id="btn-fetch-missing", color="primary", className="mt-2", n_clicks=0, style={"width": "100%"}),
                                                            md=6,
                                                        ),
                                                        dbc.Col(
                                                            dbc.Button("Refresh all ETF holdings", id="btn-refresh-all", color="secondary", outline=True, className="mt-2", n_clicks=0, style={"width": "100%"}),
                                                            md=6,
                                                        ),
                                                    ],
                                                    className="g-2",
                                                ),
                                                html.Div(id="etf-refresh-output", className="mt-2"),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Throttle (seconds per ETF fetch)"),
                                                                dbc.Input(
                                                                    id="per-etf-sleep",
                                                                    type="number",
                                                                    min=0,
                                                                    step=0.25,
                                                                    value=INITIAL_UI_STATE.get("per_etf_sleep", 0.0),
                                                                ),
                                                                dbc.FormText("Use 0–1s to reduce API rate-limit errors.", className="mt-1"),
                                                            ],
                                                            md=6,
                                                        ),
                                                    ],
                                                    className="g-2",
                                                ),
                                            ],
                                            label="2) ETF Holdings",
                                            tab_id="tab-etf",
                                        ),
                                        dbc.Tab(
                                            [
                                                html.Div(
                                                    "Step 3: Equity characteristics. We enrich holdings with sector/country/industry using the Nasdaq screener dataset.",
                                                    className="text-muted mb-2",
                                                ),
                                                html.Div("Security Master (Nasdaq screener CSV)", className="text-muted mb-1", style={"fontSize": "0.9rem"}),
                                                dbc.Label("Upload Nasdaq screener CSV"),
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
                                                        "marginBottom": "0.5rem",
                                                    },
                                                ),
                                                dbc.Button("Update security master from uploaded CSV", id="btn-refresh-master", color="primary", className="mt-2", n_clicks=0, style={"width": "100%"}),
                                                html.Div(id="security-master-status", className="mt-2"),
                                            ],
                                            label="3) Equity Data",
                                            tab_id="tab-master",
                                        ),
                                        dbc.Tab(
                                            [
                                                html.Div(
                                                    "Step 4: Run the look-through. This step uses cached ETF holdings (Step 2) and the security master (Step 3).",
                                                    className="text-muted mb-2",
                                                ),
                                                dbc.Button("Run Look-Through", id="btn-run", color="success", className="mt-2", n_clicks=0, style={"width": "100%"}),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(dbc.Button("Save Inputs", id="btn-save-inputs", color="secondary", outline=True, className="mt-2", n_clicks=0), md=6),
                                                        dbc.Col(dbc.Button("Reset", id="btn-reset", color="secondary", outline=True, className="mt-2", n_clicks=0), md=6),
                                                    ],
                                                    className="g-2",
                                                ),
                                                                                                dbc.Button("Load Saved Inputs", id="btn-load-inputs", color="secondary", outline=True, className="mt-2", n_clicks=0, style={"width": "100%"}),
html.Div(id="run-status", className="mt-2"),
                                            ],
                                            label="4) Run",
                                            tab_id="tab-run",
                                        ),
                                    ],
                                    active_tab="tab-portfolio",
                                )
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
                                    md=5,
                                ),
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
                                    md=7,
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
    Input("store-status-inputs", "data"),
    prevent_initial_call=True,
)
def mux_status(refresh_status, run_status, inputs_status):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""
    if trig == "store-status-refresh.data":
        return refresh_status
    if trig == "store-status-run.data":
        return run_status
    if trig == "store-status-inputs.data":
        return inputs_status
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
        help_text = "We only ask for this if we truly need it (i.e., when some positions use percent weights)."
        msg = "Upload or paste a portfolio to continue."
        return False, help_text, dbc.Alert(msg, color="secondary", className="py-2 mb-0")

    cols = [c.strip().lower() for c in df.columns]

    # Determine whether total portfolio value is actually needed.
    pct_col = df.columns[cols.index("percent")] if "percent" in cols else None
    dollars_col = df.columns[cols.index("dollars")] if "dollars" in cols else None
    shares_col = df.columns[cols.index("shares")] if "shares" in cols else None
    pps_col = df.columns[cols.index("price_per_share")] if "price_per_share" in cols else None

    need_total = False
    if pct_col is not None:
        pct = df[pct_col]
        pct_used = pct.notna()

        dollars_ok = False
        if dollars_col is not None:
            dollars_ok = df[dollars_col].notna()

        shares_ok = False
        if shares_col is not None and pps_col is not None:
            shares_ok = df[shares_col].notna() & df[pps_col].notna()

        # total needed if any row relies on percent and does NOT already have a resolvable value input
        need_total = bool((pct_used & ~(dollars_ok | shares_ok)).any())

    if need_total:
        total_help = "Needed: at least one row uses percent without dollars or shares*price. We'll compute dollars as percent × total."
    elif pct_col is not None:
        total_help = "Not needed: percent exists, but every row already has dollars or shares*price. We'll ignore total."
    else:
        total_help = "Not required unless you choose to provide percent weights."

    # Lightweight summary for the user
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
        base_msg = f"Loaded {n_rows:,} rows. Detected {n_etf:,} ETFs."

    if need_total:
        base_msg += " Total portfolio value is required to translate percent weights into dollars."
    return need_total, total_help, dbc.Alert(base_msg, color="info", className="py-2 mb-0")


@app.callback(
    Output("portfolio-text", "value"),
    Output("api-key", "value"),
    Output("total-value", "value"),
    Output("etf-refresh-date", "value"),
    Output("per-etf-sleep", "value"),
    Output("store-status-inputs", "data"),
    Input("btn-save-inputs", "n_clicks"),
    Input("btn-load-inputs", "n_clicks"),
    Input("upload-portfolio", "contents"),
    Input("upload-portfolio", "filename"),
    State("portfolio-text", "value"),
    State("api-key", "value"),
    State("total-value", "value"),
    State("etf-refresh-date", "value"),
    State("per-etf-sleep", "value"),
    prevent_initial_call=True,
)
def handle_inputs(save_clicks, load_clicks, pf_contents, pf_filename, pf_text, api_key, total_value, etf_refresh_date, per_etf_sleep):
    trig = callback_context.triggered[0]["prop_id"] if callback_context.triggered else ""

    if trig.startswith("btn-save-inputs"):
        save_ui_state(
            {
                "portfolio_text": pf_text or "",
                "api_key": api_key or "",
                "total_value": total_value or 0,
                "etf_refresh_date": etf_refresh_date or "",
                "per_etf_sleep": per_etf_sleep or 0.0,
            }
        )
        return no_update, no_update, no_update, no_update, no_update, {"message": "Saved inputs.", "kind": "success"}

    if trig.startswith("btn-load-inputs"):
        st = load_ui_state()
        return (
            st.get("portfolio_text", ""),
            st.get("api_key", ""),
            st.get("total_value", 0),
            st.get("etf_refresh_date", today_str()),
            st.get("per_etf_sleep", 0.0),
            {"message": "Loaded saved inputs.", "kind": "info"},
        )

    # If a portfolio file is uploaded, we don't overwrite pasted text automatically.
    return no_update, no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("etf-cache-status", "children"),
    Input("upload-portfolio", "contents"),
    Input("upload-portfolio", "filename"),
    Input("portfolio-text", "value"),
    Input("api-key", "value"),
    Input("store-status-refresh", "data"),  # re-render after a refresh action
)
def render_etf_cache_status(pf_contents, pf_filename, pf_text, api_key, _refresh_status):
    df = load_portfolio_df(pf_contents, pf_filename, pf_text)
    if df is None:
        return dbc.Alert("Add your portfolio in Step 1 to check ETF holdings status.", color="secondary", className="py-2 mb-0")

    cols = [c.strip().lower() for c in df.columns]

    # Identify ETFs as explicitly as we can
    etf_tickers = []
    tick_col = df.columns[cols.index("ticker")] if "ticker" in cols else None
    if tick_col is None:
        return dbc.Alert("Missing required column: ticker.", color="warning", className="py-2 mb-0")

    if "is_etf" in cols:
        s = df[df.columns[cols.index("is_etf")]]
        s_norm = s.fillna("").astype(str).str.strip().str.lower()
        etf_mask = s_norm.isin(["true", "1", "yes", "y", "etf"])
        etf_tickers = df.loc[etf_mask, tick_col].dropna().astype(str).tolist()
    elif "asset_type" in cols:
        s = df[df.columns[cols.index("asset_type")]].astype(str).str.upper()
        etf_mask = s.eq("ETF")
        etf_tickers = df.loc[etf_mask, tick_col].dropna().astype(str).tolist()
    else:
        return dbc.Alert(
            "ETF list unknown. Add an is_etf (true/false) or asset_type (ETF/EQUITY) column so we know which holdings to fetch.",
            color="secondary",
            className="py-2 mb-0",
        )

    etf_tickers = sorted({symbol_norm(t) for t in etf_tickers if str(t).strip()})
    if not etf_tickers:
        return dbc.Alert("No ETFs detected in your portfolio.", color="info", className="py-2 mb-0")

    today = today_str()
    rows = []
    missing = []
    stale = []
    fresh = []

    for t in etf_tickers:
        last = latest_cached_etf_holdings_date(t)
        if last is None:
            missing.append(t)
            rows.append(html.Li(f"{t}: no cached holdings"))
        else:
            rows.append(html.Li(f"{t}: cached as of {last}"))
            if last == today:
                fresh.append(t)
            else:
                stale.append(t)

    body = html.Div(
        [
            html.Div(f"ETFs detected: {len(etf_tickers)}. Cached snapshots are stored per ETF per date.", className="mb-1"),
            html.Ul(rows, style={"marginBottom": "0.5rem"}),
        ]
    )

    if missing and not api_key:
        return dbc.Alert(
            html.Div(
                [
                    body,
                    html.Div(f"Missing holdings for: {', '.join(missing)}. Add an API key above to fetch.", className="mt-1"),
                ]
            ),
            color="warning",
            className="py-2 mb-0",
        )

    if missing:
        return dbc.Alert(
            html.Div([body, html.Div(f"Missing holdings for: {', '.join(missing)}. Click “Fetch missing ETF holdings”.", className="mt-1")]),
            color="warning",
            className="py-2 mb-0",
        )

    if stale:
        return dbc.Alert(
            html.Div([body, html.Div(f"Cached data is not from today ({today}) for: {', '.join(stale)}. Refresh if you want today's snapshot.", className="mt-1")]),
            color="info",
            className="py-2 mb-0",
        )

    return dbc.Alert(html.Div([body, html.Div("All ETF holdings are cached for today.", className="mt-1")]), color="success", className="py-2 mb-0")


@app.callback(
    Output("etf-refresh-output", "children"),
    Output("store-status-refresh", "data"),
    Input("btn-fetch-missing", "n_clicks"),
    Input("btn-refresh-all", "n_clicks"),
    State("upload-portfolio", "contents"),
    State("upload-portfolio", "filename"),
    State("portfolio-text", "value"),
    State("etf-refresh-date", "value"),
    State("api-key", "value"),
    State("per-etf-sleep", "value"),
    prevent_initial_call=True,
)
def refresh_etf_holdings_ui(n_fetch_missing, n_refresh_all, pf_contents, pf_filename, pf_text, refresh_date, api_key, per_etf_sleep):
    trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""

    df = load_portfolio_df(pf_contents, pf_filename, pf_text)
    if df is None:
        return dbc.Alert("Add your portfolio in Step 1 first.", color="secondary", className="py-2 mb-0"), no_update

    if not api_key:
        return dbc.Alert("Add your Alpha Vantage API key to fetch ETF holdings.", color="warning", className="py-2 mb-0"), no_update

    cols = [c.strip().lower() for c in df.columns]
    if "ticker" not in cols:
        return dbc.Alert("Missing required column: ticker.", color="warning", className="py-2 mb-0"), no_update

    tick_col = df.columns[cols.index("ticker")]

    if "is_etf" in cols:
        s = df[df.columns[cols.index("is_etf")]]
        s_norm = s.fillna("").astype(str).str.strip().str.lower()
        etf_mask = s_norm.isin(["true", "1", "yes", "y", "etf"])
    elif "asset_type" in cols:
        s = df[df.columns[cols.index("asset_type")]].astype(str).str.upper()
        etf_mask = s.eq("ETF")
    else:
        return dbc.Alert("ETF list unknown. Add is_etf or asset_type to your portfolio.", color="secondary", className="py-2 mb-0"), no_update

    etf_tickers = sorted({symbol_norm(t) for t in df.loc[etf_mask, tick_col].dropna().astype(str).tolist() if str(t).strip()})
    if not etf_tickers:
        return dbc.Alert("No ETFs detected in your portfolio.", color="info", className="py-2 mb-0"), no_update

    # decide which ETFs to fetch
    if trig == "btn-fetch-missing":
        targets = [t for t in etf_tickers if latest_cached_etf_holdings_date(t) is None]
        if not targets:
            return dbc.Alert("No missing ETF holdings. Nothing to fetch.", color="info", className="py-2 mb-0"), {
                "message": "ETF holdings: nothing missing.",
                "kind": "info",
            }
    else:
        targets = etf_tickers

    d = (refresh_date or "").strip() or today_str()

    ok = []
    failed = []
    for t in targets:
        try:
            refresh_etf_holdings(api_key=api_key, etf_symbol=t, asof_date=d, sleep_seconds=float(per_etf_sleep or 0.0))
            ok.append(t)
        except Exception as e:
            failed.append(f"{t} ({str(e)})")

    if failed:
        out = dbc.Alert(
            html.Div(
                [
                    html.Div(f"Fetched {len(ok)} ETFs for {d}."),
                    html.Div("Failures:", className="mt-1 fw-semibold"),
                    html.Ul([html.Li(x) for x in failed]),
                ]
            ),
            color="warning",
            className="py-2 mb-0",
        )
        status = {"message": f"ETF holdings refresh completed with {len(failed)} failures.", "kind": "warning"}
        return out, status

    out = dbc.Alert(f"Fetched {len(ok)} ETFs for {d}.", color="success", className="py-2 mb-0")
    status = {"message": f"ETF holdings cached for {d}.", "kind": "success"}
    return out, status

# -------------------------
# Callback: refresh security master from Nasdaq screener upload
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

    import base64
    b64 = contents.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64)

    try:
        df = refresh_security_master_from_csv_bytes(raw_bytes, filename)
        meta = security_master_meta() or {}
        meta["rows"] = int(len(df))
        return meta, {"message": f"Security master updated ({len(df):,} rows).", "kind": "success"}
    except Exception as e:
        return no_update, {"message": f"Security master update failed: {str(e)}", "kind": "error"}


@app.callback(
    Output("security-master-status", "children"),
    Input("store-master-meta", "data"),
    Input("store-status-refresh", "data"),
)
def render_master_status(_meta_store, _refresh_status):
    meta = security_master_meta()
    if not meta:
        return dbc.Alert(
            "No security master found yet. Download the Nasdaq screener CSV, upload it above, then click “Update security master”.",
            color="secondary",
            className="py-2 mb-0",
        )

    uploaded_at = meta.get("uploaded_at", "unknown")
    row_count = meta.get("row_count", meta.get("rows", "unknown"))
    raw_copy = meta.get("raw_copy", "")

    lines = [
        html.Li(f"Last updated: {uploaded_at}"),
        html.Li(f"Rows: {row_count}"),
    ]
    if raw_copy:
        lines.append(html.Li(f"Saved raw copy: {raw_copy}"))

    return dbc.Alert(
        html.Div(
            [
                html.Div("Security master is ready.", className="fw-semibold"),
                html.Ul(lines, style={"marginBottom": "0.25rem"}),
                html.Div("To refresh: upload a newer Nasdaq screener CSV and click “Update security master”.", className="mt-1"),
            ]
        ),
        color="success",
        className="py-2 mb-0",
    )



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
    prevent_initial_call=True,
)
def run_lookthrough(n, api_key, pf_contents, pf_filename, pf_text, total_value):
    try:
        master = load_security_master()
        if master is None or len(master) == 0:
            return no_update, {"message": "Security master is empty. Go to Step 3 and upload the Nasdaq screener CSV.", "kind": "error"}

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

        # Normalize + compute position values
        pf_norm = normalize_portfolio_inputs(df, total_portfolio_value=float(total_value or 0.0))

        # Apply ETF classification (uses explicit columns + saved overrides)
        pf_class = apply_etf_classification(pf_norm)

        # Pre-check: if ETFs exist, ensure holdings are cached (we do NOT auto-fetch during Run)
        etfs = pf_class.loc[pf_class["asset_type"].astype(str).str.upper().eq("ETF"), "ticker_norm"].dropna().astype(str).tolist()
        etfs = sorted({symbol_norm(x) for x in etfs if str(x).strip()})
        missing = []
        for t in etfs:
            if load_cached_etf_holdings(t, asof_date=None) is None:
                missing.append(t)

        if missing:
            return (
                no_update,
                {
                    "message": f"Missing ETF holdings for: {', '.join(missing)}. Go to Step 2 to fetch/cache them, then run again.",
                    "kind": "error",
                },
            )

        exp = build_lookthrough_exposures(
            api_key=api_key or "",
            portfolio_df=pf_class,
            master_df=master,
            refresh_missing_etf_holdings=False,
            asof_date=None,  # use latest cached per ETF
            per_etf_sleep_seconds=0.0,
            progress_cb=None,
        )

        payload = exp.to_json(date_format="iso", orient="split")
        msg = f"Done. Exposure rows: {len(exp):,}."
        return payload, {"message": msg, "kind": "success"}

    except Exception as e:
        return no_update, {"message": f"Run failed: {str(e)}", "kind": "error"}


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
