# ============================================================
# MTF DASHBOARD — COMBINED
# STOCK + INDEX | SECTOR | MINI PANELS | DARK MODE
# ============================================================

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from ready_exporter import save_ready_stock
from mtf_signal_ui import prd_label, pvd_label, cip_label, rsi_div_label, entry_label, ready_label
from mtf_entry_candle import detect_entry_candle
from mtf_signal_logic import (
    detect_prd,
    detect_pvd,
    detect_rsi_bullish_divergence,
    detect_cip_change_in_polarity
)



print("📌 Starting MTF Combined Dashboard")

THEMES = {
    "light": {"bg": "#F2F2F2", "grid": "#C0C0C0", "font": "#000000"},
    "grey":  {"bg": "#E0E0E0", "grid": "#B0B0B0", "font": "#000000"},
    "soft":  {"bg": "#D4D6D9", "grid": "#A5A7AA", "font": "#000000"},
    "charcoal": {"bg": "#1E1E1E", "grid": "#3A3A3A", "font": "#EAEAEA"},
}

BULL = "#10C900"
BEAR_RED = "#EF5350"
BEAR_BLACK = "black"
MAX_CANDLES = 50


def rsi_wilder(series, period=14):
    series = series.astype(float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    for i in range(period + 1, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_indicators(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    df["BB_upper"] = df["SMA20"] + 2 * df["STD20"]
    df["BB_lower"] = df["SMA20"] - 2 * df["STD20"]
    df["RSI"] = rsi_wilder(df["Close"], 14)
    df["VolDir"] = np.where(df["Close"] >= df["Open"], "green", "red")
    return df


def compute_metrics(df):
    prev = df.iloc[-2]
    last = df.iloc[-1]
    rsi_d = round(last["RSI"], 2)
    sma_gap_d = round(((last["Close"] - last["SMA20"]) / last["SMA20"]) * 100, 2)
    move = round(((prev["High"] - prev["Low"]) / prev["Low"]) * 100, 2)
    range_pct_prev = round(((prev["High"] - prev["Low"]) / prev["Low"]) * 100, 2)
    vol_ok = prev["Volume"] >= df["Volume"].rolling(20).mean().iloc[-2]
    high_52 = df["High"].tail(252).max()
    pct_52w = round(((last["Close"] - high_52) / high_52) * 100, 2)
    return rsi_d, sma_gap_d, move, range_pct_prev, vol_ok, pct_52w

BASE = Path(__file__).parent
CSV_FILE = BASE / "mtf_rsi_nse_cash_signals.csv"
SECTOR_FILE = BASE / "sector_map.csv"



def load_symbols():
    try:
        df = pd.read_csv(CSV_FILE)
        return sorted(df["symbol"].dropna().astype(str).unique())
    except:
        return ["RELIANCE"]


stock_list = load_symbols()

index_list = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "FIN NIFTY": "^NSEFIN",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY IT": "^CNXIT",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY FIN SERVICE": "^CNXFINANCE",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY MIDCAP": "^CNXMIDCAP",
    "CRUDE OIL": "CL=F",
    "NATURAL GAS": "NG=F",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "USD / INR": "INR=X",
    "DOW JONES": "^DJI",
    "NASDAQ": "^IXIC",
}


def load_sector_map():
    if not SECTOR_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(SECTOR_FILE)
        df["nse_symbol"] = df["nse_symbol"].astype(str)
        return df
    except:
        return pd.DataFrame()


sector_df = load_sector_map()


@lru_cache(maxsize=64)
def fetch(symbol, is_index=False):
    tk = yf.Ticker(symbol if is_index else symbol + ".NS")
    df = tk.history(period="5y", interval="1d").reset_index()
    if df.empty:
        return None, None, None
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    d = add_indicators(df.copy()).tail(MAX_CANDLES)
    w = df.resample("W", on="Date").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna().reset_index()
    w = add_indicators(w).tail(MAX_CANDLES)
    m = df.resample("ME", on="Date").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna().reset_index()
    m = add_indicators(m).tail(MAX_CANDLES)
    return d, w, m


def build_chart(df, title, theme, vol_mode, bear_color, is_daily):
    t = THEMES[theme]
    bear = BEAR_BLACK if bear_color == "black" else BEAR_RED
    vol_colors = df["VolDir"] if vol_mode == "direction" else ["#7A7A7A"] * len(df)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=BULL,
        increasing_fillcolor=BULL,
        increasing_line_width=0.8,
        decreasing_line_color=bear,
        decreasing_fillcolor=bear,
        decreasing_line_width=0.8,
    ))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], line=dict(color="orange", width=1)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_upper"], line=dict(color="blue", width=1)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_lower"], line=dict(color="blue", width=1)))
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], marker_color=vol_colors, yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], yaxis="y3", line=dict(color="purple", width=1.2)))

    fig.add_hline(y=60, yref="y3", line_dash="dash", line_color="green")
    fig.add_hline(y=40, yref="y3", line_dash="dash", line_color="red")

    fig.update_layout(
        title=title,
        showlegend=False,
        paper_bgcolor=t["bg"],
        plot_bgcolor=t["bg"],
        font=dict(color=t["font"]),
        xaxis_rangeslider_visible=False,
        yaxis=dict(domain=[0.45, 1], gridcolor=t["grid"]),
        yaxis2=dict(domain=[0.25, 0.42], gridcolor=t["grid"]),
        yaxis3=dict(domain=[0, 0.22], gridcolor=t["grid"]),
        margin=dict(l=25, r=25, t=30, b=25),
    )

    if is_daily:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=True)
    else:
        fig.update_xaxes(showgrid=True)

    return fig


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div(id="info-strip", style={
        "padding": "6px 10px",
        "background": "#E0E0E0",
        "fontSize": "13px",
        "fontWeight": "600",
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "6px",
        "justifyContent": "space-between"
    }),

    dcc.Store(id="mode", data="stock"),
    dcc.Store(id="idx", data=0),
    dcc.Store(id="theme", data="grey"),
    dcc.Store(id="bear", data="red"),
    dcc.Store(id="vol", data="direction"),

    dcc.Interval(id="auto_refresh", interval=15*60*1000, n_intervals=0),

    html.Div([
        html.Button("STOCK MODE", id="btn_stock"),
        html.Button("INDEX MODE", id="btn_index"),
        html.Button("Light", id="t_light"),
        html.Button("Grey", id="t_grey"),
        html.Button("Soft", id="t_soft"),
        html.Button("Charcoal", id="t_charcoal"),
    ], style={"marginBottom": "6px"}),

    dcc.RadioItems(
        id="bear_toggle",
        options=[{"label": " Red", "value": "red"},
                 {"label": " Black", "value": "black"}],
        value="red", inline=True
    ),

    dcc.RadioItems(
        id="vol_toggle",
        options=[{"label": " Direction (Green/Red)", "value": "direction"},
                 {"label": " Grey Only", "value": "grey"}],
        value="direction", inline=True
    ),

    dcc.Dropdown(id="symbol"),

    html.Div([
        html.Button("Previous", id="prev"),
        html.Button("Next", id="next"),
    ], style={"margin": "8px 0"}),

    html.Div(id="charts")
],
    style={
        "backgroundColor": "#303030",
        "minHeight": "100vh",
        "padding": "6px",
    }
)


@app.callback(Output("mode","data"),
              Input("btn_stock","n_clicks"),
              Input("btn_index","n_clicks"),
              prevent_initial_call=True)
def mode_switch(_,__):
    return "index" if ctx.triggered_id=="btn_index" else "stock"


@app.callback(
    Output("theme","data"),
    Input("t_light","n_clicks"),
    Input("t_grey","n_clicks"),
    Input("t_soft","n_clicks"),
    Input("t_charcoal","n_clicks"),
    prevent_initial_call=True)
def theme_switch(a,b,c,d):
    return {
        "t_light":"light",
        "t_grey":"grey",
        "t_soft":"soft",
        "t_charcoal":"charcoal",
    }.get(ctx.triggered_id,"grey")


@app.callback(Output("bear","data"), Input("bear_toggle","value"))
def set_bear(v): return v


@app.callback(Output("vol","data"), Input("vol_toggle","value"))
def set_vol(v): return v


@app.callback(
    Output("symbol","options"),
    Output("symbol","value"),
    Input("mode","data"),
    Input("idx","data"))
def symbol_list_cb(mode, idx):
    syms = list(index_list.keys()) if mode=="index" else stock_list
    return [{"label":s,"value":s} for s in syms], syms[idx]


@app.callback(
    Output("idx","data"),
    Input("prev","n_clicks"),
    Input("next","n_clicks"),
    State("idx","data"),
    Input("mode","data"),
    prevent_initial_call=True)
def nav(_,__,idx,mode):
    syms = list(index_list.keys()) if mode=="index" else stock_list
    return (idx-1)%len(syms) if ctx.triggered_id=="prev" else (idx+1)%len(syms)


@app.callback(
    Output("info-strip","children"),
    Output("charts","children"),
    Input("symbol","value"),
    Input("mode","data"),
    Input("theme","data"),
    Input("bear","data"),
    Input("vol","data"),
    Input("auto_refresh","n_intervals")
)
def update(symbol, mode, theme, bear, vol, _):

    d,w,m = fetch(index_list[symbol],True) if mode=="index" else fetch(symbol,False)
    if d is None: return "⚠ No data", []

    rsi_d, sma_gap_d, move, rng_prev, vol_ok, pct_52w = compute_metrics(d)

    rsi_w = round(w.iloc[-1]["RSI"], 2)
    rsi_m = round(m.iloc[-1]["RSI"], 2)

    entry_pattern, entry_vol_ok, entry_idx = detect_entry_candle(d)

    prd_d = detect_prd(d["Close"].values, d["RSI"].values)
    prd_w = detect_prd(w["Close"].values, w["RSI"].values)
    prd_m = detect_prd(m["Close"].values, m["RSI"].values)

    pvd_d = detect_pvd(d["Close"].values, d["Volume"].values)
    pvd_w = detect_pvd(w["Close"].values, w["Volume"].values)
    pvd_m = detect_pvd(m["Close"].values, m["Volume"].values)

    rsi_div_d = detect_rsi_bullish_divergence(d["Close"].values, d["RSI"].values)
    rsi_div_w = detect_rsi_bullish_divergence(w["Close"].values, w["RSI"].values)
    rsi_div_m = detect_rsi_bullish_divergence(m["Close"].values, m["RSI"].values)

    cip_w, cip_level_w, _ = detect_cip_change_in_polarity(
        w["High"].values, w["Low"].values, w["Close"].values
    )

    cip_m, cip_level_m, _ = detect_cip_change_in_polarity(
        m["High"].values, m["Low"].values, m["Close"].values
    )

    # --------------------------
# FLEXIBLE ENTRY CONFIRMATION
# --------------------------

    any_strength = (
        prd_d or pvd_d or rsi_div_d or
        prd_w or pvd_w or cip_w or rsi_div_w or
        prd_m or pvd_m or rsi_div_m
    )

    trade_ready = False

    if entry_pattern and entry_vol_ok and any_strength:
        trade_ready = True
   
    if trade_ready:
        save_ready_stock(symbol, d.iloc[-1]["High"])





    gap_m = round(((m.iloc[-1]["Close"] - m.iloc[-1]["SMA20"]) / m.iloc[-1]["SMA20"]) * 100, 2)
    gap_w = round(((w.iloc[-1]["Close"] - w.iloc[-1]["SMA20"]) / w.iloc[-1]["SMA20"]) * 100, 2)
    gap_d = round(((d.iloc[-1]["Close"] - d.iloc[-1]["SMA20"]) / d.iloc[-1]["SMA20"]) * 100, 2)

    yahoo_sector = "UNKNOWN"
    sector_name = "UNKNOWN"
    sector_symbol = None

    if not sector_df.empty:
        row = sector_df[sector_df["nse_symbol"] == symbol]
        if len(row):
            yahoo_sector = row.iloc[0]["yahoo_sector"]
            sector_name = row.iloc[0]["sector_index_name"]
            if row.iloc[0]["sector_index_symbol"] != "NO_INDEX":
                sector_symbol = row.iloc[0]["sector_index_symbol"]

    if yahoo_sector == "UNKNOWN":
        try:
            tk = yf.Ticker(symbol + ".NS")
            info = tk.info
            yahoo_sector = info.get("sector", "UNKNOWN")
        except:
            pass

    sector_status = html.Span("NO INDEX", style={"color":"red"})
    bullish_sector = False

    if sector_symbol:
        sd,sw,sm = fetch(sector_symbol,True)
        if sd is not None:
            bullish_sector = (sd.iloc[-1]["RSI"]>=60 and sw.iloc[-1]["RSI"]>=60 and sm.iloc[-1]["RSI"]>=60)
            sector_status = html.Span("BULLISH" if bullish_sector else "BEARISH",
                                      style={"color": "green" if bullish_sector else "red"})

    # --------------------------
    # STAR RATING (UPDATED RULES)
    # --------------------------
    stars = 0

    # 1) RSI GREEN (Monthly + Weekly ONLY)
    if rsi_m >= 60 and rsi_w >= 60:
        stars += 1

    # 2) Sector bullish
    if bullish_sector:
        stars += 1

    # 3) Above SMA all TF
    if m.iloc[-1]["Close"] > m.iloc[-1]["SMA20"] and \
       w.iloc[-1]["Close"] > w.iloc[-1]["SMA20"] and \
       d.iloc[-1]["Close"] > d.iloc[-1]["SMA20"]:
        stars += 1

    # 4) PD VOL green
    pd_vol_green = d.iloc[-2]["Volume"] >= d["Volume"].rolling(20).mean().iloc[-2]
    if pd_vol_green:
        stars += 1

    # 5) DOJI / HAMMER
    candle_name = ""
    prev = d.iloc[-2]

    body = abs(prev["Close"] - prev["Open"])
    upper = prev["High"] - max(prev["Close"], prev["Open"])
    lower = min(prev["Close"], prev["Open"]) - prev["Low"]
    range_pct = ((prev["High"] - prev["Low"]) / prev["Low"]) * 100

    # hammer
    if lower > upper and lower >= 2 * body:
        candle_name = "HAMMER"
        stars += 1
    # doji
    elif body <= 0.30 * (upper + lower) and abs(upper - lower) <= 0.1 * (upper + lower) and range_pct <= 2.5:
        candle_name = "DOJI"
        stars += 1

    # 6) PDC bullish and < 5%
    if move > 0 and move < 5:
        stars += 1

    stars_display = " ".join(["⭐"] * stars) if stars > 0 else "—"
    star_badge = html.Span(
        f"{stars_display}   {stars} STAR TRADE",
        style={
            "background":"yellow",
            "color":"black",
            "padding":"4px 6px",
            "borderRadius":"6px",
            "fontWeight":"800"
        }
    )

    def col(ok): return "green" if ok else "red"

    now = datetime.now().strftime("%d-%b-%Y %H:%M")

    header = [
        html.Div([
            star_badge,
            html.Span(f"  {symbol} | Sector: {sector_name} | From Yahoo: {yahoo_sector}")
        ], style={"flex":"1"}),

        html.Div([
            html.Span("52W: "),
            html.Span(f"{pct_52w}% | "),
            html.Span("Sector: "),
            sector_status,
            f" | {'🟢 STOCK' if mode=='stock' else '🔵 INDEX'}"
        ], style={"flex":"1","textAlign":"center"}),

        html.Div([
            f"Last Update: {now} | Auto: 15m"
        ], style={"flex":"1","textAlign":"right"})
    ]

    monthly_panel = html.Div(
        [
            html.B("Monthly"),
            html.Span(f" | RSI: {m.iloc[-1]['RSI']:.2f}",
                    style={"color": col(m.iloc[-1]['RSI']>=60)}),

            prd_label(prd_m),
            prd_label(pvd_m),
            prd_label(rsi_div_m),
            prd_label(cip_m),

            html.Span(f" | {'≥' if gap_m>=0 else '≤'} SMA {gap_m:+.2f}%",
                    style={"color":"green" if gap_m>=0 else "red"})
        ],
        style={"background":"#EFEFEF","padding":"6px","borderRadius":"6px","display":"flex","gap":"8px"}
    )


    weekly_panel = html.Div(
        [
            html.B("Weekly"),
            html.Span(f" | RSI: {w.iloc[-1]['RSI']:.2f}",
                     style={"color": col(w.iloc[-1]['RSI']>=60)}),

            prd_label(prd_w),
            prd_label(pvd_w),
            prd_label(rsi_div_w),
            prd_label(cip_w),

           

            html.Span(f" | {'≥' if gap_w>=0 else '≤'} SMA {gap_w:+.2f}%",
                     style={"color":"green" if gap_w>=0 else "red"})
        ],
        style={"background":"#EFEFEF","padding":"6px","borderRadius":"6px","display":"flex","gap":"8px"}
    )


    d_panel = html.Div(
[
    html.B("Daily"),

    # RSI
    html.Span(f" | RSI {d.iloc[-1]['RSI']:.1f}",
              style={"color":"green" if d.iloc[-1]['RSI']>=60 else "red"}),

    # SIGNALS (blink only if present)
    prd_label(prd_d),
    pvd_label(pvd_d),
    rsi_div_label(rsi_div_d),

    # ENTRY CANDLE (blink like PRD)
    entry_label(entry_pattern),

    # VOLUME CONFIRMATION
    html.Span(" | VOL OK" if entry_vol_ok else " | VOL LOW",
              style={"color":"green" if entry_vol_ok else "red"}),

    # SMA %
    html.Span(f" | SMA {gap_d:+.2f}%",
              style={"color":"green" if gap_d>=0 else "red"}),

    # PDC
    html.Span(f" | PDC {move:+.2f}%",
              style={"color":"green" if move>0 else "red"}),

    # TRADE READY (blink)
    ready_label(trade_ready),
],
style={
    "background":"#EFEFEF",
    "padding":"6px",
    "borderRadius":"6px",
    "display":"flex",
    "gap":"8px",
    "flexWrap":"wrap"
}
)




    col_style = {"width":"33%","display":"inline-block","verticalAlign":"top","padding":"6px","boxSizing":"border-box"}

    charts = html.Div([
        html.Div([monthly_panel,
                  dcc.Graph(figure=build_chart(m,f"{symbol} — Monthly",theme,vol,bear,False))], style=col_style),
        html.Div([weekly_panel,
                  dcc.Graph(figure=build_chart(w,f"{symbol} — Weekly",theme,vol,bear,False))], style=col_style),
        html.Div([d_panel,
                  dcc.Graph(figure=build_chart(d,f"{symbol} — Daily",theme,vol,bear,True))], style=col_style),
    ])

    return header, charts


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)




