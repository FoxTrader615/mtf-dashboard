import pandas as pd
import yfinance as yf
import os
import math
from datetime import datetime
from tqdm import tqdm
import requests


# ===============================
# PATHS
# ===============================
from pathlib import Path

BASE = Path(__file__).parent

FILE_PATH = BASE / "NSE_CASH_YAHOO_CORRECT.csv"
CACHE_DIR = BASE / "cache"
LOG_FILE = BASE / "scanner_log.txt"

CACHE_DIR.mkdir(exist_ok=True)



# ===============================
# TELEGRAM
# ===============================
BOT_TOKEN = "7781237545:AAExfLsa_u3bSUS5VOXc7c0_l05UgykRhlI"
CHAT_ID = "1724385120"


def send_alert(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        log(f"[TELEGRAM ERROR] {e}")


# ===============================
# LOGGING
# ===============================
def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


log("=========== SCANNER STARTED ===========")


# ===============================
# LOAD SYMBOL LIST
# ===============================
df_symbols = pd.read_csv(FILE_PATH)

for col in ["symbol","Symbol","SYMBOL","ticker","Ticker","TICKER"]:
    if col in df_symbols.columns:
        symbols = df_symbols[col].astype(str).tolist()
        break

symbols = [s if s.endswith(".NS") else f"{s}.NS" for s in symbols]


# ===============================
# RSI (TRUE WILDER)
# ===============================
def rsi_wilder(series, period=14):
    series = series.astype(float)
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    for i in range(period + 1, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1]*(period-1)+gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1]*(period-1)+loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ===============================
# CACHE HELPERS
# ===============================
def normalize_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    df.index.name = "date"

    if "close" not in df.columns and "adj close" in df.columns:
        df["close"] = df["adj close"]

    return df


def load_cache(ticker):
    path = f"{CACHE_DIR}\\{ticker}.csv"
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path, parse_dates=["date"], index_col="date")
    except:
        log(f"[CACHE CORRUPT] {ticker} — rebuilding")
        return None


def save_cache(ticker, df):
    df.to_csv(f"{CACHE_DIR}\\{ticker}.csv")


def download(ticker):
    try:
        log(f"[DOWNLOAD] {ticker}")
        df = yf.download(
            ticker,
            period="2y",
            interval="1d",
            progress=False
        )

        if len(df) < 25:
            log(f"[SKIP] {ticker} — not enough candles")
            return None

        df = normalize_df(df)
        save_cache(ticker, df)
        return df

    except Exception as e:
        log(f"[ERROR DOWNLOAD] {ticker}: {e}")
        return None


# ===============================
# PROCESS SYMBOL
# ===============================
def process_symbol(df, ticker):

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.set_index("date")

    close = df["close"]

    # DAILY
    rsi_daily = rsi_wilder(close)

    # WEEKLY — NSE FRIDAY
    weekly = df.resample("W-FRI").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    rsi_weekly = rsi_wilder(weekly["close"])

    # MONTHLY — END OF MONTH
    monthly = df.resample("ME").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    rsi_monthly = rsi_wilder(monthly["close"])

    # ===== FINAL RSI VALUES =====
    d = round(rsi_daily.iloc[-1], 2)       # Daily exact
    w = round(rsi_weekly.iloc[-1], 2)      # Weekly exact
    m = math.ceil(rsi_monthly.iloc[-1])   # Monthly optimistic

    log(f"[RSI] {ticker}  D={d}  W={w}  M={m}")

    return d, w, m


# ===============================
# MAIN LOOP
# ===============================
results = []

for ticker in tqdm(symbols, desc="Scanning NSE"):
    log(f"--- {ticker} ---")

    # ========== CACHE + REFRESH LAST 10 DAYS ==========
    df = load_cache(ticker)

    try:
        fresh = yf.download(
            ticker,
            period="10d",
            interval="1d",
            progress=False
        )

        if fresh is not None and len(fresh) > 0:
            fresh = normalize_df(fresh)

            if df is not None:
                df = pd.concat([df[:-10], fresh]).drop_duplicates()
            else:
                df = fresh

            save_cache(ticker, df)
            log(f"[REFRESH] Updated last 10 candles for {ticker}")

    except Exception as e:
        log(f"[REFRESH ERROR] {ticker}: {e}")

    # If still missing → full download
    if df is None or len(df) < 25:
        df = download(ticker)

    if df is None:
        log(f"[SKIP] {ticker} — no data available")
        continue

    try:
        d, w, m = process_symbol(df, ticker)

        daily_ok = (37 <= d <= 47) or (57 <= d <= 63)
        entry = (m >= 60 and w >= 60 and daily_ok)

        if entry:
            log(f"[ENTRY READY] {ticker} (D={d}, W={w}, M={m})")

            send_alert(
                f"📢 MTF RSI READY\n"
                f"{ticker.replace('.NS','')}\n"
                f"Daily: {d}\nWeekly: {w}\nMonthly: {m}"
            )

        results.append({
            "symbol": ticker.replace(".NS", ""),
            "rsi_daily": d,
            "rsi_weekly": w,
            "rsi_monthly": m,
            "ENTRY_READY": entry
        })

    except Exception as e:
        log(f"[PROCESS ERROR] {ticker}: {e}")


# ===============================
# SAVE OUTPUT
# ===============================
df = pd.DataFrame(results)

df.to_csv(rf"{BASE}\mtf_rsi_nse_cash_final.csv", index=False)
df[df["ENTRY_READY"]].to_csv(
    rf"{BASE}\mtf_rsi_nse_cash_signals.csv",
    index=False
)

log("=========== SCANNER FINISHED ==========")

print("\nDONE ✔ — open scanner_log.txt to review logs")

