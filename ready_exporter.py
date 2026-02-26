import pandas as pd
from datetime import datetime
from pathlib import Path

FILE = Path("ready_stocks.csv")

def save_ready_stock(symbol, high):

    row = {
        "symbol": symbol,
        "high": round(float(high), 2),
        "date": datetime.now().strftime("%Y-%m-%d")
    }

    if FILE.exists():
        df = pd.read_csv(FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.drop_duplicates(subset=["symbol","date"], inplace=True)

    df.to_csv(FILE, index=False)

    print("✅ READY saved:", row)
