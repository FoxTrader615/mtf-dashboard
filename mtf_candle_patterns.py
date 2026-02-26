# ==================================================
# MTF CANDLE PATTERN RULEBOOK — PRO VERSION
# ==================================================
# Non-overlapping, trader-realistic, backtest safe
# ==================================================

def body(c):
    return abs(c["Close"] - c["Open"])

def upper_wick(c):
    return c["High"] - max(c["Close"], c["Open"])

def lower_wick(c):
    return min(c["Close"], c["Open"]) - c["Low"]

def range_(c):
    return c["High"] - c["Low"]

# ---------- BASIC FILTER ----------
def valid_range(c):
    return range_(c) > 0

# ---------------- DOJI ----------------
def is_doji(c):
    return (
        valid_range(c) and
        body(c) <= 0.12 * range_(c) and
        abs(upper_wick(c) - lower_wick(c)) <= 0.15 * range_(c)
    )

# ---------------- SPINNING TOP ----------------
def is_spinning_top(c):
    return (
        valid_range(c) and
        body(c) > 0.12 * range_(c) and
        body(c) <= 0.30 * range_(c) and
        upper_wick(c) > body(c) and
        lower_wick(c) > body(c)
    )

# ---------------- HAMMER ----------------
def is_hammer(c):
    return (
        valid_range(c) and
        body(c) >= 0.12 * range_(c) and
        body(c) <= 0.35 * range_(c) and
        lower_wick(c) >= 2.5 * body(c) and
        lower_wick(c) > upper_wick(c)
    )

# ---------------- INVERTED HAMMER ----------------
def is_inverted_hammer(c):
    return (
        valid_range(c) and
        body(c) >= 0.12 * range_(c) and
        body(c) <= 0.35 * range_(c) and
        upper_wick(c) >= 2.5 * body(c) and
        upper_wick(c) > lower_wick(c)
    )

# ---------------- BULLISH MARUBOZU ----------------
def is_bullish_marubozu(c):
    return (
        valid_range(c) and
        c["Close"] > c["Open"] and
        upper_wick(c) <= 0.1 * body(c) and
        lower_wick(c) <= 0.1 * body(c) and
        body(c) >= 0.75 * range_(c)
    )

# ---------------- BULLISH ENGULFING ----------------
def is_bullish_engulfing(prev, curr):
    return (
        prev["Close"] < prev["Open"] and
        curr["Close"] > curr["Open"] and
        curr["Open"] < prev["Close"] and
        curr["Close"] > prev["Open"] and
        body(curr) >= 1.2 * body(prev)
    )

# ---------------- BULLISH HARAMI ----------------
def is_bullish_harami(prev, curr):
    return (
        prev["Close"] < prev["Open"] and
        curr["Close"] > curr["Open"] and
        curr["High"] < prev["High"] and
        curr["Low"] > prev["Low"]
    )

# ---------------- PIERCING ----------------
def is_piercing(prev, curr):
    mid = (prev["Open"] + prev["Close"]) / 2
    return (
        prev["Close"] < prev["Open"] and
        curr["Close"] > mid and
        curr["Close"] < prev["Open"]
    )

# ---------------- TWEEZER BOTTOM ----------------
def is_tweezer_bottom(prev, curr):
    return abs(prev["Low"] - curr["Low"]) <= 0.05 * range_(prev)

# ---------------- MORNING STAR ----------------
def is_morning_star(c1, c2, c3):
    return (
        c1["Close"] < c1["Open"] and
        body(c2) <= 0.30 * range_(c2) and
        c3["Close"] > (c1["Open"] + c1["Close"]) / 2
    )

# ---------------- THREE WHITE SOLDIERS ----------------
def is_three_white_soldiers(c1, c2, c3):
    return (
        c1["Close"] > c1["Open"] and
        c2["Close"] > c2["Open"] and
        c3["Close"] > c3["Open"] and
        c2["Close"] > c1["Close"] and
        c3["Close"] > c2["Close"]
    )

# ---------------- BULLISH KICKER ----------------
def is_bullish_kicker(prev, curr):
    return (
        prev["Close"] < prev["Open"] and
        curr["Open"] > prev["High"] and
        curr["Close"] > curr["Open"]
    )

# ---------------- GAP SUPPORT ----------------
def is_gap_support(prev, curr):
    return curr["Low"] > prev["High"]
