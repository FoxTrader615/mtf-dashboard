# ==================================================
# MTF ENTRY CANDLE SELECTOR MODULE
# ==================================================
# Uses candle patterns from mtf_candle_patterns.py
# Applies 4 PM rule + volume filter
# ==================================================

from datetime import datetime, time
from mtf_candle_patterns import *

# -------------------------
# 4 PM Rule
# -------------------------
def get_entry_index(df):
    """
    After 4 PM → today's candle is completed
    Before 4 PM → yesterday's candle is used
    """
    now = datetime.now().time()
    return -1 if now >= time(16, 0) else -2


# -------------------------
# Volume Filter
# -------------------------
def volume_ok(df, idx, lookback=20, min_ratio=0.5):
    """
    Entry candle volume must be >= 50% of last 20 day average volume
    """
    avg_vol = df["Volume"].rolling(lookback).mean().iloc[idx]
    return df.iloc[idx]["Volume"] >= min_ratio * avg_vol


# -------------------------
# Entry Candle Detector
# -------------------------
def detect_entry_candle(df):
    """
    Returns:
        pattern_name (str or None)
        volume_ok (bool)
        index (int)
    """

    idx = get_entry_index(df)
    c = df.iloc[idx]
    p = df.iloc[idx-1] if idx-1 >= -len(df) else None
    p2 = df.iloc[idx-2] if idx-2 >= -len(df) else None

    pattern = None

    # ---- Single candle patterns ----
    if is_hammer(c):
        pattern = "HAMMER"
    elif is_inverted_hammer(c):
        pattern = "INVERTED_HAMMER"
    elif is_doji(c):
        pattern = "DOJI"
    elif is_spinning_top(c):
        pattern = "SPINNING_TOP"
    elif is_bullish_marubozu(c):
        pattern = "MARUBOZU"

    # ---- Two candle patterns ----
    elif p is not None and is_bullish_engulfing(p, c):
        pattern = "ENGULFING"
    elif p is not None and is_bullish_harami(p, c):
        pattern = "HARAMI"
    elif p is not None and is_piercing(p, c):
        pattern = "PIERCING"
    elif p is not None and is_tweezer_bottom(p, c):
        pattern = "TWEEZER"
    elif p is not None and is_bullish_kicker(p, c):
        pattern = "KICKER"
    elif p is not None and is_gap_support(p, c):
        pattern = "GAP_SUPPORT"

    # ---- Three candle patterns ----
    elif p is not None and p2 is not None and is_morning_star(p2, p, c):
        pattern = "MORNING_STAR"
    elif p is not None and p2 is not None and is_three_white_soldiers(p2, p, c):
        pattern = "3_SOLDIERS"

    vol_ok = volume_ok(df, idx)

    return pattern, vol_ok, idx
