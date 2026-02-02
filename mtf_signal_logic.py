# ============================================================
# MTF SIGNAL LOGIC ENGINE (PRO)
# ============================================================
# Contains ONLY signal detection logic:
#   - PRD (Positive Reverse Divergence)  [Price HL + RSI LL]
#   - Bullish RSI Divergence             [Price LL/EL + RSI HL]
#   - PVD (Price drifting down + volume down)
#   - CIP (Change In Polarity)           [Resistance -> Support retest]
#
# IMPORTANT:
# - Keep this file pure logic (no Dash/UI code).
# - Dashboard/scanner/backtest should import from here.
# ============================================================

import numpy as np


# -------------------------------
# Helpers: swing points
# -------------------------------
def find_swing_lows(series, left=3, right=3):
    """
    A swing low at i means:
    series[i] is the minimum in window [i-left, i+right]
    """
    s = np.asarray(series, dtype=float)
    idxs = []
    for i in range(left, len(s) - right):
        window = s[i - left : i + right + 1]
        if s[i] == np.min(window):
            idxs.append(i)
    return idxs


def find_swing_highs(series, left=3, right=3):
    """
    A swing high at i means:
    series[i] is the maximum in window [i-left, i+right]
    """
    s = np.asarray(series, dtype=float)
    idxs = []
    for i in range(left, len(s) - right):
        window = s[i - left : i + right + 1]
        if s[i] == np.max(window):
            idxs.append(i)
    return idxs


def pct(a, b):
    """Percent difference between a and b using b as base."""
    if b == 0:
        return 999999.0
    return (a - b) / b * 100.0


# -------------------------------
# PRD (Positive Reverse Divergence)
# Definition (your PRD):
#   Price makes Higher Low (HL)
#   RSI makes Lower Low (LL)
# -------------------------------
def detect_prd(price_close, rsi, left=3, right=3):
    """
    Uses swing lows on PRICE only (more stable), compares RSI at same swing points.
    Returns True/False.
    """
    p = np.asarray(price_close, dtype=float)
    r = np.asarray(rsi, dtype=float)

    lows = find_swing_lows(p, left=left, right=right)
    if len(lows) < 2:
        return False

    i1, i2 = lows[-2], lows[-1]
    # Price HL
    price_hl = p[i2] > p[i1]
    # RSI LL
    rsi_ll = r[i2] < r[i1]

    return bool(price_hl and rsi_ll)


# -------------------------------
# Bullish RSI Divergence (normal)
# Locked Point 12:
#   Price makes Equal/Lower Low
#   RSI makes Higher Low
# -------------------------------
def detect_rsi_bullish_divergence(price_close, rsi, left=3, right=3):
    """
    Returns True/False.
    """
    p = np.asarray(price_close, dtype=float)
    r = np.asarray(rsi, dtype=float)

    lows = find_swing_lows(p, left=left, right=right)
    if len(lows) < 2:
        return False

    i1, i2 = lows[-2], lows[-1]
    price_ll_or_equal = p[i2] <= p[i1]
    rsi_hl = r[i2] > r[i1]

    return bool(price_ll_or_equal and rsi_hl)


# -------------------------------
# PVD (Price-Volume Divergence)
# Locked Point 10:
#   Price drifts down slowly while volume decreases
# -------------------------------
def detect_pvd(price_close, volume, lookback=12, max_price_drop_pct=6.0):
    """
    High-signal, low-noise PVD:
      - price is down from start to end in lookback window
      - volume is down from start to end in lookback window
      - price drop is not "crash" (optional filter), by default <= 6%
    """
    p = np.asarray(price_close, dtype=float)
    v = np.asarray(volume, dtype=float)

    if len(p) < lookback + 1:
        return False

    p0, p1 = p[-lookback], p[-1]
    v0, v1 = v[-lookback], v[-1]

    price_down = p1 < p0
    vol_down = v1 < v0

    drop_pct = abs(pct(p1, p0))
    slow_down = drop_pct <= max_price_drop_pct  # “slow drift” filter

    return bool(price_down and vol_down and slow_down)


# -------------------------------
# CIP (Change In Polarity) — Monthly/Weekly
# Definition (your CIP):
#   1) A resistance level exists (pivot high)
#   2) Price breaks above it (close above level + buffer)
#   3) Later price retests near that level (within tolerance)
#   4) Retest holds: close stays above level (support confirmed)
# -------------------------------
def detect_cip_change_in_polarity(
    high,
    low,
    close,
    pivot_left=3,
    pivot_right=3,
    search_lookback=60,
    breakout_buffer_pct=0.20,
    retest_tolerance_pct=0.50,
    retest_window=20,
):
    """
    Returns:
      (cip_bool, level, details_dict)

    Interpretation:
      - Use on Weekly/Monthly OHLC arrays.
      - "level" is the pivot resistance that flipped to support.

    Parameters:
      search_lookback: how far back to search for pivot/high + breakout event.
      breakout_buffer_pct: close must exceed level by this % to count as breakout.
      retest_tolerance_pct: low must come within this % of the level to count as retest.
      retest_window: after breakout, look forward within last N bars to find retest hold.
    """
    H = np.asarray(high, dtype=float)
    L = np.asarray(low, dtype=float)
    C = np.asarray(close, dtype=float)

    n = len(C)
    if n < max(search_lookback, pivot_left + pivot_right + 5):
        return (False, None, {"reason": "not_enough_bars"})

    # Focus on last portion (recent structure)
    start = max(0, n - search_lookback)
    Hs = H[start:]
    Ls = L[start:]
    Cs = C[start:]

    # Find pivot highs (resistance candidates) in this window
    pivots = find_swing_highs(Hs, left=pivot_left, right=pivot_right)
    if not pivots:
        return (False, None, {"reason": "no_pivot_high"})

    # We'll scan pivots from most recent to older (prefer latest meaningful resistance)
    pivots_sorted = sorted(pivots, reverse=True)

    for pi in pivots_sorted:
        level = Hs[pi]

        # 1) Breakout: any later candle close > level + buffer
        buffer_level = level * (1.0 + breakout_buffer_pct / 100.0)

        breakout_idx = None
        for j in range(pi + 1, len(Cs)):
            if Cs[j] > buffer_level:
                breakout_idx = j
                break

        if breakout_idx is None:
            continue

        # 2) Retest window after breakout (but near recent end)
        # We'll look from breakout_idx+1 to end, but cap to last retest_window bars.
        retest_start = breakout_idx + 1
        if retest_start >= len(Cs):
            continue

        retest_scan_start = max(retest_start, len(Cs) - retest_window)

        cip_found = False
        retest_j = None

        for j in range(retest_scan_start, len(Cs)):
            # retest touches near level
            touch = abs(pct(Ls[j], level)) <= retest_tolerance_pct
            # support holds: close >= level (small buffer allowed)
            holds = Cs[j] >= level

            if touch and holds:
                cip_found = True
                retest_j = j
                break

        if cip_found:
            details = {
                "level": float(level),
                "breakout_close": float(Cs[breakout_idx]),
                "breakout_idx_window": int(breakout_idx),
                "retest_idx_window": int(retest_j),
                "params": {
                    "breakout_buffer_pct": breakout_buffer_pct,
                    "retest_tolerance_pct": retest_tolerance_pct,
                    "retest_window": retest_window,
                    "search_lookback": search_lookback,
                },
            }
            return (True, float(level), details)

    return (False, None, {"reason": "no_breakout_retest_match"})
