# ==================================================
# MTF SIGNAL UI HELPERS
# ==================================================
# Contains ONLY UI label helpers for dashboard
# ==================================================

from dash import html


def prd_label(show):
    if not show:
        return ""
    return html.Span(" | PRD", className="prd-blink")


def cip_label(show):
    if not show:
        return ""
    return html.Span(" | CIP", className="prd-blink")


def pvd_label(show):
    if not show:
        return ""
    return html.Span(" | PVD", className="prd-blink")


def rsi_div_label(show):
    if not show:
        return ""
    return html.Span(" | RSI-DIV", className="prd-blink")


def signal_group(prd=False, cip=False, pvd=False, rsi_div=False):
    return [
        prd_label(prd),
        cip_label(cip),
        pvd_label(pvd),
        rsi_div_label(rsi_div),
    ]
def entry_label(name):
    if not name:
        return ""
    return html.Span(f" | {name}", className="prd-blink")


def ready_label(is_ready):
    if is_ready:
        return html.Span(" | READY", className="prd-blink")
    return html.Span(" | WAIT", style={"color":"red","fontWeight":"800"})

