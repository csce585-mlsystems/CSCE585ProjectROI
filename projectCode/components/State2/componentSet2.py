# componentSet2.py
from reactpy import component, html
from constants import PAGE_WRAPPER, UI, STYLE_CARD_WIDE

def pct(x):
    try:
        return f"{round(float(x)*100, 1)}%"
    except:
        return "—"

def money(x):
    try:
        v = float(x)
        s = f"{v:,.2f}"
        return f"${s}"
    except:
        return f"${x}"

@component
def DashboardView(data=None):
    d = data or {}
    inputs = d.get("inputs", {})
    recs = d.get("recommendations", [])
    cagr = d.get("required_cagr", 0.0)

    items = [
        html.h1({"style": {"color": UI["text_color"], "marginTop": 0}}, "Dashboard"),
        html.p({"style": {"color": UI["help_text"]}}, "Your plan and starter portfolio."),
        html.hr(),
        html.h3({"style": {"color": UI["text_color"]}}, "Plan"),
        html.ul(
            {"style": {"color": UI["muted_text"]}},
            [
                html.li(f"Invest Amount: {money(inputs.get('invest_amount'))}"),
                html.li(f"Target Amount: {money(inputs.get('target_amount'))}"),
                html.li(f"Required CAGR: {pct(cagr)}"),
            ],
        ),
        html.h3({"style": {"color": UI["text_color"], "marginTop": "16px"}}, "Recommendations"),
        html.ul(
            {"style": {"color": UI["muted_text"]}},
            [html.li(f"{r.get('ticker','')} — {r.get('name','')} ({pct(r.get('weight',0))})") for r in recs],
        ),
    ]

    card = html.div({ "style": {**STYLE_CARD_WIDE, "textAlign": "left"} }, items)
    return html.div({"style": {**PAGE_WRAPPER, "fontFamily": UI["font_family"]}}, [card])
