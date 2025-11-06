from reactpy import component, html, hooks
from constants import (
    PAGE_WRAPPER,
    UI,
    SIDEBAR_STYLE,
    CONTENT_STYLE,
    CARD_STYLE,
    BTN_SIDEBAR,
    BTN_SIDEBAR_ACTIVE,
)

def btn_item(text, active, on_click):
    style = BTN_SIDEBAR_ACTIVE if active else BTN_SIDEBAR
    return html.button({"style": style, "on_click": on_click}, text)

def H(t, s=28):
    return html.h1({"style": {"margin": 0, "color": UI["text_color"], "fontSize": f"{s}px"}}, t)

def P(t, muted=True):
    return html.p({"style": {"margin": "6px 0 0", "color": UI["help_text"] if muted else UI["muted_text"]}}, t)

@component
def PageDashboard():
    header_style = dict(CARD_STYLE); header_style["color"] = UI["text_color"]
    return html.div(
        {"style": {"display": "flex", "flexDirection": "column", "gap": "18px"}},
        [
            html.div({"style": header_style}, [H("Dashboard", 24), P("blah blah blah dashboard junk.")]),
            html.div({"style": CARD_STYLE}, [H("Top Picks", 18), P("plah blah blah generated suggestions.")]),
        ],
    )

@component
def PagePortfolio():
    return html.div(
        {"style": {"display": "flex", "flexDirection": "column", "gap": "18px"}},
        [html.div({"style": CARD_STYLE}, [H("My Portfolio", 24), P("Allocations and performance.")])],
    )

@component
def PageDiscover():
    return html.div(
        {"style": {"display": "flex", "flexDirection": "column", "gap": "18px"}},
        [html.div({"style": CARD_STYLE}, [H("Discover", 24), P("Screen for new ideas by factors.")])],
    )

@component
def PageSettings():
    return html.div(
        {"style": {"display": "flex", "flexDirection": "column", "gap": "18px"}},
        [html.div({"style": CARD_STYLE}, [H("Settings", 24), P("Preferences and connections.")])],
    )

@component
def DashboardView(data=None):
    d = data or {}
    tab, set_tab = hooks.use_state("dashboard")

    brand = html.div(
        {"style": {
            "display": "flex",
            "alignItems": "center",
            "gap": "10px",
            "color": UI["text_color"],
            "margin": "0 0 8px 6px"
        }},
        [
            html.div({"style": {"width": "28px", "height": "28px", "background": "#1d4ed8", "borderRadius": "8px"}}),
            html.strong({"style": {"fontSize": "1.2rem"}}, "USER"),
        ],
    )

    nav = html.nav(
        {"style": {"display": "grid", "gap": "10px", "padding": "6px 4px"}},
        [
            btn_item("Dashboard", tab == "dashboard", lambda _e: set_tab("dashboard")),
            btn_item("My Portfolio", tab == "portfolio", lambda _e: set_tab("portfolio")),
            btn_item("Discover", tab == "discover", lambda _e: set_tab("discover")),
            btn_item("Settings", tab == "settings", lambda _e: set_tab("settings")),
        ],
    )

    sidebar = html.div({"style": SIDEBAR_STYLE}, [brand, nav])

    if tab == "dashboard":
        main = PageDashboard()
    elif tab == "portfolio":
        main = PagePortfolio()
    elif tab == "discover":
        main = PageDiscover()
    else:
        main = PageSettings()

    content = html.div({"style": CONTENT_STYLE}, [main])

    wrapper_style = dict(PAGE_WRAPPER)
    wrapper_style["justifyContent"] = "flex-start"
    wrapper_style["alignItems"] = "flex-start"
    wrapper_style["background"] = "#3b3b3b"
    wrapper_style["fontFamily"] = UI["font_family"]

    return html.div({"style": wrapper_style}, [sidebar, content])
