# componentSet1.py
from reactpy import component, html, hooks
import asyncio, httpx
from constants import (
    URL,
    PAGE_WRAPPER,
    UI,
    RECOMMEND_ROUTE,
    STYLE_CARD_WIDE,
    STYLE_INPUT_FULL,
    STYLE_BTN_PRIMARY,
    LABEL_STYLE,
    HELP_STYLE,
)

async def api_post(route, payload):
    full_url = URL.rstrip("/") + route
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(full_url, json=payload, timeout=20)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, (r.text or "")
    except Exception as ex:
        return 0, f"{type(ex).__name__}: {ex}"

def FieldMoney(label, value, setter, placeholder, help_text):
    return html.div(
        {"style": {"marginBottom": "18px"}},
        [
            html.label({"style": LABEL_STYLE}, label),
            html.input({
                "type": "text",
                "style": STYLE_INPUT_FULL,
                "value": value,
                "placeholder": "$" + placeholder if not value else value,
                "on_change": lambda e: setter(format_money(e["target"]["value"])),
            }),
            html.div({"style": HELP_STYLE}, help_text),
        ],
    )

def format_money(val):
    s = str(val).strip().replace("$", "")
    if not s:
        return ""
    return "$" + s

def FieldDuration(duration, set_duration, duration_unit, set_duration_unit):
    return html.div(
        {"style": {"marginBottom": "18px"}},
        [
            html.div(
                {"style": {"display": "flex", "justifyContent": "space-between", "alignItems": "center"}},
                [
                    html.label({"style": LABEL_STYLE}, "How long do you plan to invest?"),
                    html.select(
                        {
                            "style": {
                                "background": "#3a3a3a",
                                "color": "#fff",
                                "border": "none",
                                "borderRadius": "8px",
                                "padding": "6px 8px",
                                "fontSize": "0.9rem",
                            },
                            "value": duration_unit,
                            "on_change": lambda e: set_duration_unit(e["target"]["value"]),
                        },
                        [
                            html.option({"value": "Years"}, "Years"),
                            html.option({"value": "Months"}, "Months"),
                            html.option({"value": "Days"}, "Days"),
                        ],
                    ),
                ],
            ),
            html.input({
                "type": "number",
                "style": STYLE_INPUT_FULL,
                "value": duration,
                "placeholder": "5",
                "on_change": lambda e: set_duration(e["target"]["value"]),
            }),
            html.div({"style": HELP_STYLE}, "Longer time frames often allow for more growth potential."),
        ],
    )

@component
def State1View(on_success=None):
    invest, set_invest = hooks.use_state("")
    target, set_target = hooks.use_state("")
    duration, set_duration = hooks.use_state("")
    duration_unit, set_duration_unit = hooks.use_state("Years")

    loading, set_loading = hooks.use_state(False)
    error, set_error = hooks.use_state(None)

    async def submit():
        set_loading(True); set_error(None)
        try:
            payload = {
                "invest_amount": invest,
                "target_amount": target,
                "years": convert_to_years(duration, duration_unit),
            }
            status, data = await api_post(RECOMMEND_ROUTE, payload)
            if status == 200 and isinstance(data, dict) and data.get("ok"):
                if on_success:
                    on_success("state2", data)
            else:
                msg = data.get("message", "") if isinstance(data, dict) else str(data)
                set_error(msg or "Request failed.")
        except Exception as e:
            set_error(str(e))
        finally:
            set_loading(False)

    def convert_to_years(value, unit):
        try:
            v = float(value)
        except:
            return 0
        if unit == "Months":
            return v / 12
        elif unit == "Days":
            return v / 365
        return v

    header = html.div(
        {"style": {"textAlign": "center", "marginBottom": "18px"}},
        [
            html.h1({"style": {"color": UI["text_color"], "fontSize": "28px", "fontWeight": 800, "margin": 0}}, "Start Investing!"),
            html.h2({"style": {"color": UI["text_color"], "fontSize": "22px", "fontWeight": 700, "marginTop": "10px"}}, "Let's Plan Your Investment"),
            html.p({"style": {"color": UI["help_text"], "marginTop": "4px"}}, "Tell us your goals, and we'll find the right investments for you."),
        ],
    )

    btn_style = {**STYLE_BTN_PRIMARY, "cursor": "pointer"}
    button = html.button(
        {
            "type": "button",
            "role": "button",
            "tabIndex": 0,
            "style": btn_style,
            "on_click": lambda _e: asyncio.create_task(submit()),
            "on_keydown": lambda e: asyncio.create_task(submit()) if e.get("key") in ("Enter", " ") else None,
        },
        "Generate Recommendations",
    )

    elements = [
        header,
        FieldMoney("How much would you like to invest?", invest, set_invest, "5,000", "This is the total amount you're ready to put into your portfolio."),
        FieldMoney("What is your target return?", target, set_target, "20,000", "Tell us the final amount you'd like to have after your investment period."),
        FieldDuration(duration, set_duration, duration_unit, set_duration_unit),
        button,
    ]
    if loading:
        elements.append(html.p({"style": {"color": UI["muted_text"], "marginTop": "12px"}}, "Working on it..."))
    if error:
        elements.append(html.p({"style": {"color": "crimson", "marginTop": "12px"}}, error))

    form = html.div({"style": STYLE_CARD_WIDE}, elements)

    return html.div({"style": {**PAGE_WRAPPER, "fontFamily": UI["font_family"]}}, [form])
