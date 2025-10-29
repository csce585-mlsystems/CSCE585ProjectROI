# componentSet0.py
# login + signup screens
from reactpy import component, html, hooks, event
import asyncio, httpx
from constants import (
    LOGIN_ROUTE,
    SIGNUP_ROUTE,
    STYLE_CARD,
    STYLE_INPUT,
    STYLE_BUTTON,
    STYLE_BUTTON_SECONDARY,
    PAGE_WRAPPER,
    MESSAGE_STYLE,
    URL,
)

# POST to our api
async def api_post(route, payload):
    from constants import URL
    full_url = URL.rstrip("/") + route
    print("[api_post] POST ->", full_url, "payload:", payload)
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(full_url, json=payload, timeout=15)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, (r.text or "")
    except Exception as ex:
        return 0, f"{type(ex).__name__}: {ex}"


# login 
@component
def State0View(on_success=None, on_create_account=None):
    # local state for inputs + ui
    username, set_username = hooks.use_state("")
    password, set_password = hooks.use_state("")
    error, set_error       = hooks.use_state(None)
    loading, set_loading   = hooks.use_state(False)

    # input handlers
    def on_u(e): set_username((e["target"]["value"] or "").strip())
    def on_p(e): set_password((e["target"]["value"] or "").strip())

    # let the user press Enter to submit
    @event()
    def on_password_keydown(e):
        if e.get("key") == "Enter":
            asyncio.create_task(submit())

    async def submit():
        if not username or not password:
            set_error("Please enter username and password.")
            return
        set_loading(True)
        set_error(None)
        status, data = await api_post(LOGIN_ROUTE, {
            "username": username,
            "password": password,
        })
        set_loading(False)
        if status == 200 and isinstance(data, dict):
            if on_success:
                on_success(data.get("next_state", "state1"), data)
        else:
            msg = data.get("message", "") if isinstance(data, dict) else str(data)
            set_error(msg or "Login failed.")

    can_submit = bool(username and password and not loading)
    btn_primary = (STYLE_BUTTON | {
        "cursor": "pointer" if can_submit else "not-allowed",
        "background": "#000" if can_submit else "#777",
        "color": "#fff",
    })

    return html.div(
        {"style": STYLE_CARD},
        [
            html.h2({"style": {"marginBottom": "20px"}}, "Login"),
            html.input({"placeholder": "Username", "value": username, "onInput": on_u,
                        "autoComplete": "username", "style": STYLE_INPUT}),
            html.input({"placeholder": "Password", "type": "password", "value": password,
                        "onInput": on_p, "onChange": on_p, "onKeyDown": on_password_keydown,
                        "autoComplete": "current-password", "style": STYLE_INPUT}),
            html.button({"onClick": lambda _e: asyncio.create_task(submit()),
                         "disabled": not can_submit, "style": btn_primary},
                         "Sign In" if not loading else "Signing In..."),
            html.p(
                {
                    "onClick": lambda _e: on_create_account() if on_create_account else None,
                    "style": {
                        "alignSelf": "flex-start",
                        "marginLeft": "5%",
                        "marginTop": "12px",
                        "fontSize": "0.9rem",
                        "color": "#fff",
                        "cursor": "pointer",
                        "textDecoration": "underline",
                    },
                },
                "Sign Up?",
            ),
            html.p({"style": {"color": "crimson", "marginTop": "15px"}}, error)
            if error else html.span(""),
        ],
    )

# sign up 
@component
def CreateAccountView(on_success=None, on_go_back=None):
    username, set_username = hooks.use_state("")
    email, set_email       = hooks.use_state("")
    phone, set_phone       = hooks.use_state("")
    password, set_password = hooks.use_state("")
    confirm, set_confirm   = hooks.use_state("")
    error, set_error       = hooks.use_state(None)
    loading, set_loading   = hooks.use_state(False)

    def on_u(e):  set_username((e["target"]["value"] or "").strip())
    def on_e(e):  set_email((e["target"]["value"] or "").strip())
    def on_ph(e): set_phone((e["target"]["value"] or "").strip())
    def on_p(e):  set_password((e["target"]["value"] or "").strip())
    def on_c(e):  set_confirm((e["target"]["value"] or "").strip())

    async def submit():
        if not username or not email or not phone or not password:
            set_error("Please fill out all fields.")
            return
        if password != confirm:
            set_error("Passwords do not match.")
            return
        set_loading(True)
        set_error(None)
        status, data = await api_post(SIGNUP_ROUTE, {
            "username": username, "password": password,
            "email": email, "phone": phone,
        })
        set_loading(False)
        if status == 201:
            if on_success:
                on_success("account_created", {"username": username})
        else:
            msg = data.get("message", "") if isinstance(data, dict) else str(data)
            set_error(msg or "Sign up failed.")

    can_submit = bool(username and email and phone and password and confirm and not loading)
    btn_primary = (STYLE_BUTTON | {
        "cursor": "pointer" if can_submit else "not-allowed",
        "background": "#000" if can_submit else "#777",
        "color": "#fff",
    })

    return html.div(
        {"style": STYLE_CARD},
        [
            html.h2({"style": {"marginBottom": "20px"}}, "Create Account"),
            html.input({"placeholder": "Username", "value": username, "onInput": on_u,
                        "autoComplete": "username", "style": STYLE_INPUT}),
            html.input({"placeholder": "Email", "type": "email", "value": email, "onInput": on_e,
                        "autoComplete": "email", "style": STYLE_INPUT}),
            html.input({"placeholder": "Phone", "type": "tel", "value": phone, "onInput": on_ph,
                        "autoComplete": "tel", "style": STYLE_INPUT}),
            html.input({"placeholder": "Password", "type": "password", "value": password, "onInput": on_p,
                        "onChange": on_p, "autoComplete": "new-password", "style": STYLE_INPUT}),
            html.input({"placeholder": "Confirm Password", "type": "password", "value": confirm, "onInput": on_c,
                        "onChange": on_c, "autoComplete": "new-password", "style": STYLE_INPUT}),
            html.button({"onClick": lambda _e: asyncio.create_task(submit()),
                         "disabled": not can_submit, "style": btn_primary},
                         "Create Account" if not loading else "Creating..."),
            html.p(
                {
                    "onClick": lambda _e: on_go_back() if on_go_back else None,
                    "style": {
                        "alignSelf": "flex-start",
                        "marginLeft": "5%",
                        "marginTop": "12px",
                        "fontSize": "0.9rem",
                        "color": "#fff",
                        "cursor": "pointer",
                        "textDecoration": "underline",
                    },
                },
                "Back to Login?",
            ),
            html.p({"style": {"color": "crimson", "marginTop": "15px"}}, error)
            if error else html.span(""),
        ],
    )

# login <-> signup
@component
def RootView():
    screen, set_screen = hooks.use_state("login")
    message, set_message = hooks.use_state(None)

    def go_register():
        set_message(None)
        set_screen("register")

    def go_login(*_args):
        set_screen("login")
        set_message("Account created!")

    return html.div(
        {"style": PAGE_WRAPPER},
        [
            html.p({"style": MESSAGE_STYLE}, message) if message else html.span(""),
            html.div(
                {"style": {"width": "100%", "maxWidth": "480px"}},
                [
                    (
                        State0View(on_success=lambda *_: None, on_create_account=go_register)
                        if screen == "login"
                        else CreateAccountView(
                            on_success=lambda *_: go_login(),
                            on_go_back=go_login
                        )
                    )
                ],
            ),
        ],
    )
