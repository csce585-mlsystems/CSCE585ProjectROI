# components/state0.py
from reactpy import component, html, hooks, event
import asyncio, httpx

# If your Flask + ReactPy are served from the same origin (same host/port),
# this relative base path is perfect. If you run the UI separately (different port),
# change this to the full URL, e.g. "http://127.0.0.1:5001/api/state0"
API_BASE = "/api/state0"


async def api_post(path: str, payload: dict) -> tuple[int, dict | str]:
    """POST JSON to the Flask API and return (status_code, data_or_text)."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{API_BASE}{path}", json=payload, timeout=15)
        # Try JSON first; fall back to text for error messages
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text
    except Exception as ex:
        # Network error, timeout, etc.
        return 0, str(ex)


@component
def State0View(on_success=None, on_create_account=None):
    """Login screen"""
    username, set_username = hooks.use_state("")
    password, set_password = hooks.use_state("")
    error, set_error = hooks.use_state(None)
    loading, set_loading = hooks.use_state(False)

    # Handlers
    def on_u(e): set_username((e["target"]["value"] or "").strip())
    def on_p(e): set_password((e["target"]["value"] or "").strip())

    # Pressing Enter in password field submits
    @event(prevent_default=True)
    def on_password_keydown(e):
        if e["key"] == "Enter":
            asyncio.create_task(submit())

    async def submit():
        # guard rails
        if not username or not password:
            set_error("Please enter username and password.")
            return
        set_loading(True); set_error(None)

        status, data = await api_post("/login", {
            "username": username, "password": password
        })

        set_loading(False)

        if status == 200 and isinstance(data, dict):
            if on_success:
                on_success(data.get("next_state", "state1"), data)
        else:
            # prefer API message if available
            msg = ""
            if isinstance(data, dict):
                msg = data.get("message", "") or str(data)
            else:
                msg = data
            set_error(msg or "Login failed.")

    # disable
    can_submit = bool(username and password and not loading)

    return html.div(
        {
            "style": {
                "maxWidth": "420px",
                "margin": "80px auto",
                "fontFamily": "'Georgia', 'Times New Roman', serif",
                "color": "#111",
                "background": "#f9f9f9",
                "padding": "40px",
                "borderRadius": "20px",
                "boxShadow": "0 6px 12px rgba(0,0,0,0.1)",
                "textAlign": "center",
            }
        },
        [
            html.h2(
                {"style": {"marginBottom": "25px", "fontSize": "1.6rem", "fontWeight": 600, "letterSpacing": "0.5px"}},
                "Login Page"
            ),

            html.input({
                "placeholder": "Username",
                "value": username,
                "onInput": on_u,
                "autoComplete": "username",
                "style": {"width": "100%","padding": "12px","margin": "10px 0","borderRadius": "10px","border": "1px solid #ccc","outline": "none","fontSize": "0.95rem","background": "#fff"},
            }),

            html.input({
                "placeholder": "Password",
                "type": "password",
                "value": password,
                "onInput": on_p,
                "onKeyDown": on_password_keydown,
                "autoComplete": "current-password",
                "style": {"width": "100%","padding": "12px","margin": "10px 0","borderRadius": "10px","border": "1px solid #ccc","outline": "none","fontSize": "0.95rem","background": "#fff"},
            }),

            html.button(
                {
                    "onClick": lambda _e: asyncio.create_task(submit()),
                    "disabled": not can_submit,
                    "style": {
                        "marginTop": "20px","width": "100%","padding": "12px","borderRadius": "30px","border": "none",
                        "cursor": "not-allowed" if not can_submit else "pointer",
                        "background": "#000" if can_submit else "#777","color": "#fff",
                        "fontWeight": 600,"fontSize": "1rem","letterSpacing": "0.5px","transition": "background 0.2s ease"
                    },
                },
                "Sign In" if not loading else "Signing In...",
            ),

            html.button(
                {
                    "onClick": lambda _e: on_create_account() if on_create_account else None,
                    "style": {"marginTop": "15px","width": "100%","padding": "12px","borderRadius": "30px","border": "1px solid #000","cursor": "pointer","background": "#fff","color": "#000","fontWeight": 500,"fontSize": "0.95rem","letterSpacing": "0.5px","transition": "all 0.2s ease"},
                },
                "Sign Up",
            ),

            html.p({"style": {"color": "crimson", "marginTop": "15px"}}, error) if error else html.span(""),
        ],
    )


@component
def CreateAccountView(on_success=None, on_go_back=None):
    """Registration screen"""
    username, set_username = hooks.use_state("")
    email, set_email = hooks.use_state("")
    phone, set_phone = hooks.use_state("")
    password, set_password = hooks.use_state("")
    confirm, set_confirm = hooks.use_state("")
    error, set_error = hooks.use_state(None)
    loading, set_loading = hooks.use_state(False)

    # Handlers
    def on_u(e): set_username((e["target"]["value"] or "").strip())
    def on_e(e): set_email((e["target"]["value"] or "").strip())
    def on_ph(e): set_phone((e["target"]["value"] or "").strip())
    def on_p(e): set_password((e["target"]["value"] or "").strip())
    def on_c(e): set_confirm((e["target"]["value"] or "").strip())

    async def submit():
        if not username or not email or not phone or not password:
            set_error("Please enter a username, email, phone, and password.")
            return
        if password != confirm:
            set_error("Passwords do not match.")
            return

        set_loading(True); set_error(None)

        # sign up info, usernmae, pswd, email phone number
        status, data = await api_post("/signup", {
            "username": username,
            "password": password,
            "email": email,
            "phone": phone,
        })

        set_loading(False)

        if status == 201:
            if on_success:
                on_success("account_created", {"username": username})
        else:
            msg = ""
            if isinstance(data, dict):
                msg = data.get("message", "") or str(data)
            else:
                msg = data
            set_error(msg or "Sign up failed.")

    can_submit = bool(username and email and phone and password and confirm and not loading)

    return html.div(
        {"style": {
            "maxWidth": "420px","margin": "80px auto","fontFamily": "'Georgia', 'Times New Roman', serif",
            "color": "#111","background": "#f9f9f9","padding": "40px","borderRadius": "20px",
            "boxShadow": "0 6px 12px rgba(0,0,0,0.1)","textAlign": "center"
        }},
        [
            html.h2({"style": {"marginBottom": "25px","fontSize": "1.6rem","fontWeight": 600,"letterSpacing": "0.5px"}}, "Sign Up"),

            html.input({
                "placeholder": "Username",
                "value": username,
                "onInput": on_u,
                "autoComplete": "username",
                "style": {"width": "100%","padding": "12px","margin": "10px 0","borderRadius": "10px","border": "1px solid #ccc","outline": "none","fontSize": "0.95rem","background": "#fff"},
            }),

            html.input({
                "placeholder": "Email",
                "type": "email",
                "value": email,
                "onInput": on_e,
                "autoComplete": "email",
                "style": {"width": "100%","padding": "12px","margin": "10px 0","borderRadius": "10px","border": "1px solid #ccc","outline": "none","fontSize": "0.95rem","background": "#fff"},
            }),

            html.input({
                "placeholder": "Phone",
                "type": "tel",
                "value": phone,
                "onInput": on_ph,
                "autoComplete": "tel",
                "style": {"width": "100%","padding": "12px","margin": "10px 0","borderRadius": "10px","border": "1px solid #ccc","outline": "none","fontSize": "0.95rem","background": "#fff"},
            }),

            html.input({
                "placeholder": "Password",
                "type": "password",
                "value": password,
                "onInput": on_p,
                "autoComplete": "new-password",
                "style": {"width": "100%","padding": "12px","margin": "10px 0","borderRadius": "10px","border": "1px solid #ccc","outline": "none","fontSize": "0.95rem","background": "#fff"},
            }),

            html.input({
                "placeholder": "Confirm Password",
                "type": "password",
                "value": confirm,
                "onInput": on_c,
                "autoComplete": "new-password",
                "style": {"width": "100%","padding": "12px","margin": "10px 0","borderRadius": "10px","border": "1px solid #ccc","outline": "none","fontSize": "0.95rem","background": "#fff"},
            }),

            html.button(
                {
                    "onClick": lambda _e: asyncio.create_task(submit()),
                    "disabled": not can_submit,
                    "style": {
                        "marginTop": "20px","width": "100%","padding": "12px","borderRadius": "30px","border": "none",
                        "cursor": "not-allowed" if not can_submit else "pointer",
                        "background": "#000" if can_submit else "#777","color": "#fff",
                        "fontWeight": 600,"fontSize": "1rem","letterSpacing": "0.5px","transition": "background 0.2s ease"
                    },
                },
                "Create Account" if not loading else "Creating...",
            ),

            html.button(
                {
                    "onClick": lambda _e: on_go_back() if on_go_back else None,
                    "style": {"marginTop": "15px","width": "100%","padding": "12px","borderRadius": "30px","border": "1px solid #000","cursor": "pointer","background": "#fff","color": "#000","fontWeight": 500,"fontSize": "0.95rem","letterSpacing": "0.5px","transition": "all 0.2s ease"},
                },
                "Back to Login",
            ),

            html.p({"style": {"color": "crimson", "marginTop": "15px"}}, error) if error else html.span(""),
        ],
    )


@component
def RootView():
    """Simple screen switcher between Login and Create Account"""
    screen, set_screen = hooks.use_state("login")
    message, set_message = hooks.use_state(None)

    def go_register():
        set_message(None)
        set_screen("register")

    def go_login(*_args):
        set_screen("login")
        set_message("account created.")

    return html.div(
        {},
        [
            html.p({"style": {"textAlign": "center", "color": "black"}} , message) if message else html.span(""),
            (
                State0View(
                    on_success=lambda next_state, data: print("Logged in:", data),
                    on_create_account=go_register
                )
                if screen == "login"
                else CreateAccountView(
                    on_success=lambda *_: go_login(),
                    on_go_back=go_login
                )
            )
        ]
    )
