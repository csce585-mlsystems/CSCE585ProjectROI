import os

HOST = "127.0.0.1"
PORT = 5000
DEBUG = True
URL = "http://127.0.0.1:5000"

UI = {
    "font_family": "system-ui, sans-serif",
    "background_color": "#3b3b3b",
    "text_color": "#ffffff",
    "muted_text": "#cbd5e1",
    "help_text": "#b3b3b3",
}

PAGE_WRAPPER = {
    "background": UI["background_color"],
    "position": "fixed",
    "inset": "0",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
    "alignItems": "center",
}

API_BASE_STATE0 = "/api/state0"
API_BASE_STATE1 = "/api/state1"
API_BASE_STATE2 = "/api/state2"

LOGIN_ROUTE = f"{API_BASE_STATE0}/login"
SIGNUP_ROUTE = f"{API_BASE_STATE0}/signup"
RECOMMEND_ROUTE = f"{API_BASE_STATE1}/recommend"
PORTFOLIO_ROUTE = f"{API_BASE_STATE2}/portfolio"

MESSAGE_STYLE = {
    "color": UI["text_color"],
    "marginBottom": "40px",
    "fontSize": "1rem",
    "fontWeight": 500,
}

STYLE_CARD = {
    "maxWidth": "420px",
    "width": "100%",
    "margin": "0 auto",
    "display": "flex",
    "flexDirection": "column",
    "alignItems": "center",
    "justifyContent": "center",
    "fontFamily": UI["font_family"],
    "color": UI["text_color"],
    "background": "#2e2e2e",
    "padding": "40px",
    "borderRadius": "16px",
    "boxShadow": "0 8px 16px rgba(0,0,0,0.3)",
    "textAlign": "center",
}

STYLE_INPUT = {
    "width": "90%",
    "padding": "12px",
    "margin": "10px 0",
    "borderRadius": "10px",
    "border": "none",
    "fontSize": "1rem",
    "background": "#3a3a3a",
    "color": "#fff",
    "outline": "none",
    "textAlign": "left",
}

STYLE_BUTTON = {
    "marginTop": "16px",
    "width": "90%",
    "padding": "12px",
    "borderRadius": "30px",
    "border": "none",
    "fontWeight": 600,
    "fontSize": "1rem",
    "background": "#000",
    "color": "#fff",
}

STYLE_BUTTON_SECONDARY = {
    "marginTop": "12px",
    "width": "90%",
    "padding": "12px",
    "borderRadius": "30px",
    "border": "none",
    "fontWeight": 500,
    "fontSize": "0.95rem",
    "background": "transparent",
    "color": "#fff",
}

STATE0_BUTTONS = {
    "sign_in": {
        "text": "Sign In",
        "action": "POST username + password",
        "route": LOGIN_ROUTE,
        "success": "go to next_state",
    },
    "sign_up": {
        "text": "Sign Up",
        "action": "login -> signup (no API)",
    },
    "create_account": {
        "text": "Create Account",
        "action": "POST username, email, phone, password",
        "route": SIGNUP_ROUTE,
        "success": "on 201, go back to login",
    },
    "back_to_login": {
        "text": "Back to Login",
        "action": "signup -> login (no API)",
    },
}

STYLE_CARD_WIDE = {
    **STYLE_CARD,
    "maxWidth": "720px",
    "textAlign": "left",
}

STYLE_INPUT_FULL = {
    **STYLE_INPUT,
    "width": "100%",
}

STYLE_BTN_PRIMARY = {
    **STYLE_BUTTON,
    "width": "100%",
    "borderRadius": "10px",
}

LABEL_STYLE = {
    "color": UI["muted_text"],
    "fontSize": "14px",
    "marginBottom": "8px",
}

HELP_STYLE = {
    "color": UI["help_text"],
    "fontSize": "12px",
    "marginTop": "6px",
}

STATE1_BUTTONS = {
    "recommend": {
        "text": "Generate Recommendations",
        "route": RECOMMEND_ROUTE,
        "method": "POST",
    }
}

SIDEBAR_STYLE = {
    "width": "16.666%",
    "minWidth": "240px",
    "maxWidth": "300px",
    "height": "100vh",
    "background": "#2e2e2e",
    "borderRight": "2px solid #111",
    "boxShadow": "inset -10px 0 20px rgba(0,0,0,0.35)",
    "padding": "24px 20px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "18px",
    "position": "fixed",
    "top": "0",
    "left": "0",
}

CONTENT_STYLE = {
    "position": "relative",
    "height": "100vh",
    "overflowY": "auto",
    "background": "#3b3b3b",
    "color": UI["text_color"],
    "padding": "32px 48px 48px",
    "marginLeft": "clamp(260px, 17%, 300px)",
}

CARD_STYLE = {
    "background": "#2e2e2e",
    "border": "1px solid #1e293b",
    "borderRadius": "12px",
    "padding": "18px",
    "color": UI["muted_text"],
    "width": "100%",
}

BTN_SIDEBAR = {
    "display": "block",
    "textAlign": "left",
    "padding": "12px 16px",
    "borderRadius": "12px",
    "border": "none",
    "cursor": "pointer",
    "background": "transparent",
    "color": "#d0d0d0",
    "fontSize": "1rem",
    "fontWeight": 500,
    "width": "100%",
    "boxSizing": "border-box",
    "transition": "background 0.25s ease, color 0.25s ease",
    "margin": "2px 0",
}

BTN_SIDEBAR_ACTIVE = {
    **BTN_SIDEBAR,
    "background": "#000",
    "color": "#fff",
    "fontWeight": 600,
}
