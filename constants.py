# constants.py
import os

HOST = "127.0.0.1"
PORT = 5000
DEBUG = True
URL = "http://127.0.0.1:5000"

# ui default
UI = {
    "font_family": "system-ui, sans-serif",
    "background_color": "#3b3b3b",
    "text_color": "#fff",
    "muted_text": "#cbd5e1",
    "help_text": "#b3b3b3",
}

# whole page background 
PAGE_WRAPPER = {
    "background": "#3b3b3b",
    "position": "fixed",
    "inset": "0",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
    "alignItems": "center",
}

# small message (e.x, “Account created!”)
MESSAGE_STYLE = {
    "color": UI["text_color"],
    "marginBottom": "40px",
    "fontSize": "1rem",
    "fontWeight": 500,
}

# blueprint bases (add more states later)
API_BASE_STATE0 = "/api/state0"
API_BASE_STATE1 = "/api/state1"

# endpoints for state 0 
LOGIN_ROUTE  = "/api/state0/login"
SIGNUP_ROUTE = "/api/state0/signup"

# endpoints for state 1
RECOMMEND_ROUTE = "/api/state1/recommend"

# shared styles (dark mode) 
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

# state 0 
STATE0_BUTTONS = {
    "sign_in": {
        "text": "Sign In",
        "action": "POST username + password",
        "route": LOGIN_ROUTE,
        "success": "go to next_state",
        "fails_if": "missing fields / wrong creds",
    },
    "sign_up": {
        "text": "Sign Up",
        "action": "login -> signup (no API)",
        "route": None,
        "success": "shows CreateAccountView",
    },
    "create_account": {
        "text": "Create Account",
        "action": "POST username, email, phone, password",
        "route": SIGNUP_ROUTE,
        "success": "on 201, go back to login",
        "fails_if": "missing fields / wrong pass / username in use",
    },
    "back_to_login": {
        "text": "Back to Login",
        "action": "signup -> login (no API)",
        "route": None,
        "success": "shows login again",
    },
}

# state 1
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

# state 2
