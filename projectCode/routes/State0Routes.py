# state0routes.py
# backend routes for state 0 (login + signup)
from flask import Blueprint, request, jsonify
from constants import API_BASE_STATE0

state0_bp = Blueprint("state0", __name__, url_prefix=API_BASE_STATE0)

# fake db -> resets on server restart (good enough for testing)
# USERS = { "ava": {"password": "pass", "email": "email@email.com", "phone": "123-4567"} }
USERS = {}

# SIGN UP
@state0_bp.post("/signup")
def signup():
    """create new acc"""
    # get json
    data = request.get_json(silent=True) or {}

    # inputted values
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    email    = (data.get("email") or "").strip()
    phone    = (data.get("phone") or "").strip()

    # make sure inputted info
    if not username or not password or not email or not phone:
        return jsonify({"message": "Username, password, email, and phone number are required!"}), 400

    if username in USERS:
        return jsonify({"message": "Username already exists! Please pick a new one."}), 409

    # Save the user in our fake database
    USERS[username] = {"password": password, "email": email, "phone": phone}

    # Send a success message back
    return jsonify({
        "message": "Account created!",
        "username": username,
        "email": email,
        "phone": phone
    }), 201


# LOGIN
@state0_bp.post("/login")
def login():
    """log in existing user"""
    # get json
    data = request.get_json(silent=True) or {}

    # inputted values
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    #existing user?
    user = USERS.get(username)

    # error messages
    if not user:
        return jsonify({"message": "User not found. Please sign up first."}), 404
    if user["password"] != password:
        return jsonify({"message": "Incorrect password."}), 401

    # success messages
    return jsonify({
        "message": "Welcome!",
        "username": username,
        "email": user.get("email"),
        "phone": user.get("phone"),
        "next_state": "state1" 
    }), 200
