# state0routes.py
from flask import Blueprint, request, jsonify
from constants import API_BASE_STATE0
import json, os

# using a blueprint just for the auth / state0 routes
state0_bp = Blueprint("state0", __name__, url_prefix=API_BASE_STATE0)

# simple json "db" for users so we don't need a real database yet
USER_FILE = "users.json"

def load_users():
    # read all users from the json file (or return empty dict if it doesn't exist)
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    # overwrite the json file with the latest users dict
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

# SIGN UP
@state0_bp.post("/signup")
def signup():
    # grab the json body and clean up the fields
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    email    = (data.get("email") or "").strip()
    phone    = (data.get("phone") or "").strip()

    # quick validation so nothing important is missing
    if not username or not password or not email or not phone:
        return jsonify({"message": "All fields are required!"}), 400

    users = load_users()

    # make sure username is unique
    if username in users:
        return jsonify({"message": "Username already exists!"}), 409

    # save the new user into the json "db"
    users[username] = {"password": password, "email": email, "phone": phone}
    save_users(users)

    # 201 = created, send username back so ui can show a message
    return jsonify({"message": "Account created!", "username": username}), 201

# state0routes.py
@state0_bp.post("/login")
def login():
    # pull username/password off the request
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    users = load_users()
    user = users.get(username)
    # basic auth checks: user exists + password matches
    if not user:
        return jsonify({"message": "User not found."}), 404
    if user["password"] != password:
        return jsonify({"message": "Incorrect password."}), 401
    # if login works, send the user data + which state to go to next
    return jsonify({
        "message": "Welcome!",
        "username": username,
        "email": user.get("email"),
        "phone": user.get("phone"),
        "next_state": "state1",
    }), 200
