# state0routes.py
from flask import Blueprint, request, jsonify
from constants import API_BASE_STATE0
import json, os

state0_bp = Blueprint("state0", __name__, url_prefix=API_BASE_STATE0)
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f, indent=2)

# SIGN UP
@state0_bp.post("/signup")
def signup():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    email    = (data.get("email") or "").strip()
    phone    = (data.get("phone") or "").strip()

    if not username or not password or not email or not phone:
        return jsonify({"message": "All fields are required!"}), 400

    users = load_users()
    if username in users:
        return jsonify({"message": "Username already exists!"}), 409

    users[username] = {"password": password, "email": email, "phone": phone}
    save_users(users)

    return jsonify({"message": "Account created!", "username": username}), 201

# LOGIN
@state0_bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    users = load_users()
    user = users.get(username)
    if not user:
        return jsonify({"message": "User not found."}), 404
    if user["password"] != password:
        return jsonify({"message": "Incorrect password."}), 401

    return jsonify({
        "message": "Welcome!",
        "username": username,
        "email": user.get("email"),
        "phone": user.get("phone"),
        "next_state": "state1",
    }), 200
