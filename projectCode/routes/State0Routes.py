# state0routes.py

from flask import Blueprint, request, jsonify

state0_bp = Blueprint("state0", __name__, url_prefix="/api/state0")

# fake database for testing.
# stored as: USERS = { "ava": {"password": "pswd", "email": "email@email.com", "phone": "123-1234"} }
USERS = {}

# SIGN UP
@state0_bp.post("/signup")
def signup():
    """Create a new user account"""
    # get json
    data = request.get_json(silent=True) or {}

    # inputted values
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    email = (data.get("email") or "").strip()
    phone = (data.get("phone") or "").strip()

    # make sure inputted info
    if not username or not password or not email or not phone:
        return jsonify({"message": "Username, password, email, and phone are required!"}), 400

    if username in USERS:
        return jsonify({"message": "Username already exists!"}), 409

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
    """Log in an existing user"""
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
        "message": "welcome",
        "username": username,
        "email": user.get("email"),
        "phone": user.get("phone"),
        "next_state": "state1" 
    }), 200
