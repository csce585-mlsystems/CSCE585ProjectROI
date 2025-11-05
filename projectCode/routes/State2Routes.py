# State2Routes.py
from flask import Blueprint, request, jsonify

state2_bp = Blueprint("state2", __name__, url_prefix="/api/state2")

@state2_bp.post("/portfolio")
def portfolio():
    data = request.get_json(silent=True) or {}
    return jsonify({"ok": True, "received": data}), 200
