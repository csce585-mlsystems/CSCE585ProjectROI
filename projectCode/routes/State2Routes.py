# projectCode/routes/State2Routes.py
from flask import Blueprint, request, jsonify

state2_bp = Blueprint("state2", __name__, url_prefix="/api/state2")

@state2_bp.get("/health")
def health():
    return jsonify({"ok": True})
