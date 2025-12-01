# State2Routes.py
from flask import Blueprint, request, jsonify
state2_bp = Blueprint("state2", __name__, url_prefix="/api/state2")
@state2_bp.get("/health")
def health():
    return jsonify({"ok": True}), 200
@state2_bp.post("/portfolio")
def portfolio():
    data = request.get_json(silent=True) or {}
    budget = data.get("budget", 10000)
    years = data.get("years", 5)
    risk = data.get("risk", "Medium")
    # replace w whatever ML model returns.
    # keep the same keys so the UI wont break
    picks = [
        {
            "company": "Company 1",      # placeholder
            "ticker": "TICK1",           # placeholder
            "score": 90,                 # placeholder
            "tags": ["Tag A", "Tag B"],  # placeholder
            "risk": "Low",               # placeholder
        },
        {
            "company": "Company 2",
            "ticker": "TICK2",
            "score": 85,
            "tags": ["Tag A", "Tag C"],
            "risk": "Medium",
        },
        {
            "company": "Company 3",
            "ticker": "TICK3",
            "score": 78,
            "tags": ["Tag D"],
            "risk": "High",
        },
    ]
    return jsonify(
        {
            "ok": True,
            "inputs": {
                "budget": budget,
                "years": years,
                "risk": risk,
            },
            "picks": picks,
        }
    ), 200