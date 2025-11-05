# state1routes.py
from flask import Blueprint, request, jsonify
from constants import API_BASE_STATE1

state1_bp = Blueprint("state1", __name__, url_prefix="/api/state1")

# reccomendation generation
@state1_bp.post("/recommend")
def recommend():
    data = request.get_json(silent=True) or {}

    def parse_money(x):
        if x is None:
            return None
        s = str(x).replace(",", "").replace("$", "").strip()
        if not s:
            return None
        try:
            return float(s)
        except:
            return None

    invest = parse_money(data.get("invest_amount"))
    target = parse_money(data.get("target_amount"))
    try:
        years = float(str(data.get("years", "")).strip())
    except:
        years = None

    if not all([invest, target, years]) or invest <= 0 or target <= 0 or years <= 0:
        return jsonify({"ok": False, "message": "Please provide positive values for all fields."}), 400

    cagr = (target / invest) ** (1.0 / years) - 1.0

    recs = [
        {"ticker": "VTV", "name": "Vanguard Value ETF", "weight": 0.40},
        {"ticker": "BRK.B", "name": "Berkshire Hathaway", "weight": 0.30},
        {"ticker": "SCHD", "name": "Schwab U.S. Dividend Equity", "weight": 0.30},
    ]

    return jsonify({
        "ok": True,
        "inputs": {"invest_amount": invest, "target_amount": target, "years": years},
        "required_cagr": cagr,
        "recommendations": recs
    }), 200
