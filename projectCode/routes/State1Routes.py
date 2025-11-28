# state1routes.py
from flask import Blueprint, request, jsonify
from constants import API_BASE_STATE1
import json, os
state1_bp = Blueprint("state1", __name__, url_prefix=API_BASE_STATE1)
# store user preferences
preferencesFile = "preferences.json"
def loadPrefs():
    # read file
    if os.path.exists(preferencesFile):
        try:
            with open(preferencesFile, "r") as file:
                return json.load(file)
        except Exception:
            return {}
    return {}
def savePrefs(data):
    # write file
    with open(preferencesFile, "w") as file:
        json.dump(data, file, indent=2)
def formatMoneySimple(value):
    # turn "$1,000" into 1000.0
    if value is None:
        return None
    text = str(value).replace(",", "").replace("$", "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None
@state1_bp.post("/recommend")
def generateRecs():
    # JSON from the goal screen
    data = request.get_json(silent=True) or {}
    # username
    username = (data.get("username") or "").strip()
    # inputs from user
    startAmtInput = data.get("startAmt")
    targetAmtInput = data.get("targetAmt")
    timeFrameInput = data.get("timeFrame")           # numeric (years)
    timeFrameDisplay = data.get("timeFrameDisplay")  # pretty label (ex: "6 Months")
    # formatted numbers
    startAmt = formatMoneySimple(startAmtInput)
    targetAmt = formatMoneySimple(targetAmtInput)
    try:
        timeFrame = float(str(timeFrameInput).strip())
    except Exception:
        timeFrame = None
    # simple validation
    if (
        startAmt is None or startAmt <= 0
        or targetAmt is None or targetAmt <= 0
        or timeFrame is None or timeFrame <= 0
    ):
        return jsonify({
            "ok": False,
            "message": "Please fill out all fields with positive numbers."
        }), 400
    # growth rate (can delete if we dont need)
    growthRate = (targetAmt / startAmt) ** (1 / timeFrame) - 1
    # placeholder recs (ML model will replace this)
    recommendations = [
        {"ticker": "ticker1", "name": "stock1", "abriv": "abriv1", "weight": 0.33},
        {"ticker": "ticker2", "name": "stock2", "abriv": "abriv2", "weight": 0.33},
        {"ticker": "ticker3", "name": "stock3", "abriv": "abriv3", "weight": 0.34},
    ]
    # save prefs for this user
    prefs = loadPrefs()
    key = username or "lastUser"
    prefs[key] = {
        "username": key,
        "startAmt": startAmt,
        "targetAmt": targetAmt,
        "timeFrame": timeFrame,
        "timeFrameDisplay": timeFrameDisplay,
        "growthRate": growthRate,
    }
    savePrefs(prefs)
    # send cleaned inputs to the UI
    return jsonify({
        "ok": True,
        "inputs": {
            "startAmt": startAmt,
            "targetAmt": targetAmt,
            "timeFrame": timeFrame,
            "timeFrameDisplay": timeFrameDisplay,
        },
        "growthRate": growthRate,
        "recommendations": recommendations,
    }), 200