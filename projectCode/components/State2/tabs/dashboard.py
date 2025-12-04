# Body of imports required for loading and using model
import os
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np  # <-- May be optional not sure as of 10/13/25.
import pdb as pb
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Dropout, Input
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.keras.datasets import mnist
import tensorflow.keras

# NOTE: May or may not need all these imports, BUT they are here just in case!
# End of Body of imports required for loading and using model

# dashboard tab
from reactpy import component, html
from constants import (
    UI,
    dashboardCard,
    dashboardHeaderBox,
    dashboardHeaderTitle,
    dashboardHeaderSubtitle,
    dashboardSummaryTitleText,
    dashboardSummaryList,
    stockTable,
    stockTableHeaderCell,
    stockTableHeaderCellRight,
    stockTableRow,
    stockCompanyCell,
    stockCompanyBox,
    stockCompanyNameText,
    stockTickerText,
    stockScoreCell,
    stockTagsCell,
    stockTagsRow,
    stockRiskCell,
    stockDetailsCell,
    stockDetailsButton,
    scoreBarRow,
    scoreBarTrack,
    scoreBarFillBase,
    scoreBarText,
    tagChip,
    riskBadgeBox,
    riskBadgeDot,
    chartsHeaderRow,
    chartsTitleText,
    chartsSubtitleText,
    chartsPlaceholderBox,
    picksHeaderRow,
    picksTitleText,
    dashboardStack,
)

# path to Jay's model prediction file (written by baseline_model.py)
# projectCode/MLLifecycle/ModelDevelopment/ModelPredictions.csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PRED_PATH = os.path.normpath(
    os.path.join(
        BASE_DIR,
        "..",  # State2
        "..",  # components
        "..",  # projectCode
        "MLLifecycle",
        "ModelDevelopment",
        "ModelPredictions.csv",
    )
)

# placeholder picks (ML model will replace if CSV is available)
placeholderPicks = [
    {
        "company": "Stock 1",
        "ticker": "TICK1",
        "score": 80,
        "tags": ["Tag A", "Tag B"],
        "risk": "Low",
    },
    {
        "company": "Stock 2",
        "ticker": "TICK2",
        "score": 75,
        "tags": ["Tag C", "Tag D"],
        "risk": "Medium",
    },
    {
        "company": "Stock 3",
        "ticker": "TICK3",
        "score": 70,
        "tags": ["Tag E", "Tag F"],
        "risk": "High",
    },
]


def load_model_picks(max_picks=3):
    """
    Goal:
      - read Jay's ModelPredictions.csv
      - turn the rows into the same dict format the UI already expects
      - if anything goes wrong, just use placeholderPicks

    This keeps things beginner-friendly and avoids breaking the dashboard
    if the model hasn't been run yet.
    """
    # simple mapping from ticker -> human-friendly company name
    tickerRealNameTable = {
        "AAPL": "Apple",
        "GOOG": "Google",
        "GOOGL": "Google",
        "AMZN": "Amazon",
        "MSFT": "Microsoft",
        # add more later if we expand the universe
    }

    # if the CSV doesn't exist yet, just use the hard-coded picks
    if not os.path.exists(MODEL_PRED_PATH):
        print(f"[Dashboard] ModelPredictions.csv not found at {MODEL_PRED_PATH}, using placeholder picks.")
        return placeholderPicks

    try:
        df = pd.read_csv(MODEL_PRED_PATH)
    except Exception as ex:
        print(f"[Dashboard] Error reading {MODEL_PRED_PATH}: {ex}")
        return placeholderPicks

    # we at least expect a 'Company' column from Jay's code
    if "Company" not in df.columns:
        print("[Dashboard] 'Company' column missing in ModelPredictions.csv, using placeholder picks.")
        return placeholderPicks

    # if there's an Optimality column, use that as our score;
    # otherwise, try Model Predictions; if not, fall back to 0.
    if "Optimality" in df.columns:
        score_col = "Optimality"
    elif "Model Predictions" in df.columns:
        score_col = "Model Predictions"
    else:
        score_col = None

    picks = []

    # just grab the top N rows for the demo
    for _, row in df.head(max_picks).iterrows():
        ticker = str(row["Company"])
        company_name = tickerRealNameTable.get(ticker, ticker)

        if score_col is not None:
            try:
                raw_score = float(row[score_col])
            except Exception:
                raw_score = 0.0
        else:
            raw_score = 0.0

        # clamp/scale score to something that looks like 0–100
        score = int((raw_score / df[score_col].max()) * 100)

        picks.append(
            {
                "company": company_name,
                "ticker": ticker,
                "score": score,
                "tags": ["Value", "Model Pick"],
                "risk": "Medium",
            }
        )

    # if CSV was empty or weird, fall back
    if not picks:
        return placeholderPicks

    print(f"[Dashboard] Loaded {len(picks)} picks from model.")
    return picks


# score bar for the Smart Score column
def scoreBar(score):
    percent = max(0, min(score, 100))
    return html.div(
        {"style": scoreBarRow},
        html.div(
            {"style": scoreBarTrack},
            html.div(
                {
                    "style": {
                        **scoreBarFillBase,
                        "width": f"{percent}%",
                        "background": "#3b82f6",
                    }
                }
            ),
        ),
        html.span({"style": scoreBarText}, str(score)),
    )


# tag pill under value metrics
def tagPill(text):
    return html.span({"style": tagChip}, text)


# risk badge
def riskBadge(level):
    colors = {"low": "#22c55e", "medium": "#3b82f6", "high": "#ef4444"}
    color = colors.get(level.lower(), "#22c55e")
    return html.span(
        {"style": riskBadgeBox},
        html.span({"style": {**riskBadgeDot, "background": color}}),
        html.span(level),
    )


@component
def DashboardTab(data=None):
    base = data or {}

    # left sidebar slider data
    budget = base.get("budget", 10000)
    years = base.get("years", 5)
    risk = str(base.get("risk", "Medium"))

    # inputs from pref
    inputs = base.get("inputs") or {}
    startAmt = inputs.get("startAmt")
    targetAmt = inputs.get("targetAmt")
    timeFrame = inputs.get("timeFrame")
    timeFrameDisplay = inputs.get("timeFrameDisplay")

    # also look at top level in case backend sends saved prefs directly
    if startAmt is None and "startAmt" in base:
        startAmt = base.get("startAmt")
    if targetAmt is None and "targetAmt" in base:
        targetAmt = base.get("targetAmt")
    if timeFrameDisplay is None and "timeFrameDisplay" in base:
        timeFrameDisplay = base.get("timeFrameDisplay")
    if timeFrame is None and "timeFrame" in base:
        timeFrame = base.get("timeFrame")

    growthRate = base.get("growthRate")

    # picks:
    # 1) if backend passes picks in data, use those
    # 2) otherwise, load from Jay's ModelPredictions.csv
    # 3) if that fails, use placeholderPicks
    picks = base.get("picks")
    if not picks:
        picks = load_model_picks()

    # summary text
    summaryBullets = base.get("summaryBullets")
    if not summaryBullets:
        # figure out how to display time frame
        if timeFrameDisplay:
            displayTime = timeFrameDisplay  # "4 Years", "6 Months", "120 Days", etc.
        elif timeFrame is not None:
            displayTime = f"{timeFrame:.2f} years"
        else:
            displayTime = f"{years} years"

        # pick which dollar amount to show
        if startAmt is not None:
            budgetNumber = startAmt
        else:
            budgetNumber = budget

        summaryBullets = [
            f"We found {len(picks)} strong value stocks matching your {risk.lower()} risk profile.",
            f"These reccomendations are tailored for a ${budgetNumber:,.0f} investment over {displayTime}.",
            "The suggested mix balances profits, growth, and reasonable debt.",
        ]

    # header
    header = html.div(
        {"style": dashboardHeaderBox},
        html.h1({"style": dashboardHeaderTitle}, "Personalized Investor Dashboard"),
        html.p(
            {"style": dashboardHeaderSubtitle},
            "Here are the top stock ideas based on your goals.",
        ),
    )

    # summary card
    summaryCard = html.div(
        dashboardCard,
        html.p({"style": dashboardSummaryTitleText}, "Your Quick Summary"),
        html.ul(
            {"style": dashboardSummaryList},
            *[html.li(line) for line in summaryBullets],
        ),
    )

    # table header
    tableHeader = html.tr(
        {},
        html.th({"style": stockTableHeaderCell}, "Company"),
        html.th({"style": stockTableHeaderCell}, "Smart Score"),
        html.th({"style": stockTableHeaderCell}, "Value Metrics"),
        html.th({"style": stockTableHeaderCell}, "Risk Level"),
        html.th({"style": stockTableHeaderCellRight}, ""),
    )

    # table rows
    rows = []
    for index, stock in enumerate(picks):
        rows.append(
            html.tr(
                {"key": f"stock{index}", "style": stockTableRow},
                html.td(
                    {"style": stockCompanyCell},
                    html.div(
                        {"style": stockCompanyBox},
                        html.span({"style": stockCompanyNameText}, stock["company"]),
                        html.span({"style": stockTickerText}, stock["ticker"]),
                    ),
                ),
                html.td({"style": stockScoreCell}, scoreBar(stock["score"])),
                html.td(
                    {"style": stockTagsCell},
                    html.div(
                        {"style": stockTagsRow},
                        *[tagPill(tag) for tag in stock["tags"]],
                    ),
                ),
                html.td({"style": stockRiskCell}, riskBadge(stock["risk"])),
                html.td(
                    {"style": stockDetailsCell},
                    html.button({"style": stockDetailsButton}, "Details"),
                ),
            )
        )

    # picks card
    picksCard = html.div(
        dashboardCard,
        html.div(
            {"style": picksHeaderRow},
            html.span({"style": picksTitleText}, "Your Top Stock Picks"),
        ),
        html.table(
            {"style": stockTable},
            html.thead({}, tableHeader),
            html.tbody({}, *rows),
        ),
    )

    # chart placeholder
    chartsCard = html.div(
        dashboardCard,
        html.div(
            {"style": chartsHeaderRow},
            html.span({"style": chartsTitleText}, "Performance & Insights"),
            html.span(
                {"style": chartsSubtitleText},
                "Charts & ML explanations eventually",
            ),
        ),
        html.div({"style": chartsPlaceholderBox}, "Graph area placeholder"),
    )

    # stack everything
    return html.div({"style": dashboardStack}, header, summaryCard, picksCard, chartsCard)
