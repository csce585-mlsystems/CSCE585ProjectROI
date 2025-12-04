
# Body of imports required for loading and using model
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np #<-- May be optional not sure as of 10/13/25. 
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
# placeholder picks (ML model will replace)
# Body of function that automatically fills in contents of placeHolderPicks using contents derived from model's output: 







# End of Body of function that automatically fills in contents of placeHolderPicks using contents derived from model's output: 
# placeholderPicks[i]["company"] = tickerRealNameTable[dataFrameReffingModelPred.loc[i,"Company"]]

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
placeholderPicks = [{"tags": ["Tag A", "Tag B"],
        "risk": "High",}
                    ,{"tags": ["Tag A", "Tag B"],
        "risk": "Medium",},
                    {"tags": ["Tag A", "Tag B"],
        "risk": "Low",}]
tickerRealNameTable = {
        "AAPL": "Apple",
        "GOOG": "Google",
        "AMZN": "Amazon",
        "MSFT": "Microsoft",
        "META": "Meta Platforms",
        "NVDA": "Nvidia Corporation",
        "TSLA": "Tesla, Inc",
        "BRK-B": "Berkshire Hathway Inc",
        "JPM": "JPMorgan Chase & Co",
        "V": "Visa Inc",
        "MA": "Mastercard Incorporated",
        "HD": "The Home Depot",
        "NFLX": "Netflix, Inc",
        "DIS": "The Walt Disney Company",
        "PEP": "PepsiCo, Inc",
        "KO": "The Coca-Cola Company",
        "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation",
        "ADBE": "Adobe Inc",
        "CSCO": "Cisco Systems, Inc"
        }
addingExtraTickers = False
if (addingExtraTickers == True):
    placeholderPicks.append([(dict() for i in range(len(tickerRealNameTable)))]) #<-- Used to allocate space for all picks
def fillPicks(modelPrediction = None):
    # Goal: a) Obtain companies chosen by user, [NOTE: Within these steps, will need to figure out how to initiate data engineering pipeline process to create said model to obtain prediction ]b) Call model to make prediction, c) Obtain model's prediction, d) Process model's prediction for user consumption.  
    # Using if-else below to constrain amt of admissible predictions for demo purposes.
    # if(modelPrediction == None and len(modelPrediction) > 4):
    if(modelPrediction != None):
        """
        new_model = tf.keras.models.load_model('my_model.keras')
        predictions = new_model.predict()
        
        NOTE: Below will be used to do conversion, ensuring that palceHolder Picks is populated [PN: May also require presence of test_stocks and ]
            test_stocks.loc[:,"Company"] = train_stocks.loc[:,"Company"].astype("str") #<-- Used to convert one-hot encoding back into strings interpretable by users. 
        np.argsort(predictions[0]) #<-- NOTE: argsort sorts argument indices in ascending order! [UPDATE: There is also an edge case, if preds coincidentally are the same, then the next maximum should be pulled instead]
        np.argsort(predictions[1])
        np.argsort(predictions[2])
        np.argsort(predictions[3])

        test_stocks[test_stocks["Optimality"] == np.argmax(predictions[0])]["Company"]
        test_stocks[test_stocks["Optimality"] == [x for x in np.argsort(predictions[1]) if x != np.argsort(predictions[0])[-1]][-1]]["Company"]
        test_stocks[test_stocks["Optimality"] == [x for x in np.argsort(predictions[2]) if x != np.argsort(predictions[0])[-1] and x != np.argsort(predictions[1])[-1] ][-1]]["Company"]
        test_stocks[test_stocks["Optimality"] == [x for x in np.argsort(predictions[3]) if x != np.argsort(predictions[0])[-1] and x != np.argsort(predictions[1])[-1] and x != np.argsort(predictions[2])[-2]][0] ]["Company"]


        
        np.argmax(predictions[0])
        [x for x in np.argsort(predictions[1]) if x != np.argsort(predictions[0])[-1]][-1]
        [x for x in np.argsort(predictions[2]) if x != np.argsort(predictions[0])[-1] and x != np.argsort(predictions[1])[-1] ][-1]
        [x for x in np.argsort(predictions[3]) if x != np.argsort(predictions[0])[-1] and x != np.argsort(predictions[1])[-1] and x != np.argsort(predictions[2])[-2]]

        UPDATE: Decided to have model's predictions as a dataframe referencing the test_stock data AND the model's predictions as columns. Thus, above will only occur within backend of model. 
        
        """
    else:
        # """
        global tickerRealNameTable
                # ^^ Add more above later on!
        local = True
        writePathForModelPreds = "C:/Users/adoct/Notes for CSCE Classes[Fall 2025]/Notes for CSCE 585/ProjectRepo/projectCode/MLLifecycle/ModelDevelopmentAndTraining/ModelPredictions.csv" if local == True else "./ModelPredictions.csv" #<-- '""' Needs to refer to virtual environment. [VIRTUAL ENVIRONMENT ADDRESS THING[NOTE]: Will need to change this file path to adhere to virtual environment!] 
        filePathToModelPredFile = writePathForModelPreds #<-- updating soon
        dataFrameReffingModelPred = pd.read_csv(f"{filePathToModelPredFile}") #<-- will change soon to dataframe refrencing a dataframe referencing the test_stock data AND the model's predictions as columns!
        # NOTE: Be\, instead of doing for, we will do three for demo purposes, hence why
        # offset '-1' is present. 
        isOrderOfTickerTablePrecedence = False
        # print("---DEBUGGING CHECKPOINT: VALIDATING PROCESSES--")
        # pb.set_trace()
        for i in range((dataFrameReffingModelPred.shape[0] - 1) if not isOrderOfTickerTablePrecedence else len(tickerRealNameTable)):
            # NOTE: May need a table that refs the ticker-Actual name pairs to be used below!
            placeholderPicks[i]["company"] = tickerRealNameTable[dataFrameReffingModelPred.loc[i,"Company"]]
            placeholderPicks[i]["ticker"] = dataFrameReffingModelPred.loc[i, "Company"]
            placeholderPicks[i]["score"] = dataFrameReffingModelPred.loc[i, "Optimality"]
            placeholderPicks[i]["tags"][0] = f"Price to Earnings Ratio: {dataFrameReffingModelPred.loc[i, "P/E"]}"
            placeholderPicks[i]["tags"][1] = f"Share Price: {dataFrameReffingModelPred.loc[i, "Share Price"]}"
        
        return
        # NOTE: May add something that grabs top 3 companies for any set of companies given.
        # NOTE: Nevertheles,s I believe this is all the code that is NEEDED. Currently
        # working towards writing file that will be read in by pd.read_csv call above. 
        # """
fillPicks()
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
    picks = base.get("picks", placeholderPicks)
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
        # html.div({"style": chartsPlaceholderBox}, "Graph area placeholder"),
        html.div({"style": chartsPlaceholderBox}, 
                 html.img({"src": "C:/users/adoct/downloads/picture.jpg" })),
    )
    # stack everything
    return html.div({"style": dashboardStack}, header, summaryCard, picksCard, chartsCard)