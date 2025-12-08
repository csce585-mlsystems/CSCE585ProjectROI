# componentSet1.py
from reactpy import component, html, hooks
import asyncio, httpx, pdb as pb
# C:\Users\adoct\Notes for CSCE Classes[Fall 2025]\Notes for CSCE 585\ProjectRepo\projectCode\components\State1\componentSet1.py
from ...MLLifecycle import script as SCR
from constants import (
    URL,
    UI,
    pageWrapper,
    recommendRoute,
    goalInputBox,
    goalInputField,
    generateRecsButton,
    inputLabel,
    inputHelp,
)
# send POST request to bsck recommendations
async def postToBackend(route, payload):
    fullUrl = URL.rstrip("/") + route
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(fullUrl, json=payload, timeout=20)
        try:
            return response.status_code, response.json()
        except Exception:
            return response.status_code, response.text
    except Exception as ex:
        return 0, f"{type(ex).__name__}: {ex}"
# keep $ formatting clean
def formatMoney(value):
    text = str(value).strip().replace("$", "")
    if not text:
        return ""
    return "$" + text
# money field
def moneyAmountField(label, value, setter, placeholder, helpTextText):
    def handleChange(event):
        newValue = formatMoney(event["target"]["value"])
        setter(newValue)
    return html.div(
        {"style": {"marginBottom": "18px"}},
        [
            html.label({"style": inputLabel}, label),
            html.input(
                {
                    "type": "text",
                    "style": goalInputField,
                    "value": value,
                    "placeholder": "$" + placeholder if not value else value,
                    "onChange": handleChange,
                }
            ),
            html.div({"style": inputHelp}, helpTextText),
        ],
    )
# time frame (number & unit)
def timeFrameField(timeFrame, setTimeFrame, timeFrameUnit, setTimeFrameUnit):
    def handleUnitChange(event):
        newUnit = event["target"]["value"]
        setTimeFrameUnit(newUnit)
    def handleTimeChange(event):
        newTime = event["target"]["value"]
        setTimeFrame(newTime)
    return html.div(
        {"style": {"marginBottom": "18px"}},
        [
            html.div(
                {
                    "style": {
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                    }
                },
                [
                    html.label(
                        {"style": inputLabel},
                        "How long are you investing for?",
                    ),
                    html.select(
                        {
                            "style": {
                                "background": "#3a3a3a",
                                "color": "#fff",
                                "border": "none",
                                "borderRadius": "8px",
                                "padding": "6px 8px",
                                "fontSize": "0.9rem",
                            },
                            "value": timeFrameUnit,
                            "onChange": handleUnitChange,
                        },
                        [
                            html.option({"value": "Years"}, "Years"),
                            html.option({"value": "Months"}, "Months"),
                            html.option({"value": "Days"}, "Days"),
                        ],
                    ),
                ],
            ),
            html.input(
                {
                    "type": "number",
                    "style": goalInputField,
                    "value": timeFrame,
                    "placeholder": "5",
                    "onChange": handleTimeChange,
                }
            ),
        ],
    )
# convert  from  years/months to days for annual comp. math
def convertToYears(value, unit):
    try:
        number = float(value)
    except Exception:
        return 0

    if unit == "Months":
        return number / 12
    if unit == "Days":
        return number / 365
    return number
def convertToDays(value, unit):
    try:
        number = float(value)
    except Exception:
        return 0

    if unit == "Months":
        return 28*number
    if unit == "Years":
        return number*365
    return number
# format
def buildTimeFrameDisplay(value, unit):
    try:
        number = float(value)
    except Exception:
        return None
    if number == 1 and unit.endswith("s"):
        labelUnit = unit[:-1]
    else:
        labelUnit = unit
    # :g keeps it from showing .0 unless needed
    return f"{number:g} {labelUnit}"

# Body of allowing choice of tickers
def choosingTickers(noTable = True):
    if noTable == True:
        companies: list[str] = ["GOOG","AAPL", "AMZN", "MSFT", "META"] #<-- will need to reference stocks desired by users. Can populate here though. 
        return companies


# End of Body of allowing choice of tickers
global timeFrameDays
timeFrameDays = 0;
@component
def State1View(onSuccess=None, username=None):
    startAmount, setStartAmount = hooks.use_state("")
    targetAmount, setTargetAmount = hooks.use_state("")
    timeFrame, setTimeFrame = hooks.use_state("")
    timeFrameUnit, setTimeFrameUnit = hooks.use_state("Years")
    loading, setLoading = hooks.use_state(False)
    error, setError = hooks.use_state(None)
    # call backend to get recommendations
    async def submit():
        setLoading(True)
        setError(None)
        try:
            global timeFrameDays
            timeFrameYears = convertToYears(timeFrame, timeFrameUnit)
            timeFrameDays = convertToDays(timeFrame, timeFrameUnit)
            timeFrameLabel = buildTimeFrameDisplay(timeFrame, timeFrameUnit)
            payload = {
                "username": username,
                "startAmt": startAmount,
                "targetAmt": targetAmount,
                "timeFrame": timeFrameYears,
                "timeFrameTester": timeFrameDays,
                "timeFrameDisplay": timeFrameLabel,
            }
            status, data = await postToBackend(recommendRoute, payload)
            if status == 200 and isinstance(data, dict) and data.get("ok"):
                if onSuccess:
                    onSuccess("state2", data)
            else:
                message = data.get("message") if isinstance(data, dict) else str(data)
                setError(message or "Request failed.")
        except Exception as ex:
            setError(str(ex))
        finally:
            setLoading(False)
    # generate button
    def handleGenerateClick(_event):
        # NOTE: Embedding data pipeline here!
        # print("--DEBUGGING CHECKPOINT: Testing case where script is embedded here instead---")
        # pb.set_trace() #<-- NOTE: Works as intended
        SCR.main(listOfCompanies=choosingTickers(True), dateToPullFrmStock=timeFrameDays)

        asyncio.create_task(submit())
    # handler for generate button
    def handleGenerateKeyDown(event):
        key = event.get("key")
        if key in ("Enter", " "):
            asyncio.create_task(submit())
    # top header
    header = html.div(
        {"style": {"textAlign": "center", "marginBottom": "18px"}},
        [
            html.h1(
                {
                    "style": {
                        "color": UI["text"],
                        "fontSize": "28px",
                        "fontWeight": 800,
                        "margin": 0,
                    }
                },
                "Tell Us Your Goals",
            ),
            html.h2(
                {
                    "style": {
                        "color": UI["text"],
                        "fontSize": "22px",
                        "fontWeight": 700,
                        "marginTop": "10px",
                    }
                },
                "We will use this to build your stock picks.",
            ),
            html.p(
                {
                    "style": {
                        "color": UI["helpText"],
                        "marginTop": "4px",
                    }
                },
                "Enter your values and we'll generate some recs.",
            ),
        ],
    )
    # main bttn
    buttonStyle = {**generateRecsButton, "cursor": "pointer"}
    button = html.button(
        {
            "type": "button",
            "style": buttonStyle,
            "onClick": handleGenerateClick,
            "onKeyDown": handleGenerateKeyDown,
        },
        "Generate Recommendations",
    )
    # fields page
    elements = [
        header,
        moneyAmountField(
            "How much are you starting with?",
            startAmount,
            setStartAmount,
            "5,000",
            "Your initial investment.",
        ),
        moneyAmountField(
            "What amount do you want to reach?",
            targetAmount,
            setTargetAmount,
            "20,000",
            "Your goal amount.",
        ),
        timeFrameField(timeFrame, setTimeFrame, timeFrameUnit, setTimeFrameUnit),
        button,
    ]
    if loading:
        elements.append(
            html.p(
                {"style": {"color": UI["lightText"], "marginTop": "12px"}},
                "Loading...",
            )
        )
    if error:
        elements.append(
            html.p(
                {"style": {"color": "crimson", "marginTop": "12px"}},
                error,
            )
        )
    form = html.div({"style": goalInputBox}, elements)
    return html.div(
        {"style": {**pageWrapper, "fontFamily": UI["font"]}},
        [form],
    )