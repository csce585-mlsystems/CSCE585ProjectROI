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
        html.div({"style": chartsPlaceholderBox}, "Graph area placeholder"),
    )
    # stack everything
    return html.div({"style": dashboardStack}, header, summaryCard, picksCard, chartsCard)