# componentSet2.py
from reactpy import component, html, hooks
from constants import (
    UI,
    pageWrapper,
    dashboardSidebar,
    dashboardContent,
    sidebarButton,
    sidebarButtonActive,
    sidebarTopSection,
    sidebarUserRow,
    sidebarUserAvatar,
    sidebarUserNameText,
    sidebarDivider,
    sidebarGoalsContainer,
    sidebarGoalsTitleText,
    sidebarSliderGroup,
    sidebarSliderHeaderRow,
    sidebarSliderRangeInput,
    sidebarSliderRangeLabelsRow,
    sidebarRiskGroup,
    sidebarRiskLabelText,
    sidebarRiskRow,
    sidebarGenerateButton,
    dashboardWrapper,
    riskPillBaseStyle,
)
from projectCode.components.State2.tabs.dashboard import DashboardTab
from projectCode.components.State2.tabs.portfolio import PortfolioTab
from projectCode.components.State2.tabs.discover import DiscoverTab
from projectCode.components.State2.tabs.settings import SettingsTab
# main dashboard shell sidebar and tabs
@component
def DashboardView(data=None, on_new_recommendation=None):
    activeTab, setActiveTab = hooks.use_state("dashboard")
    initialBudget, initialYears, initialRisk = getGoalDefaults(data)
    budgetAmount, setBudgetAmount = hooks.use_state(initialBudget)
    holdingYears, setHoldingYears = hooks.use_state(initialYears)
    riskLevel, setRiskLevel = hooks.use_state(initialRisk)
    baseData = data or {}
    goalInputs = baseData.get("inputs") or {}
    goalGrowthRate = baseData.get("growthRate")
    # nav button in the sidebar
    def navButton(label, key):
        if activeTab == key:
            buttonStyle = sidebarButtonActive
        else:
            buttonStyle = sidebarButton

        def handleNavClick(_event):
            setActiveTab(key)

        return html.button(
            {
                "style": buttonStyle,
                "onClick": handleNavClick,
            },
            label,
        )
    # slider handlers
    def handleBudgetChange(event):
        try:
            newBudget = int(event["target"]["value"])
            setBudgetAmount(newBudget)
        except Exception:
            pass
    def handleYearsChange(event):
        try:
            newYears = int(event["target"]["value"])
            setHoldingYears(newYears)
        except Exception:
            pass
    # left sidebar info
    sidebarView = html.div(
        {"style": dashboardSidebar},
        [
            html.div(
                {"style": sidebarTopSection},
                [
                    html.div(
                        {"style": sidebarUserRow},
                        [
                            html.div({"style": sidebarUserAvatar}, "U"),
                            html.p({"style": sidebarUserNameText}, "User"),
                        ],
                    ),
                    navButton("Dashboard", "dashboard"),
                    navButton("My Portfolio", "portfolio"),
                    navButton("Discover", "discover"),
                    navButton("Settings", "settings"),
                ],
            ),
            html.hr({"style": sidebarDivider}),
            html.div(
                {"style": sidebarGoalsContainer},
                [
                    html.p(
                        {"style": sidebarGoalsTitleText},
                        "Tell Us Your Goals",
                    ),
                    html.div(
                        {"style": sidebarSliderGroup},
                        [
                            html.div(
                                {"style": sidebarSliderHeaderRow},
                                [
                                    html.span("Your Budget"),
                                    html.span(f"${budgetAmount:,.0f}"),
                                ],
                            ),
                            html.input(
                                {
                                    "type": "range",
                                    "min": "1000",
                                    "max": "50000",
                                    "step": "1000",
                                    "value": str(budgetAmount),
                                    "onChange": handleBudgetChange,
                                    "style": sidebarSliderRangeInput,
                                }
                            ),
                        ],
                    ),
                    html.div(
                        {"style": sidebarSliderGroup},
                        [
                            html.div(
                                {"style": sidebarSliderHeaderRow},
                                [
                                    html.span("Holding Time (Years)"),
                                    html.span(f"{holdingYears} yrs"),
                                ],
                            ),
                            html.input(
                                {
                                    "type": "range",
                                    "min": "1",
                                    "max": "10",
                                    "step": "1",
                                    "value": str(holdingYears),
                                    "onChange": handleYearsChange,
                                    "style": sidebarSliderRangeInput,
                                }
                            ),
                            html.div(
                                {"style": sidebarSliderRangeLabelsRow},
                                [
                                    html.span("1 Year"),
                                    html.span("10 Years"),
                                ],
                            ),
                        ],
                    ),
                    html.div(
                        {"style": sidebarRiskGroup},
                        [
                            html.span(
                                {"style": sidebarRiskLabelText},
                                "Risk Tolerance",
                            ),
                            html.div(
                                {"style": sidebarRiskRow},
                                [
                                    buildRiskPill("Low", riskLevel, setRiskLevel),
                                    buildRiskPill("Medium", riskLevel, setRiskLevel),
                                    buildRiskPill("High", riskLevel, setRiskLevel),
                                ],
                            ),
                        ],
                    ),
                    html.button(
                        {"style": sidebarGenerateButton},
                        "Generate My Picks",
                    ),
                ],
            ),
        ],
    )
    tabData = {
        "budget": budgetAmount,
        "years": holdingYears,
        "risk": riskLevel,
        "inputs": goalInputs,
        "growthRate": goalGrowthRate,
    }
    if activeTab == "dashboard":
        mainView = DashboardTab(data=tabData)
    elif activeTab == "portfolio":
        mainView = PortfolioTab(data=tabData)
    elif activeTab == "discover":
        mainView = DiscoverTab(data=tabData)
    else:
        mainView = SettingsTab(data=tabData)

    contentView = html.div({"style": dashboardContent}, mainView)

    return html.div({"style": dashboardWrapper}, [sidebarView, contentView])
# pull def budget/years/risk from data
def getGoalDefaults(data):
    base = data or {}
    budget = base.get("budget", 10000)
    years = base.get("years", 5)
    risk = str(base.get("risk", "Medium"))
    return budget, years, risk
# risk level
def buildRiskPill(label, current, setValue):
    baseColors = {
        "Low": "#22c55e",
        "Medium": "#3b82f6",
        "High": "#ef4444",
    }
    color = baseColors.get(label, "#3b82f6")
    isActive = current == label

    style = dict(riskPillBaseStyle)
    if isActive:
        style["border"] = f"2px solid {color}"
        style["background"] = color
    else:
        style["border"] = "2px solid rgba(255,255,255,0.2)"
        style["background"] = "#3a3a3a"

    def handlePillClick(_event):
        setValue(label)

    return html.button(
        {
            "style": style,
            "onClick": handlePillClick,
        },
        label,
    )
