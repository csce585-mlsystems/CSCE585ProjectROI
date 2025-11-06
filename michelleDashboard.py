from reactpy import component, html, hooks, event
import asyncio, httpx
import os
from constants import (
    LOGIN_ROUTE,
    SIGNUP_ROUTE,
    STYLE_CARD,
    STYLE_INPUT,
    STYLE_BUTTON,
    STYLE_BUTTON_SECONDARY,
    PAGE_WRAPPER,
    MESSAGE_STYLE,
    URL,
    UI, 
    PAGE_WRAPPER,
    MESSAGE_STYLE,
    STYLE_CARD,
    STYLE_INPUT,
    STYLE_BUTTON,
    STYLE_BUTTON_SECONDARY
)

SIDEBAR_BUTTONS = {

    "DashBoard": {
        "text":"Dashboard",
        "action": "Financial Overview",
        "route": RECOMMEND_ROUTE,   
    },

    "Portfolio":{
        "text": "My Portfolio",
        "route":PORTFOLIO_ROUTE,
    },

    "Discover": {
        "text": "Discover",
        "action": "Show",
    },

    "Settings": {
        "text": "Settings",
        "action":""
    }

}

SUMMMARY_BLOCK = {
    "Summary":{
    "text":"Your Quick Summary",
     "Point1":"",
      "Point2": "",
      "Point3" : ""}
      

},

STOCK_BLOCK = {
   "text": "Your Top Stock Picks",
   "Companies": {
    "Company1": "",
    "Company2": "",
    "Company3": "",},

    "Value Metrics":{
    "Debt_Level":"",
    "Dividend": "",
    "Market": "",
}

}

HEADSHOT = {
    "Name": "",
    'Investor_Type': "",
    "Photo": Image,
}