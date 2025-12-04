# constants.py
# basic app config
HOST = "127.0.0.1"
PORT = 5001
DEBUG = True
URL = "http://127.0.0.1:5000"
# colors + fonts for whole program
UI = {
    "font": "system-ui, sans-serif",
    "background": "#3b3b3b",
    "text": "#ffffff",
    "lightText": "#cbd5e1",
    "helpText": "#b3b3b3",
}
# wrapper
pageWrapper = {
    "background": UI["background"],
    "position": "fixed",
    "inset": "0",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
    "alignItems": "center",
}
# api routes
API_BASE_STATE0 = "/api/state0"
API_BASE_STATE1 = "/api/state1"
API_BASE_STATE2 = "/api/state2"
loginRoute = f"{API_BASE_STATE0}/login"
signupRoute = f"{API_BASE_STATE0}/signup"
recommendRoute = f"{API_BASE_STATE1}/recommend"
portfolioRoute = f"{API_BASE_STATE2}/portfolio"
# state0 login box
loginBox = {
    "maxWidth": "420px",
    "width": "100%",
    "margin": "0 auto",
    "display": "flex",
    "flexDirection": "column",
    "alignItems": "center",
    "justifyContent": "center",
    "fontFamily": UI["font"],
    "color": UI["text"],
    "background": "#2e2e2e",
    "padding": "40px",
    "borderRadius": "16px",
    "boxShadow": "0 8px 16px rgba(0,0,0,0.3)",
    "textAlign": "center",
}
# state0 inputs
loginInput = {
    "width": "90%",
    "padding": "12px",
    "margin": "10px 0",
    "borderRadius": "10px",
    "border": "none",
    "fontSize": "1rem",
    "background": "#3a3a3a",
    "color": "#fff",
    "outline": "none",
    "textAlign": "left",
}
# base login button
signInButton = {
    "marginTop": "16px",
    "width": "90%",
    "padding": "12px",
    "borderRadius": "30px",
    "border": "none",
    "fontWeight": 600,
    "fontSize": "1rem",
}
signInButtonEnabled = {
    **signInButton,
    "cursor": "pointer",
    "background": "#000",
    "color": "#fff",
}
signInButtonDisabled = {
    **signInButton,
    "cursor": "not-allowed",
    "background": "#777",
    "color": "#fff",
}
signupLink = {
    "alignSelf": "flex-start",
    "marginLeft": "5%",
    "marginTop": "12px",
    "fontSize": "0.9rem",
    "color": "#fff",
    "cursor": "pointer",
    "textDecoration": "underline",
}
loginTitle = {
    "marginBottom": "20px",
}
errorText = {
    "color": "crimson",
    "marginTop": "15px",
}
authContainer = {
    "width": "100%",
    "maxWidth": "480px",
}
loginMessage = {
    "color": UI["text"],
    "marginBottom": "40px",
    "fontSize": "1rem",
    "fontWeight": 500,
}
# state1
goalInputBox = {
    **loginBox,
    "maxWidth": "720px",
    "textAlign": "left",
}
goalInputField = {
    **loginInput,
    "width": "100%",
}
generateRecsButton = {
    **signInButton,
    "width": "100%",
    "borderRadius": "10px",
    "background": "#3a3a3a",
    "color": "#ffffff",
}
inputLabel = {
    "color": UI["lightText"],
    "fontSize": "14px",
    "marginBottom": "8px",
}
inputHelp = {
    "color": UI["helpText"],
    "fontSize": "12px",
    "marginTop": "6px",
}
# state2 
dashboardSidebar = {
    "width": "16.666%",
    "minWidth": "240px",
    "maxWidth": "300px",
    "height": "100vh",
    "background": "#2e2e2e",
    "borderRight": "2px solid #111",
    "boxShadow": "inset -10px 0 20px rgba(0,0,0,0.35)",
    "padding": "24px 20px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "18px",
    "position": "fixed",
    "top": "0",
    "left": "0",
}
dashboardContent = {
    "position": "relative",
    "height": "100vh",
    "overflowY": "auto",
    "background": UI["background"],
    "color": UI["text"],
    "padding": "32px 48px 48px",
    "marginLeft": "clamp(280px, 19%, 340px)",
    "flex": "1",
    "boxSizing": "border-box",
}
dashboardCard = {
    "background": "#2e2e2e",
    "border": "1px solid #1e293b",
    "borderRadius": "12px",
    "padding": "18px",
    "color": UI["lightText"],
    "width": "100%",
}
sidebarButton = {
    "display": "block",
    "textAlign": "left",
    "padding": "12px 16px",
    "borderRadius": "12px",
    "border": "none",
    "cursor": "pointer",
    "background": "transparent",
    "color": "#d0d0d0",
    "fontSize": "1rem",
    "fontWeight": 500,
    "width": "100%",
    "boxSizing": "border-box",
    "transition": "background 0.25s ease, color 0.25s ease",
    "margin": "2px 0",
}
sidebarButtonActive = {
    **sidebarButton,
    "background": "#000",
    "color": "#fff",
    "fontWeight": 600,
}
dashboardHeaderBox = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "4px",
    "marginBottom": "6px",
}
dashboardHeaderTitle = {
    "margin": 0,
    "fontSize": "1.6rem",
}
dashboardHeaderSubtitle = {
    "margin": 0,
    "fontSize": "0.9rem",
    "color": UI["lightText"],
}
dashboardSummaryTitleText = {
    "marginTop": 0,
    "marginBottom": "8px",
    "fontWeight": 600,
}
dashboardSummaryList = {
    "margin": 0,
    "paddingLeft": "18px",
    "fontSize": "0.9rem",
    "color": UI["lightText"],
}
stockTable = {
    "width": "100%",
    "borderCollapse": "collapse",
    "fontSize": "0.85rem",
}
stockTableHeaderCell = {
    "textAlign": "left",
    "padding": "6px 0",
}
stockTableRow = {
    "borderTop": "1px solid rgba(255,255,255,0.08)",
}
stockTableHeaderCellRight = {
    **stockTableHeaderCell,
    "textAlign": "right",
}
stockCompanyCell = {
    "padding": "10px 0",
    "verticalAlign": "middle",
}
stockCompanyBox = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "2px",
}
stockCompanyNameText = {
    "fontSize": "0.95rem",
    "fontWeight": 500,
    "color": UI["text"],
}
stockTickerText = {
    "fontSize": "0.75rem",
    "color": UI["lightText"],
}
stockScoreCell = {
    "padding": "10px 12px",
    "width": "220px",
}
stockTagsCell = {
    "padding": "10px 12px",
}
stockTagsRow = {
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "6px",
}
stockRiskCell = {
    "padding": "10px 12px",
}
stockDetailsCell = {
    "padding": "10px 0",
    "textAlign": "right",
}
stockDetailsButton = {
    "padding": "6px 10px",
    "borderRadius": "999px",
    "border": "1px solid rgba(255,255,255,0.25)",
    "background": "transparent",
    "color": UI["text"],
    "fontSize": "0.8rem",
    "cursor": "pointer",
}
scoreBarRow = {
    "display": "flex",
    "alignItems": "center",
    "gap": "8px",
}
scoreBarTrack = {
    "flex": "1",
    "height": "8px",
    "borderRadius": "999px",
    "background": "rgba(0,0,0,0.35)",
    "overflow": "hidden",
}
scoreBarFillBase = {
    "height": "100%",
}
scoreBarText = {
    "fontSize": "0.8rem",
    "fontWeight": 600,
}
tagChip = {
    "fontSize": "0.7rem",
    "padding": "2px 8px",
    "borderRadius": "999px",
    "border": "1px solid rgba(255,255,255,0.2)",
    "background": "rgba(0,0,0,0.35)",
}
riskBadgeBox = {
    "display": "inline-flex",
    "alignItems": "center",
    "gap": "6px",
    "fontSize": "0.75rem",
    "padding": "2px 10px",
    "borderRadius": "999px",
    "background": "rgba(0,0,0,0.35)",
}
riskBadgeDot = {
    "width": "8px",
    "height": "8px",
    "borderRadius": "999px",
}
chartsHeaderRow = {
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
    "marginBottom": "8px",
}
chartsTitleText = {
    "fontWeight": 600,
}
chartsSubtitleText = {
    "fontSize": "0.8rem",
    "color": UI["lightText"],
}
chartsPlaceholderBox = {
    "height": "160px",
    "borderRadius": "12px",
    "border": "1px dashed rgba(255,255,255,0.3)",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "fontSize": "0.85rem",
    "color": UI["lightText"],
}
sidebarTopSection = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "14px",
    "marginBottom": "10px",
}
sidebarUserRow = {
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
}
sidebarUserAvatar = {
    "width": "32px",
    "height": "32px",
    "borderRadius": "10px",
    "background": "#3b82f6",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "fontWeight": 700,
    "fontSize": "0.9rem",
}
sidebarUserNameText = {
    "margin": 0,
    "fontWeight": 600,
    "fontSize": "1rem",
}
sidebarDivider = {
    "border": "none",
    "borderTop": "1px solid #111",
    "margin": "10px 0 8px",
}
sidebarGoalsContainer = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "16px",
    "paddingLeft": "16px",
    "paddingRight": "16px",
}
sidebarGoalsTitleText = {
    "fontSize": "0.8rem",
    "textTransform": "uppercase",
    "letterSpacing": "0.06em",
    "color": UI["lightText"],
    "margin": 0,
}
sidebarSliderGroup = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "6px",
}
sidebarSliderHeaderRow = {
    "display": "flex",
    "justifyContent": "space-between",
    "fontSize": "0.85rem",
    "color": UI["text"],
}
sidebarSliderRangeInput = {
    "width": "100%",
}
sidebarSliderRangeLabelsRow = {
    "display": "flex",
    "justifyContent": "space-between",
    "fontSize": "0.75rem",
    "color": UI["lightText"],
}
sidebarRiskGroup = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "6px",
}
sidebarRiskLabelText = {
    "fontSize": "0.85rem",
    "color": UI["text"],
}
sidebarRiskRow = {
    "display": "flex",
    "flexDirection": "row",
    "gap": "6px",
}
sidebarGenerateButton = {
    "marginTop": "6px",
    "padding": "12px",
    "borderRadius": "10px",
    "border": "none",
    "background": "#3b82f6",
    "color": "#ffffff",
    "fontWeight": 600,
    "fontSize": "0.9rem",
    "cursor": "pointer",
    "width": "100%",
}
riskPillBaseStyle = {
    "flex": "1",
    "padding": "6px 0",
    "borderRadius": "999px",
    "fontSize": "0.8rem",
    "cursor": "pointer",
    "transition": "0.2s",
}
dashboardWrapper = {
    **pageWrapper,
    "flexDirection": "row",
    "justifyContent": "flex-start",
    "alignItems": "flex-start",
    "background": UI["background"],
    "fontFamily": UI["font"],
}
picksHeaderRow = {
    "display": "flex",
    "justifyContent": "flex-start",
    "alignItems": "center",
    "marginBottom": "8px",
}
# stack layout for the full dash tab page
dashboardStack = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "16px",
}
picksHeaderRow = {
    "display": "flex",
    "justifyContent": "flex-start",
    "alignItems": "center",
    "marginBottom": "8px",
}
picksTitleText = {
    "fontWeight": 600,
}
dashboardStack = {
    "display": "flex",
    "flexDirection": "column",
    "gap": "16px",
}