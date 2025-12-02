# componentSet0.py
from reactpy import component, html, hooks
import asyncio, httpx
from constants import (
    URL,
    pageWrapper,
    loginBox,
    loginInput,
    signInButtonEnabled,
    signInButtonDisabled,
    signupLink,
    loginTitle,
    errorText,
    authContainer,
    loginMessage,
    loginRoute,
    signupRoute,
)
# helper send POST to backend
async def postToBackend(route, payload):
    fullUrl = URL.rstrip("/") + route
    print("POST to", fullUrl, "payload:", payload)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(fullUrl, json=payload, timeout=15)
        try:
            return response.status_code, response.json()
        except:
            return response.status_code, (response.text or "")
    except Exception as ex:
        return 0, f"{type(ex).__name__}: {ex}"
# login screen
@component
def State0View(onSuccess=None, onCreateAccount=None):
    username, setUsername = hooks.use_state("")
    password, setPassword = hooks.use_state("")
    error, setError = hooks.use_state(None)
    loading, setLoading = hooks.use_state(False)
    # track username input
    def handleUsernameChange(event):
        text = (event["target"]["value"] or "").strip()
        setUsername(text)
    # track password input
    def handlePasswordChange(event):
        text = (event["target"]["value"] or "").strip()
        setPassword(text)
    # call login api
    async def submit():
        if not username or not password:
            setError("Please enter username and password.")
            return
        setLoading(True)
        setError(None)
        status, data = await postToBackend(
            loginRoute,
            {"username": username, "password": password},
        )
        setLoading(False)
        if status == 200 and isinstance(data, dict):
            if onSuccess:
                merged = dict(data)
                merged["username"] = username
                nextState = merged.get("next_state", "state1")
                onSuccess(nextState, merged)
        else:
            message = data.get("message", "") if isinstance(data, dict) else str(data)
            setError(message or "Login failed.")
    # click handler sign in
    def handleLoginClick(_event):
        asyncio.create_task(submit())

    # click handler Sign Up?
    def handleSignupClick(_event):
        if onCreateAccount:
            onCreateAccount()

    canSubmit = bool(username and password and not loading)
    buttonStyle = signInButtonEnabled if canSubmit else signInButtonDisabled

    return html.div(
        {"style": loginBox},
        [
            html.h2({"style": loginTitle}, "Login"),
            html.input(
                {
                    "placeholder": "Username",
                    "value": username,
                    "onChange": handleUsernameChange,
                    "autoComplete": "username",
                    "style": loginInput,
                }
            ),
            html.input(
                {
                    "placeholder": "Password",
                    "type": "password",
                    "value": password,
                    "onChange": handlePasswordChange,
                    "autoComplete": "current-password",
                    "style": loginInput,
                }
            ),
            html.button(
                {
                    "onClick": handleLoginClick,
                    "disabled": not canSubmit,
                    "style": buttonStyle,
                },
                "Sign In" if not loading else "Signing In...",
            ),
            html.p(
                {
                    "onClick": handleSignupClick,
                    "style": signupLink,
                },
                "Sign Up?",
            ),
            html.p({"style": errorText}, error) if error else html.span(""),
        ],
    )
# create account
@component
def CreateAccountView(onSuccess=None, onGoBack=None):
    username, setUsername = hooks.use_state("")
    email, setEmail = hooks.use_state("")
    phone, setPhone = hooks.use_state("")
    password, setPassword = hooks.use_state("")
    confirmPassword, setConfirmPassword = hooks.use_state("")
    error, setError = hooks.use_state(None)
    loading, setLoading = hooks.use_state(False)
    # create account inputs
    def handleUsernameChange(event):
        setUsername((event["target"]["value"] or "").strip())

    def handleEmailChange(event):
        setEmail((event["target"]["value"] or "").strip())

    def handlePhoneChange(event):
        setPhone((event["target"]["value"] or "").strip())

    def handlePasswordChange(event):
        setPassword((event["target"]["value"] or "").strip())

    def handleConfirmChange(event):
        setConfirmPassword((event["target"]["value"] or "").strip())
    # back to login screen
    def handleBackClick(_event):
        if onGoBack:
            onGoBack()
    # call signup api
    async def submit():
        if not username or not email or not phone or not password:
            setError("Please fill out all fields.")
            return
        if password != confirmPassword:
            setError("Passwords do not match.")
            return
        setLoading(True)
        setError(None)
        status, data = await postToBackend(
            signupRoute,
            {
                "username": username,
                "password": password,
                "email": email,
                "phone": phone,
            },
        )
        setLoading(False)
        if status == 201:
            if onSuccess:
                onSuccess("account_created", {"username": username})
        else:
            message = data.get("message", "") if isinstance(data, dict) else str(data)
            setError(message or "Sign up failed.")

    # handler for create account
    def handleCreateClick(_event):
        asyncio.create_task(submit())

    canSubmit = bool(
        username and email and phone and password and confirmPassword and not loading
    )
    buttonStyle = signInButtonEnabled if canSubmit else signInButtonDisabled
    return html.div(
        {"style": loginBox},
        [
            html.h2({"style": loginTitle}, "Create Account"),
            html.input(
                {
                    "placeholder": "Username",
                    "value": username,
                    "onChange": handleUsernameChange,
                    "style": loginInput,
                }
            ),
            html.input(
                {
                    "placeholder": "Email",
                    "type": "email",
                    "value": email,
                    "onChange": handleEmailChange,
                    "style": loginInput,
                }
            ),
            html.input(
                {
                    "placeholder": "Phone",
                    "type": "tel",
                    "value": phone,
                    "onChange": handlePhoneChange,
                    "style": loginInput,
                }
            ),
            html.input(
                {
                    "placeholder": "Password",
                    "type": "password",
                    "value": password,
                    "onChange": handlePasswordChange,
                    "style": loginInput,
                }
            ),
            html.input(
                {
                    "placeholder": "Confirm Password",
                    "type": "password",
                    "value": confirmPassword,
                    "onChange": handleConfirmChange,
                    "style": loginInput,
                }
            ),
            html.button(
                {
                    "onClick": handleCreateClick,
                    "disabled": not canSubmit,
                    "style": buttonStyle,
                },
                "Create Account" if not loading else "Creating...",
            ),
            html.p(
                {
                    "onClick": handleBackClick,
                    "style": signupLink,
                },
                "Back to Login?",
            ),
            html.p({"style": errorText}, error) if error else html.span(""),
        ],
    )
@component
def ScreenController():
    screen, setScreen = hooks.use_state("login")
    message, setMessage = hooks.use_state(None)
    currentStep, setCurrentStep = hooks.use_state("auth")
    dashboardData, setDashboardData = hooks.use_state(None)
    currentUser, setCurrentUser = hooks.use_state(None)
    # login to register
    def goRegister():
        setMessage(None)
        setScreen("register")
    # back to login
    def goLogin(*_args):
        setScreen("login")
        setMessage("Account created!")
    # successful login
    def handleAuthSuccess(nextState, data):
        username = data.get("username")
        setCurrentUser(username)
        if nextState == "state1":
            setCurrentStep("state1")
        elif nextState == "state2":
            setDashboardData(data)
            setCurrentStep("state2")
    # success from the goal page
    def handleState1Success(nextState, data):
        if nextState == "state2":
            setDashboardData(data)
            setCurrentStep("state2")
    # goal page from dashboard
    def handleNewRecommendation():
        setCurrentStep("state1")
    # show state1
    if currentStep == "state1":
        from projectCode.components.State1.componentSet1 import State1View
        return State1View(onSuccess=handleState1Success, username=currentUser)
    # show state2
    if currentStep == "state2":
        from projectCode.components.State2.componentSet2 import DashboardView
        return DashboardView(data=dashboardData, on_new_recommendation=handleNewRecommendation)
    # def login n register
    return html.div(
        {"style": pageWrapper},
        [
            html.p({"style": loginMessage}, message) if message else html.span(""),
            html.div(
                {"style": authContainer},
                [
                    (
                        State0View(
                            onSuccess=handleAuthSuccess,
                            onCreateAccount=goRegister,
                        )
                        if screen == "login"
                        else CreateAccountView(
                            onSuccess=goLogin,
                            onGoBack=goLogin,
                        )
                    )
                ],
            ),
        ],
    )
@component
def RootView():
    return ScreenController()
