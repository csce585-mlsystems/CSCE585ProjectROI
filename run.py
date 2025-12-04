# run.py
from flask import Flask
from reactpy.backend.flask import configure
from constants import HOST, PORT, DEBUG
from dotenv import load_dotenv
load_dotenv()
# components + routes
# ScreenController as main UI entry pt
from projectCode.components.State0.componentSet0 import ScreenController
from projectCode.routes.State0Routes import state0_bp
from projectCode.routes.State1Routes import state1_bp
from projectCode.routes.State2Routes import state2_bp
def create_app():
    app = Flask(__name__)
    # register api routes for each state
    app.register_blueprint(state0_bp)   # /api/state0/
    app.register_blueprint(state1_bp)   # /api/state1/
    app.register_blueprint(state2_bp)   # /api/state2/
    # connect ReactPy frontend
    configure(app, ScreenController)
    return app
if __name__ == "__main__":
    app = create_app()
    app.run(host=HOST, port=PORT, debug=DEBUG)