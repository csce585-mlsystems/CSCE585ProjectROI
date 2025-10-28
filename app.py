# app.py
from flask import Flask
from reactpy.backend.flask import configure

# proj code -> comp/routes -> state# -> .py
from projectCode.components.State0.componentSet0 import RootView
from projectCode.routes.State0Routes import state0_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(state0_bp)   # =/api/state0/*
    configure(app, RootView)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=True)
