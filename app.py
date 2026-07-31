import os
import sys

from webapp.app import app, init_app

if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=7860, debug=False)
