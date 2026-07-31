import os
import sys
import spaces

@spaces.GPU
def _zero_gpu_bypass():
    pass

from webapp.app import app, init_app

if __name__ == "__main__":
    init_app()
    app.run(host="0.0.0.0", port=7860, debug=False)
