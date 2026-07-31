import os
import sys

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import gradio as gr
import spaces
from webapp.app import app as flask_app, init_app

fastapi_app = FastAPI()

@spaces.GPU
def dummy_gpu_function(text):
    return "ZeroGPU is active!"

gr_interface = gr.Interface(fn=dummy_gpu_function, inputs="text", outputs="text")

fastapi_app = gr.mount_gradio_app(fastapi_app, gr_interface, path="/dummy")
fastapi_app.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    init_app()
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
