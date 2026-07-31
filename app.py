import os
import sys

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import gradio as gr
import spaces
from webapp.app import app as flask_app, init_app
import uvicorn

@spaces.GPU
def dummy_gpu_function(text):
    return "ZeroGPU is active!"

gr_interface = gr.Interface(fn=dummy_gpu_function, inputs="text", outputs="text")

if __name__ == "__main__":
    init_app()
    
    fastapi_app = FastAPI()
    fastapi_app = gr.mount_gradio_app(fastapi_app, gr_interface, path="/dummy")
    fastapi_app.mount("/", WSGIMiddleware(flask_app))
    
    original_run = uvicorn.Server.run
    def custom_run(self, *args, **kwargs):
        self.config.app = fastapi_app
        original_run(self, *args, **kwargs)
    uvicorn.Server.run = custom_run
    
    gr_interface.launch(server_name="0.0.0.0", server_port=7860)
