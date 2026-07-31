import os
import sys

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import gradio as gr
import spaces
from webapp.app import app as flask_app, init_app

@spaces.GPU
def dummy_gpu_function(text):
    return "ZeroGPU is active!"

gr_interface = gr.Interface(fn=dummy_gpu_function, inputs="text", outputs="text")

if __name__ == "__main__":
    init_app()
    # Launch Gradio on a background port to satisfy ZeroGPU's supervisor hooks
    gr_interface.launch(server_port=7861, prevent_thread_lock=True)
    
    # Run the actual Flask app via ASGI on the exposed port (7860)
    fastapi_app = FastAPI()
    fastapi_app.mount("/", WSGIMiddleware(flask_app))
    
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)
