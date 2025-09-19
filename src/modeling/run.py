"""ASGI entry point that mounts the Gradio demo inside a FastAPI app."""

import gradio as gr
from fastapi import FastAPI

from gradio_ui import demo

app = FastAPI()


@app.get("/")
async def root() -> tuple[str, int]:
    """Return a friendly pointer to the mounted Gradio interface.

    Returns:
        tuple[str, int]: Message and HTTP status code indicating where to find
        the interactive UI.
    """

    return "Gradio app is running at /gradio", 200


# Expose the interactive demo under ``/gradio`` while keeping the FastAPI app
# extensible for future health checks or auxiliary endpoints.
app = gr.mount_gradio_app(app, demo, path="/gradio")
