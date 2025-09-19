"""Gradio interface definition for interactive melanoma predictions."""

import gradio as gr

from predict import get_predictions

# The minimalist demo exposes probability scores directly, making the model
# easier to sanity-check during development sessions.
demo = gr.Interface(
    fn=get_predictions,
    inputs=gr.components.Image(label="Input"),
    outputs=gr.components.Label(label="Predictions"),
    allow_flagging="never",
)
