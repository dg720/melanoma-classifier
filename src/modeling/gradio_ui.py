import gradio as gr

from predict import get_predictions

demo = gr.Interface(
    fn=get_predictions,
    inputs=gr.components.Image(label="Input"),
    outputs=gr.components.Label(label="Predictions"),
    allow_flagging="never",
)
