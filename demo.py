import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        gr.Image("your_image_path.png")  # Replace with your image file path
        gr.Markdown("Your text goes here.")

demo.launch()
