import gradio as gr

# Define a simple function for demonstration
def greet(name):
    return f"Hello, {name}!"

# Create a Gradio interface
demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Your Name"),
    outputs=gr.Textbox(label="Greeting")
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
