import torch
import spaces
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    TextIteratorStreamer,
)
from threading import Thread
import re

# SETUP
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# lOAD MODEL AND TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#quantization_config=bnb_config,
model = AutoModelForCausalLM.from_pretrained(
    MODEL
).to(device)

# HELPERS
def clean_response(response: str) -> str:
    """
    Removes 'assistant\n\n' from the start of the response if it exists.
    """
    return (
        response.lstrip("assistant\n\n")
        if response.startswith("assistant\n\n")
        else response
    )

def extract_feedback(markdown_text):
    match = re.search(r"### FEEDBACK\n(.+?)(\n###|\Z)", markdown_text, re.DOTALL)
    return match.group(1).strip() if match else "No feedback found."

def user_prompt_for(criteria, draft):
    return f"""
    Evaluate the provided draft based on the given criteria in a single pair of <thinking></thinking> tags and <feedback></feedback tags.
    == CRITERIA ==
    {criteria}

    == DRAFT ==
    {draft}
    """

def get_feedback_prompt(draft, feedback):
    return f"""
    Refine the last draft using the provided feedback.
    == THE PREVIOUS DRAFT ==
    {draft}
    == THE FEEDBACK ==
    {extract_feedback(feedback)}
    """

# PROMPTS
evaluator_system_prompt = """
You are an evaluator. Respond with two distinct sections, clearly marked by the headers **THINKING** and **FEEDBACK**.

1. **THINKING**: Analyze the response based on the given criteria. Highlight key observations, strengths, and weaknesses. Keep this section focused on evaluation only—do not include suggestions.

2. **FEEDBACK**: Provide specific, actionable steps to improve the response. Focus on practical adjustments to better meet the criteria. Avoid repeating analysis here.

Structure your response exactly as shown, using Markdown format:
### THINKING
[Your analysis here]

### FEEDBACK
[Your actionable feedback here]
"""

default_role_definition = (
    "You are an expert joke teller. "
    "Respond in markdown."
)

default_task = (
    "Write a clever dad joke that subverts typical dad joke expectations."
)

default_criteria = (
    "The joke must have a clever linguistic twist. "
    "It should utilize brevity to keep the readers on their toes. "
    "The punchline must be unexpected yet logically connected to the setup. "
    "It should sound like something a witty parent might spontaneously blurt out. "
    "The joke must make semantic sense while being humorously absurd. "
)

# BACKEND
def stream_generation(system_prompt, user_prompt, prev_draft=None, prev_feedback=None):
    """
    Streams a response from the model for a given system and user prompt.
    """
    
    conversation = [{"role": "system", "content": system_prompt}]
    conversation.append({"role": "user", "content": user_prompt})
    
    if prev_draft and prev_feedback:
        conversation.append({"role": "system", "content": get_feedback_prompt(prev_draft, prev_feedback)})
        print(get_feedback_prompt(prev_draft, prev_feedback))

    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(input_ids=inputs, max_new_tokens=1337, streamer=streamer, temperature=0.6)

    response_buffer = ""
    print("Generating...")
    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        for chunk in streamer:
            response_buffer += chunk
            formatted_response_buffer = clean_response(response_buffer)

            yield formatted_response_buffer

def stream_evaluation(evaluation_criteria, generator_draft):
    """
    Streams a response from the model for a given system and user prompt.
    """
    conversation = [{"role": "system", "content": evaluator_system_prompt}]
    conversation.append({"role": "user", "content": user_prompt_for(evaluation_criteria, generator_draft)})

    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(input_ids=inputs, max_new_tokens=5000, streamer=streamer, temperature=0.6)

    response_buffer = ""
    print("Evaluating...")
    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        for chunk in streamer:
            response_buffer += chunk
            formatted_response_buffer = clean_response(response_buffer)

            yield formatted_response_buffer

@spaces.GPU(duration=120)
def auto_generate(role, task, criteria, max_retries):
    """
    1) Stream content from the generator first
    2) Then stream content from the evaluator
    3) Return final generator & evaluator outputs
    """
    gen_text = ""
    eval_text = ""

    for _ in range(max_retries):
        # First pass: generation streaming
        for token in stream_generation(role, task, gen_text, eval_text):
            gen_text = token
            yield (gen_text, eval_text)
        
        # Second pass: evaluator streaming
        for token in stream_evaluation(criteria, gen_text):
            eval_text = token
            yield (gen_text, eval_text)

    # Utilize last feedback
    for token in stream_generation(role, task, gen_text, eval_text):
        gen_text = token
        yield (gen_text, eval_text)

# UI

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Generator and Evaluator Demo")

    with gr.Row():
        with gr.Column():
            role_input = gr.Textbox(label="Role Definition", placeholder="Enter a role definition..", value=default_role_definition)
            generator_input = gr.Textbox(label="Task Input", placeholder="Enter your task...", value=default_task)
        evaluator_input = gr.Textbox(label="Evaluation Criteria", placeholder="Enter evaluation criteria...", value=default_criteria)

    # the part that should be made functional
    with gr.Row():
        max_retries = gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            value=1,
            label="Max Retries"
        )
        auto_generate_btn = gr.Button("Auto-Generate & Evaluate")
        
    with gr.Row():
        clear_button = gr.Button("Clear")

    with gr.Row():
        generator_output = gr.Markdown(label="Generator Response")
        evaluator_output = gr.Markdown(label="Evaluator Response")
        
    auto_generate_btn.click(
        auto_generate,
        inputs=[role_input, generator_input, evaluator_input, max_retries],
        outputs=[generator_output, evaluator_output],
        scroll_to_output=True
    )

    # Clear button
    clear_button.click(
        lambda: ("", "", "", ""),
        inputs=[],
        outputs=[generator_input, evaluator_input, generator_output, evaluator_output],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)