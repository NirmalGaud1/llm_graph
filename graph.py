import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
import streamlit as st

# Set up the Streamlit app
st.title("Gemma 3 Chatbot")
st.write("Interact with Google's Gemma 3 model for text generation.")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/gemma-3-4b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Gemma3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

# Input prompt from the user
user_input = st.text_area("Enter your prompt:", "Explain quantum computing in simple terms.")

# System prompt (optional)
system_prompt = st.text_input("System Prompt (optional):", "You are a helpful assistant.")

# Generate response
if st.button("Generate Response"):
    # Define the input prompt
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_input}]
        }
    ]

    # Apply the chat template and tokenize
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)

    # Generate text
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=200, do_sample=False)

    # Decode and display the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("### Response:")
    st.write(response)
