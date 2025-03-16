import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Set up the Streamlit app
st.title("Gemma 3 Multimodal Chatbot")
st.write("Upload an image and ask a question about it!")

# Load the model and processor
@st.cache_resource
def load_model():
    model_name = "google/gemma-3-4b-it"  # Replace with the correct model name
    processor = AutoProcessor.from_pretrained(model_name)
    model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    return processor, model

processor, model = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or enter an image URL:")

# Text input
user_input = st.text_area("Enter your question:", "What animal is on the candy?")

# Generate response
if st.button("Generate Response"):
    if uploaded_file or image_url:
        # Load the image
        if uploaded_file:
            image = Image.open(uploaded_file)
        else:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))

        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Define the input (image + text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_input}
                ]
            }
        ]

        # Process the input
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)

        # Generate text
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        # Decode and display the output
        response = processor.decode(outputs[0], skip_special_tokens=True)
        st.write("### Response:")
        st.write(response)
    else:
        st.error("Please upload an image or provide an image URL.")
