import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import torch
from huggingface_hub import login
from dotenv import load_dotenv
from PIL import Image
import os

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

@st.cache_resource
def load_model_and_tokenizer(model_name):
    if not HF_TOKEN:
        raise ValueError("Hugging Face token not found in environment variables!")
    
    login(token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    config = LlamaConfig.from_pretrained(model_name, use_auth_token=True)
    config.rope_scaling = {
        "type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=True
    )
    return model, tokenizer

def generate_summary(model, tokenizer, text, summary_length, summary_style=None, custom_prompt=None):
    length_prompts = {
        "short": "Provide a brief summary in 2-3 sentences",
        "medium": "Provide a concise summary in 4-5 sentences",
        "long": "Provide a detailed summary in 6-8 sentences"
    }
    style_prompts = {
        "academic": "Write an academic summary",
        "news": "Write a news-style summary",
        "default": "Summarize the following text",
        "custom": custom_prompt
    }
    
    if summary_style and summary_style != "default":
        if isinstance(summary_length, int):
            prompt = f"{style_prompts[summary_style]} in approximately {summary_length} words:\n\n{text}\n\nSummary:"
            max_new_tokens = summary_length * 2
            min_new_tokens = summary_length
        else:
            prompt = f"{style_prompts[summary_style]} {length_prompts[summary_length]}:\n\n{text}\n\nSummary:"
            max_new_tokens = {"short": 100, "medium": 200, "long": 300}[summary_length]
            min_new_tokens = {"short": 50, "medium": 100, "long": 150}[summary_length]
    else:
        prompt = f"{style_prompts['default']} {length_prompts[summary_length]}:\n\n{text}\n\nSummary:"
        max_new_tokens = {"short": 100, "medium": 200, "long": 300}[summary_length]
        min_new_tokens = {"short": 50, "medium": 100, "long": 150}[summary_length]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            no_repeat_ngram_size=3,
            do_sample=True,
            early_stopping=True
        )
    
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    summary = summary.split("Summary:")[1].strip()
    
    # Ensure the summary ends with a complete sentence
    sentences = summary.split('.')
    if len(sentences) > 1:
        summary = '.'.join(sentences[:-1]) + '.'
    return summary

def post_process_summary(summary, target_length):
    words = summary.split()
    if len(words) > target_length:
        words = words[:target_length]
        summary = ' '.join(words)
        # Ensure the last sentence is complete
        last_period = summary.rfind('.')
        if last_period != -1:
            summary = summary[:last_period+1]
    return summary

def generate_title(model, tokenizer, text):
    prompt = f"Generate a short, precise title (maximum 10 words) for this text:\n\n{text}\n\nTitle:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=15,
            num_return_sequences=1,
            temperature=0.3,
            top_p=0.9,
            no_repeat_ngram_size=3,
            do_sample=True
        )
    
    title = tokenizer.decode(output[0], skip_special_tokens=True)
    return title.split("Title:")[1].strip()

# Set page configuration
st.set_page_config(page_title="Kreativespace Text Summarizer", layout="wide")

# Create a container for the header
header = st.container()

# Inside the header container, create two columns
col1, col2 = header.columns([1, 4])

# In the first column, display the logo
logo_path = "logo.png"
try:
    logo = Image.open(logo_path)
    col1.image(logo, width=250)
except FileNotFoundError:
    col1.write("Logo not found")

# In the second column, display the title
col2.title("Kreativespace Text Summarizer")

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.2-3B"
try:
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Text input
text_to_summarize = st.text_area("Enter the text you want to summarize:", height=200)

# Summary style selection
summary_style = st.selectbox("Choose summary style:", ["default", "academic", "news", "custom"])

# Custom prompt handling
custom_prompt = None
if summary_style == "custom":
    custom_prompt = st.text_input("Enter your custom summary prompt:",
                                 placeholder="E.g., Make it academic, Give summary in reference of another character, etc.")
    st.text("Example prompts:")
    st.text("1. Make it academic")
    st.text("2. Give summary in reference of Harry Potter")
    st.text("3. Summarize as if explaining to a 5-year-old")

# Length selection
if summary_style != "default":
    customize_length = st.checkbox("Customize summary length?")
    if customize_length:
        summary_length = st.number_input("Enter the desired summary length (in words):", min_value=1, value=100)
    else:
        summary_length = st.selectbox("Choose summary length:", ["short", "medium", "long"])
else:
    summary_length = st.selectbox("Choose summary length:", ["short", "medium", "long"])

# Title generation option
generate_title_option = st.checkbox("Generate title")

# Generate button and output
if st.button("Generate Summary"):
    if text_to_summarize:
        with st.spinner("Generating summary..."):
            if summary_style != "default" and isinstance(summary_length, float):
                summary_length = int(summary_length)
            summary = generate_summary(model, tokenizer, text_to_summarize, summary_length, summary_style, custom_prompt)
            if isinstance(summary_length, int):
                summary = post_process_summary(summary, summary_length)
        
        if generate_title_option:
            with st.spinner("Generating title..."):
                title = generate_title(model, tokenizer, text_to_summarize)
            st.subheader("Generated Title:")
            st.write(title)
        
        st.subheader("Generated Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")