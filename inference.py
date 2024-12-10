import os
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import torch

def model_fn(model_dir):
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    config = LlamaConfig.from_pretrained(model_name, use_auth_token=hf_token)
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
        use_auth_token=hf_token
    )
    return model, tokenizer

def predict_fn(input_data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    text = input_data.pop("text", "")
    summary_length = input_data.pop("summary_length", "medium")
    summary_style = input_data.pop("summary_style", "default")
    custom_prompt = input_data.pop("custom_prompt", None)

    summary = generate_summary(model, tokenizer, text, summary_length, summary_style, custom_prompt)
    
    if input_data.get("generate_title", False):
        title = generate_title(model, tokenizer, text)
        return {"summary": summary, "title": title}
    
    return {"summary": summary}

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
    prompt = f"Generate a concise title for the following text:\n\n{text}\n\nTitle:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95
        )
    title = tokenizer.decode(output[0], skip_special_tokens=True)
    return title.split("Title:")[1].strip()