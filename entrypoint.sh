#!/bin/bash
# Set Hugging Face token from environment variable
export HF_TOKEN=${HF_TOKEN:-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} 

# Start the Streamlit app
streamlit run app.py