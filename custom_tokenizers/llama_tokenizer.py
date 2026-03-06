# custom_tokenizers/llama_tokenizer.py
# Uses huggyllama/llama-7b — freely available HF tokenizer representing
# the LLaMA/Mistral/Vicuna family that dominates open-source LLMs in 2024-2026
from transformers import AutoTokenizer
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_llama_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")


def tokenize_with_llama(text):
    try:
        tokenizer = _load_llama_tokenizer()
        encoding = tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return {
            "tokens": tokenizer.convert_ids_to_tokens(encoding.input_ids[0]),
            "ids": encoding.input_ids[0].tolist(),
            "attention_mask": encoding.attention_mask[0].tolist(),
            "prefix_added": False,
        }
    except Exception as e:
        return {"error": str(e)}
