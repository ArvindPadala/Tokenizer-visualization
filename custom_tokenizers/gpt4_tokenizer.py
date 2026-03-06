# custom_tokenizers/gpt4_tokenizer.py
# Uses tiktoken — the same tokenizer powering GPT-4o (o200k_base vocabulary, ~200k tokens)
import tiktoken
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_gpt4_tokenizer():
    return tiktoken.get_encoding("o200k_base")


def tokenize_with_gpt4(text):
    try:
        enc = _load_gpt4_tokenizer()
        ids = enc.encode(text)
        # tiktoken returns bytes; decode each token to a readable string
        tokens = [enc.decode_single_token_bytes(t).decode("utf-8", errors="replace") for t in ids]
        attention_mask = [1] * len(ids)  # GPT-4 has no padding concept; mask is all 1s
        return {
            "tokens": tokens,
            "ids": ids,
            "attention_mask": attention_mask,
            "prefix_added": False,
        }
    except Exception as e:
        return {"error": str(e)}
