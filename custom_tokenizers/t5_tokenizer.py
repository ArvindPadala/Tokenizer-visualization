# custom_tokenizers/t5_tokenizer.py
from transformers import AutoTokenizer
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_t5_tokenizer():
    return AutoTokenizer.from_pretrained("t5-small")


def tokenize_with_t5(text):
    try:
        tokenizer = _load_t5_tokenizer()
        prefix_added = False
        if not text.lower().startswith("summarize: "):
            text = "summarize: " + text
            prefix_added = True

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
            "prefix_added": prefix_added,
        }
    except Exception as e:
        return {"error": str(e)}