# custom_tokenizers/bert_tokenizer.py
from transformers import AutoTokenizer
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_bert_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize_with_bert(text):
    try:
        tokenizer = _load_bert_tokenizer()
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
