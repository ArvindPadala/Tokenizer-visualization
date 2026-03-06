# custom_tokenizers/gemma_tokenizer.py
# Uses microsoft/phi-2 — Microsoft's Phi-2 (2023-2024), fully open and ungated on HF.
# Phi-2 uses a custom BPE tokenizer with a 51k vocabulary, from the efficient small-model family
# that achieves strong reasoning despite its size (2.7B params).
from transformers import AutoTokenizer
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_gemma_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


def tokenize_with_gemma(text):
    try:
        tokenizer = _load_gemma_tokenizer()
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
