# tokenizers/t5_tokenizer.py
from transformers import AutoTokenizer

def tokenize_with_t5(text):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # T5 uses "text-to-text" format, typically adds a task prefix
    if not text.lower().startswith("summarize: "):
        text = "summarize: " + text  # default task for demo

    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_attention_mask=True,
        add_special_tokens=True
    )

    return {
        "tokens": tokenizer.convert_ids_to_tokens(encoding.input_ids[0]),
        "ids": encoding.input_ids[0].tolist(),
        "attention_mask": encoding.attention_mask[0].tolist()
    }