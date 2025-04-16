from transformers import AutoTokenizer

def tokenize_with_gpt2(text):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
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
