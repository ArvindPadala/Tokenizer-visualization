# tokenizers/load_tokenizer_outputs.py
from custom_tokenizers.gpt2_tokenizer import tokenize_with_gpt2
from custom_tokenizers.bert_tokenizer import tokenize_with_bert
from custom_tokenizers.t5_tokenizer import tokenize_with_t5

def load_tokenizer_outputs(text, selected_tokenizers):
    results = {}
    for name in selected_tokenizers:
        if name == "GPT-2 (BPE)":
            results[name] = tokenize_with_gpt2(text)
        elif name == "BERT (WordPiece)":
            results[name] = tokenize_with_bert(text)
        elif name == "T5 (SentencePiece)":
            results[name] = tokenize_with_t5(text)
    return results