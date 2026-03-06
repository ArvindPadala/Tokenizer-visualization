# custom_tokenizers/load_tokenizer_outputs.py
from custom_tokenizers.gpt2_tokenizer import tokenize_with_gpt2
from custom_tokenizers.gpt4_tokenizer import tokenize_with_gpt4
from custom_tokenizers.bert_tokenizer import tokenize_with_bert
from custom_tokenizers.t5_tokenizer import tokenize_with_t5
from custom_tokenizers.llama_tokenizer import tokenize_with_llama
from custom_tokenizers.gemma_tokenizer import tokenize_with_gemma

# Registry: adding a new tokenizer = one entry here
TOKENIZER_MAP = {
    "GPT-2 (BPE)":           tokenize_with_gpt2,
    "GPT-4o (tiktoken)":     tokenize_with_gpt4,
    "BERT (WordPiece)":      tokenize_with_bert,
    "T5 (SentencePiece)":    tokenize_with_t5,
    "LLaMA (SentencePiece)": tokenize_with_llama,
    "Phi-2 (BPE)":           tokenize_with_gemma,
}


def load_tokenizer_outputs(text, selected_tokenizers):
    results = {}
    for name in selected_tokenizers:
        fn = TOKENIZER_MAP.get(name)
        if fn:
            results[name] = fn(text)
    return results