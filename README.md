# 🔬 Tokenizer Visualizer

An interactive **Streamlit** web app for comparing how **6 modern LLM tokenizers** split text — side by side, with colorized token pills, per-tokenizer metrics, and a comparison table. Spans the full history of tokenization from BERT (2018) to GPT-4o (2024).

---

## ✨ Features

- 📝 **Type any sentence** and instantly see how each tokenizer breaks it down
- 🔬 **6 tokenizers** across all major families and eras
- 🎨 **Colorized token pills** — color-coded by type (whole word, subword, special token)
- 📊 **Per-tokenizer metrics** — Total Tokens, Unique Tokens, Special Tokens, Vocab Size
- 📋 **Side-by-side comparison table** — Tokens/Word ratio and stats across all selected tokenizers
- ℹ️ **T5 prefix transparency** — notifies you when `"summarize: "` is auto-prepended
- ⚡ **Cached model loading** — models load once per session, no lag on reruns
- 🛡️ **Graceful error handling** — failed tokenizers show an inline error, never crash the app

---

## 🧠 Tokenizers Supported

| Tokenizer | Model | Algorithm | Vocab Size | Era |
|-----------|-------|-----------|------------|-----|
| 🤖 GPT-2 (BPE) | `gpt2` | Byte-Pair Encoding | 50k | 2019 |
| ✨ GPT-4o (tiktoken) | `o200k_base` | tiktoken BPE | 200k | 2024 |
| 🔵 BERT (WordPiece) | `bert-base-uncased` | WordPiece | 30k | 2018 |
| 🟢 T5 (SentencePiece) | `t5-small` | SentencePiece | 32k | 2019 |
| 🦙 LLaMA (SentencePiece) | `huggyllama/llama-7b` | SentencePiece BPE | 32k | 2023–25 |
| 🔷 Phi-2 (BPE) | `microsoft/phi-2` | BPE | 51k | 2023 |

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Streamlit ≥ 1.35** — interactive web UI
- **Hugging Face Transformers ≥ 4.40** — BERT, GPT-2, T5, LLaMA, Phi-2 tokenizers
- **tiktoken ≥ 0.7** — GPT-4o tokenizer (OpenAI)
- **PyTorch ≥ 2.0** — tensor backend for HF tokenizers
- **sentencepiece ≥ 0.1.99** — SentencePiece model support

---

## 📁 Project Structure

```
Tokenizer-visualization/
├── app.py                              ← Streamlit entry point (UI)
├── requirements.txt                    ← Dependencies
├── custom_tokenizers/
│   ├── load_tokenizer_outputs.py       ← Registry dispatcher
│   ├── gpt2_tokenizer.py
│   ├── gpt4_tokenizer.py               ← tiktoken (GPT-4o)
│   ├── bert_tokenizer.py
│   ├── t5_tokenizer.py
│   ├── llama_tokenizer.py
│   └── gemma_tokenizer.py              ← Phi-2 (microsoft/phi-2)
└── utils/
    └── render_utils.py                 ← Token pill renderer & color classifier
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ArvindPadala/Tokenizer-visualization.git
cd Tokenizer-visualization
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Navigate to **http://localhost:8501** in your browser. Models are downloaded automatically from Hugging Face on first run and cached locally.

---

## 🎓 What You'll Learn

- How **BPE** (GPT-2, GPT-4o, Phi-2) handles unknown words via byte merges
- How **WordPiece** (BERT) uses `##` continuations for subwords
- How **SentencePiece** (T5, LLaMA) uses `▁` to mark word boundaries
- Why **vocabulary size matters** — GPT-4o's 200k vocab tokenizes far more efficiently than BERT's 30k
- How the same sentence produces **different token counts** across models, impacting context length and cost
- What **special tokens** (`[CLS]`, `[SEP]`, `<s>`, `</s>`) each model adds automatically

---

## 📬 Contact

Connect on [LinkedIn](https://www.linkedin.com/in/arvindcharypadala/) · [GitHub](https://github.com/ArvindPadala?tab=repositories)

---
