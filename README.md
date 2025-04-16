#  Tokenizer Visualizer

An interactive Streamlit app to explore how different language models tokenize text. This project allows users to input any sentence and visualize how popular LLMs like GPT-2, BERT, and T5 break down the input into tokens, token IDs, and attention masks.

##  Features
- Input any text and choose one or more tokenizers
- Supports GPT-2 (BPE), BERT (WordPiece), and T5 (SentencePiece)
- Shows:
  - Tokens
  - Token IDs
  - Attention Mask
  - Token count

##  Technologies Used
- Python
- Streamlit
- Hugging Face Transformers

##  Installation
```bash
git clone https://github.com/yourusername/tokenizer-visualizer.git
cd tokenizer-visualizer
pip install -r requirements.txt
```

##  Running the App
```bash
streamlit run app.py
```

## Example Prompt
Input: `"The future of AI is incredibly exciting!"`

You’ll see how:
- GPT-2 splits words with space-tokens (e.g., `Ġfuture`)
- BERT splits rare words into subwords (e.g., `un##believable`)
- T5 uses underscore-based pieces (e.g., `▁The`, `▁future`)

##  Learning Outcomes
- Understand how different tokenizers interpret input
- Explore token fragmentation and input efficiency
- Visualize model preprocessing steps

##  Contact
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/arvindcharypadala/) or [GitHub](https://github.com/ArvindPadala?tab=repositories)

---

