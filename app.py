import streamlit as st
from custom_tokenizers.load_tokenizer_outputs import load_tokenizer_outputs

st.set_page_config(page_title="Tokenizer Visualizer", layout="wide")
st.title("🔍 Tokenizer Visualizer")

# Input text
text = st.text_area("Enter a sentence to tokenize:", "The future of AI is incredibly exciting!")

# Select models/tokenizers
tokenizer_options = ["GPT-2 (BPE)", "BERT (WordPiece)", "T5 (SentencePiece)"]
selected_tokenizers = st.multiselect("Choose tokenizers to compare:", tokenizer_options, default=tokenizer_options)

# Run when text is entered
if text and selected_tokenizers:
    results = load_tokenizer_outputs(text, selected_tokenizers)

    for tokenizer_name, output in results.items():
        st.subheader(f"🔢 {tokenizer_name}")
        st.markdown("**Tokens:**")
        st.code(" ".join(output["tokens"]), language="text")

        st.markdown("**Token IDs:**")
        st.code(output["ids"], language="text")

        st.markdown("**Attention Mask:**")
        st.code(output["attention_mask"], language="text")

        st.markdown(f"**Token Count:** {len(output['tokens'])}")
        st.markdown("---")
