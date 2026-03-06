import streamlit as st
import pandas as pd
from custom_tokenizers.load_tokenizer_outputs import load_tokenizer_outputs, TOKENIZER_MAP
from utils.render_utils import display_colored_tokens, display_legend

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tokenizer Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — white sleek theme ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* White background */
.stApp {
    background: #f8f9fb;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(120deg, #4f46e5 0%, #6366f1 55%, #0ea5e9 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.18);
}
.hero-banner h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.hero-banner p {
    color: rgba(255,255,255,0.88);
    font-size: 0.97rem;
    margin: 0;
}

/* Tokenizer cards */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover {
    border-color: #a5b4fc;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.10);
}
.card-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Section labels */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #9ca3af;
    margin: 1rem 0 0.4rem 0;
}

/* Metric overrides */
[data-testid="metric-container"] {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e7eb !important;
}
section[data-testid="stSidebar"] * {
    color: #374151 !important;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #111827 !important;
}

/* Info / warning banners */
.stAlert {
    border-radius: 10px;
    font-size: 0.88rem;
}

/* Code blocks */
.stCodeBlock, code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background: #f3f4f6 !important;
    border-radius: 8px;
}

/* Dataframe */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
}

/* Inputs */
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1px solid #d1d5db !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* Multiselect */
.stMultiSelect [data-baseweb="tag"] {
    border-radius: 20px !important;
}

/* Divider */
hr {
    border-color: #e5e7eb;
}

/* General text */
p, li, span { color: #374151; }
h1, h2, h3 { color: #111827; letter-spacing: -0.01em; }
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🔬 Tokenizer Visualizer</h1>
  <p>Compare how 6 modern LLM tokenizers split your text — from BERT (2018) to GPT-4o (2024) and beyond.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
TOKENIZER_EMOJIS = {
    "GPT-2 (BPE)":           "🤖",
    "GPT-4o (tiktoken)":     "✨",
    "BERT (WordPiece)":      "🔵",
    "T5 (SentencePiece)":   "🟢",
    "LLaMA (SentencePiece)":"🦙",
    "Phi-2 (BPE)": "🔷",
}

TOKENIZER_DESCRIPTIONS = {
    "GPT-2 (BPE)":           "OpenAI (2019) · Byte-Pair Encoding · 50k vocab",
    "GPT-4o (tiktoken)":     "OpenAI (2024) · tiktoken o200k_base · 200k vocab",
    "BERT (WordPiece)":      "Google (2018) · WordPiece · 30k vocab",
    "T5 (SentencePiece)":   "Google (2019) · SentencePiece · 32k vocab",
    "LLaMA (SentencePiece)":"Meta (2023-25) · SentencePiece BPE · 32k vocab",
    "Phi-2 (BPE)": "Microsoft (2023) · BPE · 51k vocab",
}

TOKENIZER_TYPE_HINT = {
    "GPT-2 (BPE)":           "gpt2",
    "GPT-4o (tiktoken)":     "gpt4",
    "BERT (WordPiece)":      "bert",
    "T5 (SentencePiece)":   "t5",
    "LLaMA (SentencePiece)":"llama",
    "Phi-2 (BPE)":           "gemma",
}

all_options = list(TOKENIZER_MAP.keys())

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    text = st.text_area(
        "📝 Input Text",
        value="The future of AI is incredibly exciting!",
        height=130,
        help="Type any sentence to tokenize.",
    )

    st.markdown("### 🔬 Select Tokenizers")
    selected_tokenizers = st.multiselect(
        "Choose one or more:",
        all_options,
        default=all_options,
        format_func=lambda x: f"{TOKENIZER_EMOJIS.get(x, '')} {x}",
    )

    st.markdown("---")
    st.markdown("### 🎨 Legend")
    st.markdown(
        '<span style="background:#6366f1;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.77rem;">Special</span>'
        '&nbsp;'
        '<span style="background:#0ea5e9;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.77rem;">Subword</span>'
        '&nbsp;'
        '<span style="background:#10b981;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.77rem;">Whole word</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.caption("Hover over a pill to see the raw token string.")
    st.markdown("---")
    st.caption("Built with 🤗 Transformers · tiktoken · Streamlit")

# ── Main area ─────────────────────────────────────────────────────────────────
if not text:
    st.info("Enter some text in the sidebar to get started.")
    st.stop()

if not selected_tokenizers:
    st.warning("Please select at least one tokenizer from the sidebar.")
    st.stop()

with st.spinner("Running tokenizers… (first run downloads model weights)"):
    results = load_tokenizer_outputs(text, selected_tokenizers)

# ── Per-tokenizer cards ───────────────────────────────────────────────────────
for name in selected_tokenizers:
    output = results.get(name, {})

    emoji   = TOKENIZER_EMOJIS.get(name, "🔢")
    desc    = TOKENIZER_DESCRIPTIONS.get(name, "")
    type_hint = TOKENIZER_TYPE_HINT.get(name, "generic")

    st.markdown(f'<div class="card">', unsafe_allow_html=True)

    # Header
    st.markdown(
        f'<div class="card-header">{emoji} {name}'
        f'<span style="font-size:0.72rem;font-weight:400;color:#94a3b8;margin-left:8px;">{desc}</span></div>',
        unsafe_allow_html=True,
    )

    if "error" in output:
        st.error(f"❌ Failed to load tokenizer: {output['error']}")
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    tokens         = output["tokens"]
    ids            = output["ids"]
    attention_mask = output["attention_mask"]
    prefix_added   = output.get("prefix_added", False)

    # T5 prefix banner
    if prefix_added:
        st.info('ℹ️ T5 requires a task prefix. `"summarize: "` was automatically prepended to your text — those extra tokens appear at the start.')

    # Metrics row
    special_tokens_set = {
        "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]",
        "<s>", "</s>", "<pad>", "<unk>", "<mask>",
    }
    n_special = sum(1 for t in tokens if t in special_tokens_set or (t.startswith("<") and t.endswith(">")))
    n_unique  = len(set(tokens))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Tokens",   len(tokens))
    m2.metric("Unique Tokens",  n_unique)
    m3.metric("Special Tokens", n_special)
    m4.metric("Vocab Size",     desc.split("·")[-1].strip() if desc else "—")

    # Colorized token pills
    st.markdown('<div class="section-label">Token Pills</div>', unsafe_allow_html=True)
    display_colored_tokens(tokens, type_hint)

    # Token IDs + Attention Mask (collapsible)
    with st.expander("📊 Raw IDs & Attention Mask"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Token IDs</div>', unsafe_allow_html=True)
            st.code(ids, language="text")
        with c2:
            st.markdown('<div class="section-label">Attention Mask</div>', unsafe_allow_html=True)
            st.code(attention_mask, language="text")

    st.markdown("</div>", unsafe_allow_html=True)

# ── Comparison Table ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📊 Side-by-Side Comparison")

rows = []
for name in selected_tokenizers:
    output = results.get(name, {})
    if "error" in output:
        rows.append({
            "Tokenizer": name,
            "Family": "—",
            "Vocab Size": "—",
            "Total Tokens": "Error",
            "Unique Tokens": "—",
            "Special Tokens": "—",
            "Tokens / Word (approx)": "—",
        })
        continue

    tokens = output["tokens"]
    special_count = sum(
        1 for t in tokens
        if t in {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]", "<s>", "</s>", "<pad>", "<unk>", "<mask>"}
        or (t.startswith("<") and t.endswith(">"))
    )
    word_count = len(text.split())
    tpw = round(len(tokens) / word_count, 2) if word_count else "—"
    desc = TOKENIZER_DESCRIPTIONS.get(name, "")
    family = desc.split("·")[0].strip() if desc else name

    rows.append({
        "Tokenizer": f"{TOKENIZER_EMOJIS.get(name, '')} {name}",
        "Family": family,
        "Vocab Size": desc.split("·")[-1].strip() if desc else "—",
        "Total Tokens": len(tokens),
        "Unique Tokens": len(set(tokens)),
        "Special Tokens": special_count,
        "Tokens / Word": tpw,
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🔬 **Tokenizer Visualizer** · Supporting GPT-2, GPT-4o, BERT, T5, LLaMA, and Gemma · "
    "Powered by 🤗 Hugging Face Transformers & [tiktoken](https://github.com/openai/tiktoken) · "
    "[GitHub](https://github.com/ArvindPadala)"
)
