# utils/render_utils.py
import streamlit as st

# Palette — distinct colors for each token type (tuned for white backgrounds)
COLORS = {
    "special":  {"bg": "#6366f1", "fg": "#ffffff"},   # indigo  — [CLS], [SEP], <s>, </s>
    "subword":  {"bg": "#0ea5e9", "fg": "#ffffff"},   # sky     — ##ing, Ġword, ▁word
    "whole":    {"bg": "#10b981", "fg": "#ffffff"},   # emerald — regular whole words
    "cycle":    [                                      # fallback cycle (GPT-4o / tiktoken)
        "#6366f1", "#f59e0b", "#10b981", "#ec4899",
        "#3b82f6", "#ef4444", "#0ea5e9", "#84cc16",
    ],
}

# Special token markers by tokenizer family
SPECIAL_TOKENS = {
    "[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]",   # BERT / WordPiece
    "<s>", "</s>", "<pad>", "<unk>", "<mask>",        # LLaMA / Gemma / T5
    "▁",                                               # SentencePiece space marker alone
}


def _classify_token(token: str) -> str:
    """Return 'special', 'subword', or 'whole'."""
    if token in SPECIAL_TOKENS or token.startswith("<") and token.endswith(">"):
        return "special"
    # BPE continuation (GPT-2): starts with Ġ but isn't the first token
    # WordPiece continuation: starts with ##
    # SentencePiece non-initial pieces: don't start with ▁
    if token.startswith("##") or token in {"Ġ"}:
        return "subword"
    return "whole"


def render_token_pills(tokens: list, tokenizer_type: str = "generic") -> str:
    """
    Build an HTML string of colored token pills.
    tokenizer_type hints: 'bert', 'gpt2', 't5', 'llama', 'gemma', 'gpt4'
    """
    parts = []
    for i, token in enumerate(tokens):
        display = token.replace("Ġ", "·").replace("▁", "·").replace("##", "")

        if tokenizer_type == "gpt4":
            # tiktoken tokens don't follow the same pattern → use cycle colors
            c = COLORS["cycle"][i % len(COLORS["cycle"])]
            bg, fg = c, "#ffffff"
        else:
            kind = _classify_token(token)
            bg = COLORS[kind]["bg"]
            fg = COLORS[kind]["fg"]

        parts.append(
            f'<span title="{token}" style="'
            f'background:{bg}; color:{fg}; padding:3px 8px; margin:2px; '
            f'border-radius:12px; font-size:0.85rem; font-family:monospace; '
            f'display:inline-block; cursor:default;">'
            f'{display}</span>'
        )
    return " ".join(parts)


def display_colored_tokens(tokens: list, tokenizer_type: str = "generic"):
    html = render_token_pills(tokens, tokenizer_type)
    st.markdown(html, unsafe_allow_html=True)


def display_legend():
    """Render a small color legend for special / subword / whole tokens."""
    legend_items = [
        (COLORS["special"]["bg"], "Special tokens ([CLS], </s>, …)"),
        (COLORS["subword"]["bg"], "Subword / continuation (##, ▁, Ġ)"),
        (COLORS["whole"]["bg"],   "Whole-word tokens"),
    ]
    legend_html = " &nbsp; ".join(
        f'<span style="background:{c}; color:#fff; padding:2px 10px; '
        f'border-radius:10px; font-size:0.78rem;">{label}</span>'
        for c, label in legend_items
    )
    st.markdown(f"**Legend:** {legend_html}", unsafe_allow_html=True)
