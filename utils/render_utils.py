# tokenizer_visualizer/utils/render_utils.py
import streamlit as st

def colorize_tokens(tokens):
    colored = []
    colors = ["#e8f0fe", "#d1e7dd", "#fce5cd", "#f8d7da"]
    for i, token in enumerate(tokens):
        color = colors[i % len(colors)]
        colored.append(f'<span style="background-color: {color}; padding: 3px; margin: 2px; border-radius: 4px;">{token}</span>')
    return " ".join(colored)

def display_colored_tokens(tokens):
    html = colorize_tokens(tokens)
    st.markdown(html, unsafe_allow_html=True)
