# How AI Actually "Reads" — A Visual Tour of 6 LLM Tokenizers

**By Arvind Chary Padala** · *March 2026*

---

Here's something nobody tells you when you first start working with AI: **language models don't read words. They don't even read characters.** They read *tokens* — fragments of text that exist somewhere between a letter and a word. And the way those tokens are created? That's what separates GPT-4o from BERT from LLaMA in ways that matter far more than most people realize.

I built a tool — the **Tokenizer Visualizer** — that lets you type any sentence and watch six of the most important AI tokenizers in history slice it up in real time. The results are sometimes intuitive, often surprising, and always revealing. Let's walk through all six.

---

## First: What Even Is a Token?

Before meeting the models, a quick grounding. A tokenizer is the gatekeeper between human language and the mathematical world of neural networks. Your text goes in as a string; it comes out as a list of integers that the model can actually process. The "vocabulary" is the lookup table mapping integers to text fragments.

The challenge: make fragments small enough to handle any word, large enough to be efficient, consistent enough to be reversible. That's a harder problem than it sounds — which is why there are multiple competing solutions.

Let's use a single sentence as our running example throughout:

> **"The future of AI is incredibly exciting!"**

Seven words. What happens to it depends entirely on who's reading.

---

## 🤖 GPT-2 (2019) — The Ancestor

**Algorithm:** Byte-Pair Encoding (BPE) · **Vocabulary:** 50,257 tokens

GPT-2 is the grandfather of modern language models. Released by OpenAI in 2019, it proved that a model trained on enough text (40GB of Reddit links) could write coherent paragraphs, summarize articles, and even code — all with zero task-specific training.

Its tokenizer uses **Byte-Pair Encoding**: it starts with individual characters and iteratively merges the most frequent adjacent pairs until it has ~50k tokens. Common words become single tokens; rarer words get split.

**Our sentence through GPT-2's eyes:**
```
[The] [Ġfuture] [Ġof] [ĠAI] [Ġis] [Ġincredibly] [Ġexciting] [!]
→ 8 tokens
```

Notice the `Ġ` prefix? That's GPT-2's way of encoding the space *before* a word into the token itself. `"future"` and `"Ġfuture"` are different entries in the vocabulary. Weird? Yes. Effective? Also yes.

**Today's standing (2026):** GPT-2 itself is retired at the frontier, but its tokenizer is the direct ancestor of GPT-4. Think of it as the Wright Flyer of LLM tokenizers — no longer flying commercially, but everything that came after owes it something.

---

## ✨ GPT-4o (2024) — The Current King

**Algorithm:** tiktoken BPE (`o200k_base`) · **Vocabulary:** 200,276 tokens

Zoom forward five years. GPT-4o's tokenizer has **4× the vocabulary** of GPT-2 and is written in Rust, making it 10–15× faster. The `o200k_base` encoding used by GPT-4o was specifically designed to tokenize efficiently across *all* human languages — not just English.

**Our sentence:**
```
[The] [future] [of] [AI] [is] [incredibly] [exciting] [!]
→ 8 tokens
```

Same count as GPT-2 on this sentence — because these are all common English words. But type something in Hindi, Arabic, or Japanese and the gap becomes enormous. GPT-4o's 200k vocabulary tokenizes non-English text with **30–60% fewer tokens** than its predecessor, which translates directly into lower API costs and longer effective context for non-English users.

**Today's standing:** This is the tokenizer powering the world's most-used AI products right now. Every developer using the OpenAI API is interacting with this tokenizer on every single call. Understanding it is not optional — it determines your bill.

---

## 🔵 BERT (2018) — The Encoder That Changed Everything

**Algorithm:** WordPiece · **Vocabulary:** 30,522 tokens

BERT arrived in 2018 from Google and flipped the script on how the industry thought about NLP. Instead of reading text left to right like GPT, BERT reads **both directions simultaneously** — it sees the whole sentence at once. This bidirectionality made it the best model for understanding tasks: search, classification, question answering, named entity recognition.

Its tokenizer — WordPiece — has a different philosophy from BPE. Instead of merging the most *frequent* pairs, it merges pairs that maximize the statistical likelihood of the training data. It also lowercases everything and uses a `##` prefix to mark continuation subwords.

**Our sentence:**
```
[[CLS]] [the] [future] [of] [ai] [is] [incredibly] [exciting] [!] [[SEP]]
→ 10 tokens
```

Two things jump out: `"The"` became `"the"` and `"AI"` became `"ai"` (lowercase), and two special tokens appeared that weren't in the original text. `[CLS]` is BERT's global sentence anchor — its final hidden state is what you feed to a classifier. `[SEP]` marks the end of the sequence.

**Today's standing:** BERT is behind more production AI than most people realize. Every vector database, semantic search engine, and document retrieval system you interact with daily is almost certainly running a BERT-family model under the hood. `all-MiniLM-L6-v2`, the model powering most embedding pipelines in 2026, is BERT's direct descendant.

---

## 🟢 T5 (2019) — The Unifier

**Algorithm:** SentencePiece (Unigram LM) · **Vocabulary:** 32,100 tokens

Google's T5 had a radical idea: what if every NLP task — translation, summarization, classification, Q&A — was just *text going in and text coming out*? No separate heads, no architectural changes. Just one model, trained on everything, prompted with a task prefix.

T5's SentencePiece tokenizer reflects this universality. Unlike BPE, SentencePiece makes **no whitespace assumptions** — it treats the entire text as a raw character stream and encodes spaces as a visible `▁` character. This makes it natively language-agnostic.

There's one quirk: T5 needs to know what task you're asking it to do, so the tokenizer receives a prefixed version of your sentence.

**Our sentence (with required prefix):**
```
Input sent to T5: "summarize: The future of AI is incredibly exciting!"
[▁summarize] [:] [▁The] [▁future] [▁of] [▁AI] [▁is] [▁incredibly] [▁exciting] [!] [</s>]
→ 12 tokens (including EOS)
```

The `▁` marks word boundaries. The `</s>` end-of-sequence token tells the decoder to stop generating. If you're wondering why our Visualizer automatically prepends `"summarize: "` — that's not a bug; it's required for the model to function correctly.

**Today's standing:** T5 variants power lightweight production pipelines everywhere. When a startup says "we run NLP in the cloud affordably," there's a good chance Flan-T5 is involved. It remains the best choice when you need a capable seq2seq model without GPT-4 pricing.

---

## 🦙 LLaMA (2023–2025) — The Open-Source Backbone

**Algorithm:** SentencePiece BPE · **Vocabulary:** 32,000 tokens

Meta's LLaMA models are the backbone of the entire open-source AI ecosystem. Mistral, Vicuna, Alpaca, CodeLlama, Ollama's default models — they all trace back to LLaMA. Understanding LLaMA's tokenizer means understanding 80% of self-hosted AI deployments in 2026.

**Our sentence:**
```
[<s>] [▁The] [▁future] [▁of] [▁A] [I] [▁is] [▁incred] [ibly] [▁exc] [iting] [!]
→ 12 tokens
```

Now *this* is interesting. `"AI"` splits into `[▁A]` + `[I]`. `"incredibly"` becomes `[▁incred]` + `[ibly]`. `"exciting"` becomes `[▁exc]` + `[iting]`. That's 12 tokens for a sentence that GPT-4o handles in 8.

Why? LLaMA's training corpus — while massive — didn't make these specific word forms frequent enough for them to earn single-token status. This 50% overhead is not trivial: it means LLaMA models hit their context limit faster on the same content, and every inference step costs more.

**Meta heard the feedback.** LLaMA 3 (2024) expanded the vocabulary to **128,000 tokens**, dramatically closing the efficiency gap with GPT-4o.

**Today's standing:** If you're running local AI — Ollama, llama.cpp, vLLM — you're almost certainly using a LLaMA tokenizer. It's the Linux of AI tokenizers: not always the most elegant, but everywhere.

---

## 🔷 Phi-2 (2023) — The Proof That Size Isn't Everything

**Algorithm:** BPE · **Vocabulary:** 51,200 tokens

In December 2023, Microsoft Research published something that made the whole ML community pause: a **2.7 billion parameter model** that outperformed models 10× its size on math, coding, and reasoning benchmarks. No giant compute clusters, no trillion-token training set. Just very, very carefully selected training data.

The Phi-2 tokenizer tells the same story — lean and purposeful.

**Our sentence:**
```
[The] [future] [of] [AI] [is] [incredibly] [exciting] [!]
→ 8 tokens
```

Identical to GPT-4o. On clean English prose, STEM content, and code — Phi-2's training domains — it achieves the same tokenization efficiency as a model with a vocabulary 4× larger. Curation beats scale.

**Today's standing:** Phi-2 launched an era of "small but mighty" models. Its successors — Phi-3, Phi-3.5, and Phi-4 (2025) — continue pushing the boundary of what's achievable on a laptop. In a world increasingly focused on edge AI and on-device inference, this lineage matters more every year.

---

## Putting It All Together

Here's our benchmark sentence through all six tokenizers at once:

| Tokenizer | Tokens | Tokens/Word | Notable |
|-----------|--------|-------------|---------|
| GPT-2 | 8 | 1.14 | `Ġ` space-prefix convention |
| GPT-4o | 8 | 1.14 | 200k vocab, fastest inference |
| BERT | 10 | 1.43 | `[CLS]`/`[SEP]`, lowercased |
| T5 | 12 | 1.71 | Task prefix + `</s>` required |
| LLaMA | 12 | 1.71 | Fragments "AI", "incredibly" |
| Phi-2 | 8 | 1.14 | Matches GPT-4o on clean prose |

The winner on efficiency: GPT-4o, GPT-2, and Phi-2 (tied). But "efficiency on English prose" is only one dimension. BERT wins on semantic search tasks. T5 wins on structured seq2seq work. LLaMA wins on open-source deployability. GPT-4o wins on raw capability and multilinguality.

The right tokenizer isn't the one with the best number in a table — it's the one whose tradeoffs align with your problem.

---

## Why Any of This Matters to You

If you're a developer calling an AI API, your token count is your invoice. A tokenizer that splits your content into 30% more tokens than necessary is a hidden tax on every request.

If you're building a RAG system, your tokenizer determines how much knowledge fits in your context window. A more efficient tokenizer means you can retrieve more documents and get better answers.

If you're fine-tuning a model, mismatched tokenizer and model weights will produce garbage silently — no error, just bad outputs. This is the most common beginner mistake in production LLM engineering.

And if you're just curious? **There is no better way to build intuition for how LLMs work than watching them tokenize text.** Before attention, before softmax, before any math — there's a tokenizer turning your words into numbers. That's where it all begins.

---

*The Tokenizer Visualizer is an open-source Streamlit app supporting GPT-2, GPT-4o, BERT, T5, LLaMA, and Phi-2.*
*Connect: [LinkedIn](https://www.linkedin.com/in/arvindcharypadala/) · [GitHub](https://github.com/ArvindPadala)*
