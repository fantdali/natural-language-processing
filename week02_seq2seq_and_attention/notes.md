# Lecture 2: Machine Translation and Attention

**Instructor:** Nikolay Karpachev
**Course:** ML MIPT Advanced
**Date:** 12.02.2024

---

## Overview

- Historical overview of Machine Translation (MT)

  - Statistical MT (SMT)
  - Word alignments

- Neural Machine Translation (NMT)

  - Seq2Seq architecture
  - Beam Search decoding

- Attention mechanism

---

## Historical Overview

### 1950s: Early Experiments

- **Georgetown experiment (1954)** – English–Russian translation of 60 sentences

  - 250 vocabulary entries, 6 grammar rules, IBM 701 mainframe

- **USSR experiment (1954)** – rule-based translation on BESM computer

### 1990–2010: Statistical Machine Translation (SMT)

MT uses **parallel corpora** — pairs of aligned sentences $(x, y)$.

Goal: find the best translation $y$ given a source $x$:

$$
y^* = \arg\max_y P(y|x)
$$

Using Bayes’ rule:

$$
P(y|x) = \frac{P(x|y) P(y)}{P(x)}
$$

- $P(x|y)$ — **translation model**
- $P(y)$ — **language model**

We can ignore $P(x)$ for optimization.

---

### Word Alignment

- **Alignment ($a$):** mapping between words in source and target sentences.
- Types:

  - Many-to-one
  - One-to-many
  - Many-to-many

These alignments estimate which words “correspond” across languages.

---

### Phrase-Based MT (PBMT)

- Translates **phrases (n-grams)** instead of single words.
- Uses:

  - **Phrase table:** bilingual phrase pairs with probabilities.
  - **N-gram language model (LM):** probabilities of word sequences.

**Example of phrase table:**

| Source phrase | Target phrase  | −log P |
| ------------- | -------------- | ------ |
| I saw a cat   | Я видел кошку  | 4.5    |
| I saw a cat   | Я увидел кошку | 9.8    |

**Limitations:**

- Complex, modular systems
- Extensive feature engineering
- Manual maintenance for each language pair

---

## Neural Machine Translation (NMT)

### Core Idea

- Single **end-to-end neural network** that directly models:

$$
P(y|x)
$$

- Uses a **sequence-to-sequence (Seq2Seq)** architecture built with RNNs (or later LSTMs/GRUs).

---

### Seq2Seq Architecture

#### Encoder

Processes source tokens sequentially and produces a final hidden state $h_T$ summarizing the sentence.

#### Decoder

Generates the target sequence token-by-token using:

- Previous token as input
- Hidden state initialized from encoder output
- Probability distribution over vocabulary at each step

$$
P(y_t | y_{<t}, x)
$$

The model is trained end-to-end to minimize **negative log-likelihood**:

$$
\mathcal{L} = -\sum_t \log P(y_t | y_{<t}, x)
$$

---

### Greedy Decoding

At inference time, the decoder picks the most probable token at each step:

$$
y_t = \arg\max P(y_t | y_{<t}, x)
$$

**Problem:** Errors compound — an early mistake affects all later predictions.

---

## Beam Search Decoding

Instead of greedy decoding, maintain the top-$k$ most probable partial translations (“hypotheses”).

- **Beam size $k$:** typically 5–10
- Each hypothesis has a cumulative log-probability score:

$$
s(y_{1:t}) = \sum_{i=1}^t \log P(y_i | y_{<i}, x)
$$

- On each timestep:

  - Expand each hypothesis with all possible next tokens
  - Keep only top $k$ by score

**Stopping criterion:**

- Stop when `<EOS>` (end of sentence) is generated.
- Normalize scores by length to prevent bias toward shorter hypotheses:

$$
\text{score}(y) = \frac{s(y)}{(\text{len}(y))^\alpha}
$$

---

## Evaluation Metrics

### BLEU (Bilingual Evaluation Understudy)

Compares machine translation to one or more reference human translations.

- Based on **n-gram precision** and **brevity penalty** for short outputs.
- Not perfect — penalizes valid paraphrases.

### Other Metrics

- **ROUGE** — recall-oriented metric (common in summarization).
- **METEOR** — uses synonyms and stemming (WordNet).
- **Human evaluation:**

  - **Side-by-side (SbS):** choose best translation.
  - **Direct Assessment (DA):** rate on a 1–5 scale.
  - Expensive → modern alternatives:

    - **BLEURT** (BERT-based, trained on DA ratings)
    - **XCOMET** (XLM-based, trained on DA + MQM)
    - **LLM-based estimators** (prompt LLMs for MT quality)

---

## NMT: Pros & Cons

### ✅ Advantages

- Fluent, context-aware, semantically rich translations
- End-to-end optimization (no modular subcomponents)
- Less manual feature engineering; same method for all language pairs

### ❌ Disadvantages

- Hard to interpret (black box)
- Hard to control (cannot enforce specific translation rules)
- Struggles with:

  - Long context
  - Low-resource languages
  - Domain-specific/slang text
  - Robustness to typos/errors

---

## Attention Mechanism

### Motivation

In basic Seq2Seq, the encoder’s final hidden state must encode the **entire source sentence** — a severe bottleneck.

**Solution:** Let the decoder “attend” to encoder outputs dynamically at each decoding step.

---

### Core Idea

On each decoding step $t$, compute **attention scores** between the decoder hidden state $s_t$ and each encoder hidden state $h_i$:

$$
e_{t,i} = s_t^\top h_i
$$

Normalize with softmax to get the **attention distribution**:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}
$$

Compute the **context vector** as a weighted sum of encoder states:

$$
c_t = \sum_i \alpha_{t,i} h_i
$$

The decoder then uses both $s_t$ and $c_t$ to predict the next token.

---

### Interpretation

- The attention weights $\alpha_{t,i}$ show which source tokens the model focuses on at each output step.
- Gives **alignment interpretability** “for free”.
- First introduced by **Bahdanau et al. (2014)**: _"Neural Machine Translation by Jointly Learning to Align and Translate"_.

---

### Variants of Attention

| Type                    | Formula                                     | Description                                           |
| ----------------------- | ------------------------------------------- | ----------------------------------------------------- |
| **Dot-product**         | $e_{t,i} = s_t^\top h_i$                    | simplest form                                         |
| **Multiplicative**      | $e_{t,i} = s_t^\top W h_i$                  | adds learnable matrix $W$                             |
| **Additive (Bahdanau)** | $e_{t,i} = v^\top \tanh(W_1 s_t + W_2 h_i)$ | more flexible, learnable parameters $W_1$, $W_2$, $v$ |

---

## Summary

- **Statistical MT** relied on word/phrase alignment and n-gram models.
- **Seq2Seq NMT** replaced hand-designed components with neural end-to-end learning.
- **Beam Search** improves decoding by exploring multiple hypotheses.
- **Attention** removes the encoder bottleneck and enables interpretability.
