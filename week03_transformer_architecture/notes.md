# Lecture 3: Self-Attention & Transformer

**Instructor:** Nikolay Karpachev
**Course:** ML MIPT Advanced
**Date:** 19.02.2024

---

## Outline

1. Recap: Attention in Seq2Seq
2. CNN for textual data
3. Transformer architecture
4. Self-Attention
5. Positional Encoding
6. Layer Normalization
7. Q&A

---

## 🧠 Recap: Attention in Seq2Seq

- Attention connects the decoder directly to encoder outputs.
- Helps model focus on relevant parts of the input sequence.
- Provides:

  - **Better performance on long sequences**
  - **Word alignment interpretability**

---

## 🧩 CNN for Textual Data

### Motivation

- RNNs process sequentially → limited parallelism.
- CNNs can **capture local patterns** and **process in parallel**.

### CNN Structure

- Each layer applies **convolutional filters** over text embeddings.
- Filters detect local features like $n$-grams or phrase-level patterns.
- Outputs are **invariant to sequence length**.

### CNNs in NLP

- Common in early text classification models (e.g., Kim 2014).
- Can be used in Seq2Seq encoders for faster feature extraction, but lack **global dependencies** → motivates **self-attention**.

---

## ⚙️ Transformer Architecture

### Overview

- Introduced in _“Attention Is All You Need”_ (Vaswani et al., 2017).
- Replaces **RNNs/CNNs** entirely with **self-attention**.
- Achieved **28.4 BLEU** on WMT 2014 English–German translation.
- Fully **parallelizable**, **faster**, and **scalable**.

### Key Principles

- **No recurrence or convolution.**
- Built entirely from:

  - Self-attention layers
  - Feed-forward networks
  - Residual connections
  - Layer normalization
  - Positional encoding

---

## 🔁 Self-Attention Mechanism

### Intuition

Example:

> “The animal didn’t cross the street because it was too tired.”

Self-attention allows the model to learn that _“it”_ refers to _“animal”_ — building contextual understanding across the sequence.

---

### Step-by-Step Computation

1. **Create Query, Key, Value vectors:**

   - For each input embedding $x_i$, compute:
     $$
     q_i = W_Q x_i, \quad k_i = W_K x_i, \quad v_i = W_V x_i
     $$
     where $W_Q$, $W_K$, $W_V$ are learned weight matrices.

2. **Compute attention scores:**

   $$
   e_{ij} = q_i^\top k_j
   $$

3. **Scale the scores:**

   $$
   e_{ij} ; \leftarrow ; \frac{e_{ij}}{\sqrt{d_k}}
   $$

   where $d_k$ is the dimensionality of keys (e.g., 64).
   Scaling stabilizes gradients for large $d_k$.

4. **Normalize with softmax:**

   $$
   \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_j \exp(e_{ij})}
   $$

5. **Weighted sum of values:**

   $$
   z_i = \sum_j \alpha_{ij} v_j
   $$

6. **Output $z_i$** is the contextual representation of token $i$.

---

### Matrix Form

Let $X$ be the matrix of all input embeddings.

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

Then:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}!\left(\frac{Q K^\top}{\sqrt{d_k}}\right)V
$$

---

## 🧠 Multi-Head Attention

Instead of computing one attention distribution, Transformers compute multiple **parallel heads**, each with its own $W_Q$, $W_K$, and $W_V$.

For $h$ heads:

$$
\text{head}_i = \mathrm{Attention}(QW_Q^{(i)}, KW_K^{(i)}, VW_V^{(i)})
$$

Concatenate all heads and project:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
$$

### Why Multi-Head?

- Each head captures different types of relationships:

  - Syntactic (e.g., subject–verb)
  - Semantic (e.g., word co-reference)

- Improves expressivity and representation power.

---

## 🌐 Positional Encoding

Since the Transformer has no recurrence, it needs **positional information**.

### Formula

Each position $pos$ and dimension $i$ gets a sinusoidal embedding:

$$
\text{PE}*{(pos, 2i)} = \sin!\left(\frac{pos}{10000^{2i/d*{model}}}\right)
$$

$$
\text{PE}*{(pos, 2i+1)} = \cos!\left(\frac{pos}{10000^{2i/d*{model}}}\right)
$$

### Purpose

- Injects order information into embeddings.
- Enables model to infer **relative distances** between tokens.

---

## ⚖️ Layer Normalization

Used throughout the Transformer to stabilize training.

Given an input vector $x$ with mean $\mu$ and variance $\sigma^2$:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

- $\gamma$, $\beta$ — learned scale and shift parameters
- Applied **before** or **after** residual connections depending on implementation (Pre-LN vs. Post-LN)

---

## 🧱 Encoder–Decoder Structure

### Encoder

- Stack of $N$ identical layers.
- Each layer:

  1. Multi-head self-attention
  2. Feed-forward network (2 linear layers + ReLU)
  3. Residual connection + LayerNorm

### Decoder

- Also $N$ layers, each with:

  1. **Masked multi-head self-attention** (prevents looking at future tokens)
  2. Encoder–decoder attention
  3. Feed-forward network
  4. Residual + LayerNorm

### Output

- Final linear layer + softmax → predicts next token probabilities.

---

## 🚀 Transformer Highlights

- **Parallelized**: all tokens processed simultaneously.
- **Short path length**: direct interactions between any two positions.
- **Scalable**: deeper networks with attention layers handle long dependencies.
- **Self-similarity modeling**: useful for long-range dependencies and graph structures.

---

## 🏁 Summary

- **Self-attention** replaces recurrence with global context computation.
- **Multi-head attention** learns multiple relationships in parallel.
- **Positional encoding** reintroduces word order.
- **Layer normalization** and **residual connections** stabilize training.
- **Transformer** revolutionized NLP — foundation for models like BERT, GPT, and T5.
