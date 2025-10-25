# Lecture 4: Transfer Learning

**Instructor:** Nikolay Karpachev
**Course:** ML MIPT Advanced
**Date:** 26.02.2024

---

## 🎯 Motivation

Training a text classification model **from scratch** requires:

- Labeled samples
- Pre-extracted features

→ **Expensive** and **data-hungry** process.
→ It’s unclear _which features_ are most useful for downstream tasks.

**Transfer Learning** allows us to reuse knowledge learned from one task or dataset for another.

---

## 🧩 Definition

**Transfer Learning:**
Using representations or parameters learned from one task–dataset pair to improve performance on a _different_ but related task or dataset.

---

## 🧭 Taxonomy

### 1. Sequential Transfer Learning

- Train a model on a **source task** (often large-scale, generic).
- Transfer knowledge to a **target task** (usually smaller, specific).
- Typical approach in NLP: **pretrain → finetune**.

### 2. Multi-task Transfer Learning

- Jointly train on multiple tasks to learn shared representations.

### 3. Domain Adaptation

- Adjust a model trained on one domain (e.g., news) to perform well on another (e.g., tweets).

---

## 📖 Transfer via Word Embeddings

### Basic Idea

Re-use pretrained word embeddings (like Word2Vec, GloVe) instead of training from scratch.

- Embedding layer captures **semantic similarity**.
- Often pretrained on huge corpora (e.g., Wikipedia).
- The rest of the model (e.g., classifier) is trained for the specific downstream task.

**Limitation:**
Static embeddings do not depend on context.
→ Same vector for “bank” in _river bank_ and _central bank_.

---

## 🧠 Transfer via Contextualized Representations

Unlike static embeddings, **contextual embeddings** change based on surrounding words.

Example:

- “bank” in _river bank_ ≠ “bank” in _central bank_.

Approach:

- Train models that encode tokens **with their context**.
- Use these representations for downstream tasks.

---

## 🧬 Early Contextual Models

### **CoVe (Contextualized Word Vectors)**

- From the paper _“Contextualized Word Vectors Learned in Translation”_.
- **Key idea:**
  Translation requires deep token-level understanding.
  An NMT (Neural Machine Translation) encoder learns rich sentence representations.

**Procedure:**

1. Train an NMT encoder–decoder model.
2. Take the encoder part as a **feature extractor**.
3. Use encoder outputs as contextual embeddings (CoVe).
4. Combine with pretrained static embeddings (e.g., CoVe + GloVe).

---

### **ELMo (Embeddings from Language Models)**

- Based on **Language Model (LM) pretraining**, not translation.
- Learns contextual representations by predicting the next word (forward LM) and the previous word (backward LM).

**Architecture:**

- Character-level CNN for input representation (handles OOV words).
- Multi-layer BiLSTM trained as a bidirectional LM.
- Final representation = weighted sum of internal layers:
  $$
  \text{ELMo}(w_t) = \gamma \sum_i s_i h_{t,i}
  $$
  where $h_{t,i}$ are hidden states, and $s_i$ are learned scalar weights.

**Usage:**
Use pretrained ELMo as a feature extractor; add task-specific layers on top.

---

## ⚡ Transition: From Feature Extraction to Pretrained Models

| Approach                   | Example         | What is reused? | Fine-tuning required? |
| -------------------------- | --------------- | --------------- | --------------------- |
| Word embeddings            | Word2Vec, GloVe | Word vectors    | Usually yes           |
| Contextualized reps        | CoVe, ELMo      | Encoder outputs | Often frozen          |
| Pretrained language models | GPT, BERT       | Full model      | Finetuned end-to-end  |

---

## 🧱 Pretrained Language Models

### Why Language Modeling?

- **Self-supervised** — no manual labels needed.
- **Massive data** (web text, books, Wikipedia).
- **Generalizable features** — applicable to many downstream tasks.

---

## 🔮 GPT (Generative Pretraining Transformer)

### Key Idea

Autoregressive Language Model based on **Transformer Decoder**.

**Architecture:**

- Decoder-only Transformer with a **causal mask** (triangular).
- Predicts next token given previous context.

**Training Objective:**

$$
\mathcal{L}*{\text{GPT}} = -\sum_t \log P(x_t \mid x*{<t})
$$

**Fine-tuning (GPT-1):**

- Combine **unsupervised LM loss** and **supervised task loss**.
- Add task-specific head on top (classification, QA, etc.).
- Update **all parameters** during fine-tuning.

---

## 🧩 BERT (Bidirectional Encoder Representations from Transformers)

### Key Idea

Use a **bidirectional Transformer encoder** trained with two pretraining objectives:

1. **Masked Language Modeling (MLM)**
2. **Next Sentence Prediction (NSP)**

---

### ⚙️ Masked Language Modeling (MLM)

Since an encoder can see both left and right context, we **mask some tokens** and ask the model to predict them.

Example:

```
Input: The [MASK] barked loudly.
Target: The dog barked loudly.
```

**Loss:**

$$
\mathcal{L}*{\text{MLM}} = -\sum*{i \in M} \log P(x_i \mid x_{\setminus M})
$$

where $M$ is the set of masked positions.

---

### ⚙️ Next Sentence Prediction (NSP)

Helps the model understand **sentence relationships**.

- Input pairs: `[CLS]` Sentence A `[SEP]` Sentence B
- Model predicts whether Sentence B follows Sentence A in the corpus.

**Loss:**

$$
\mathcal{L}*{\text{NSP}} = - \log P(y*{\text{NSP}} \mid A, B)
$$

---

### Combined Objective

$$
\mathcal{L}*{\text{BERT}} = \mathcal{L}*{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

---

## 🧠 BERT Fine-tuning Tasks

### 1. Sentence Classification

- Input: `[CLS]` + sentence
- Use `[CLS]` embedding as sentence-level feature.

### 2. Sentence Pair Classification

- Input: `[CLS]` + Sentence 1 + `[SEP]` + Sentence 2
- Use `[CLS]` embedding for prediction (e.g., NLI, paraphrase detection).

### 3. Token Classification (e.g., NER, POS)

- Input: `[CLS]` + sentence
- Predict label for each token from its contextual embedding.

---

## 🔁 Summary

| Stage                     | Example         | Training Objective       | Application             |
| ------------------------- | --------------- | ------------------------ | ----------------------- |
| **Pretrained Embeddings** | Word2Vec, GloVe | Co-occurrence prediction | Fixed word vectors      |
| **Contextual Embeddings** | CoVe, ELMo      | NMT or LM-based          | Dynamic contextual reps |
| **Pretrained Models**     | GPT, BERT       | LM or MLM/NSP            | Full-model fine-tuning  |

---

## 🧩 Key Takeaways

- Transfer learning drastically reduces labeled data requirements.
- Contextual representations (CoVe, ELMo) marked the shift from static to dynamic embeddings.
- Large pretrained LMs (GPT, BERT) unified NLP tasks via **pretrain + finetune** paradigm.
- Modern NLP heavily relies on **self-supervised pretraining** over massive corpora.
