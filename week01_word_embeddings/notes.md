# Lecture 1: Word Embeddings

**Instructor:** Nikolay Karpachev
**Course:** ML MIPT Advanced
**Date:** 05.02.2024

## Overview

- NLP intro → preprocessing → classical features (BoW, TF-IDF) → embeddings (Word2Vec, GloVe)

## NLP Tasks

- **Classification:** binary (spam, sentiment), multi-class (topic), multi-label (#hashtags)
- **Regression:** predict continuous values (e.g., price)

Common baselines: linear models, logistic regression.

## Text Preprocessing

- **Tokenization:** split text into tokens.
- **Normalization:**

  - _Stemming_ (Porter, Lancaster, Snowball): heuristic suffix stripping.
  - _Lemmatization_ (WordNet): dictionary base form; better with POS tags.

- **Other:** case, punctuation, contractions, numbers/dates/IDs, stopwords, tags.
- **Tools:** `nltk`, `BeautifulSoup`, `re`, `pymorphy2`.

## Classical Features

### Bag-of-Words (BoW)

- Counts per vocabulary term; ignores order; high-dimensional & sparse; no semantics.

### N-grams & Collocations

- Use bigrams/trigrams to capture short context.
- Keep meaningful collocations; drop very frequent function-word n-grams and ultra-rare noise.

## TF–IDF

Display formulas:

$$
\mathrm{TF}(t,d)=\frac{\text{count}(t \in d)}{\text{total terms in } d}
$$

$$
\mathrm{IDF}(t,D)=\log\left(\frac{|D|}{|{d \in D: t \in d}|}\right)
$$

$$
\mathrm{TF-IDF}(t,d,D)=\mathrm{TF}(t,d)\cdot \mathrm{IDF}(t,D)
$$

- Emphasizes rare-but-informative terms.
- Implementation: `sklearn.feature_extraction.text.TfidfVectorizer`.

## Why Embeddings?

- **One-hot**: huge, sparse, no similarity.
- **Goal:** dense vectors encoding meaning & similarity (distributional hypothesis: _“You shall know a word by the company it keeps.”_).

## Count-Based Embeddings (Matrix Factorization)

- Build word–context co-occurrence matrix in a fixed window.
- Use PMI:

$$
\mathrm{PMI}(u,v)=\log \frac{p(u,v)}{p(u),p(v)}
$$

- Often use **positive PMI (pPMI)** by clipping negatives to 0.
- Reduce dimensionality (e.g., SVD) → word vectors.

## Predictive Embeddings: Word2Vec (Mikolov et al., 2013)

Two training objectives:

| Model         | Objective                         | Notes                         |
| ------------- | --------------------------------- | ----------------------------- |
| **CBOW**      | Predict center word from context  | Fast; good for frequent words |
| **Skip-gram** | Predict context words from center | Slower; better for rare words |

**Efficiency tricks**

- Phrase detection: treat frequent multi-word expressions as one token.
- **Subsampling** high-frequency words (drop overly common words):

$$
P_{\text{discard}}(w) = 1 - \sqrt{\frac{t}{f(w)}}
$$

where (f(w)) is the corpus frequency and (t) is a small threshold (e.g., (10^{-5})).

- **Negative sampling:** update only a few sampled “negative” words per example (instead of a full softmax).

## Count-Predictive Hybrid: GloVe

Learns embeddings from global co-occurrence counts with a weighted least squares objective:

$$
J = \sum_{i,j} f(X_{ij})\left(w_i^\top \tilde{w}_j + b_i + \tilde{b}*j - \log X*{ij}\right)^2
$$

- $(X\_{ij})$: times word (j) appears in context of (i)
- $(w_i, \tilde{w}\_j)$: word/context vectors; $(b_i, \tilde{b}\_j)$: biases
- $(f(\cdot))$: weighting to down-weight very rare and very frequent co-occurrences

**Comparison**

| Approach | Type                      | Core idea                    |
| -------- | ------------------------- | ---------------------------- |
| Word2Vec | Predictive                | Fit local context prediction |
| GloVe    | Count-based (neural loss) | Fit global log-counts        |

## Properties & Analogies

- Similar contexts → nearby vectors.
- Linear structure captures relations:

$$
\text{king} - \text{man} + \text{woman} \approx \text{queen}
$$

- Typical dimensionality: 100–300.

## Summary

1. One-hot / BoW / TF-IDF: simple, sparse, order-agnostic.
2. Co-occurrence + factorization: PMI, SVD → dense vectors.
3. Predictive models (Word2Vec) and count-predictive hybrids (GloVe) → high-quality embeddings capturing semantics.
