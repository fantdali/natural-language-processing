# Lecture 6: Efficient Inference

**Instructor:** Natalia Luneva
**Course:** ML MIPT Advanced
**Date:** 17.03.2025

---

## 🧭 Outline

1. Key–Value (KV) Cache
2. Paged Attention
3. Flash Attention

---

## 🧠 Key–Value Cache

### Purpose

Reduce redundant computations during **autoregressive generation**.

In decoder-based models (e.g., GPT), each new token is predicted using all previously generated tokens.
Naively, this requires recomputing all **key** and **value** vectors at every step — inefficient for long sequences.

---

### Concept

Each attention layer stores intermediate matrices:

$$
\text{Attention}(Q, K, V) = \text{softmax}!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- $Q$ (query) → current token
- $K, V$ (keys, values) → all previous tokens

Instead of recomputing $K$ and $V$ at each timestep,
we **cache them** after their first computation and **reuse** them for future tokens.

---

### Benefits

- ✅ **Reduced computation** — avoids repeated KV computation
- ✅ **Memory reuse** — intermediate results stored once
- ✅ **Scalability** — enables efficient handling of long sequences

### Drawback

- ❌ **Increased memory usage** — cached matrices require extra storage

---

### KV Cache Strategies

| Type              | Description                                       | Computation                   | Use Case                                                   |
| ----------------- | ------------------------------------------------- | ----------------------------- | ---------------------------------------------------------- |
| **Static Cache**  | Keys & values computed once, remain fixed         | Precomputes KV for full input | Encoder–decoder models, batch inference with shared prefix |
| **Dynamic Cache** | Updated incrementally as new tokens are generated | Incremental KV updates        | Fully autoregressive text generation                       |

Other variants include **offloaded** and **quantized** caches (e.g., for GPU memory reduction).

---

## 💾 Paged Attention

### Goal

Reduce **memory waste** caused by inefficient KV cache allocation during inference.

---

### Key Concepts

- **Contiguous memory:** data stored in a continuous block of addresses.
- **Internal fragmentation:** process allocated more memory than needed.
- **External fragmentation:** available free blocks are too small to satisfy new requests.

---

### Problem in LLM Serving

LLM inference servers (like vLLM or TensorRT-LLM):

- Store each request’s KV cache in **contiguous memory**.
- Preallocate memory for the _maximum_ sequence length.
- Different requests → different lengths → **fragmentation** and **waste**.

**Result:** Only 20–40% of allocated KV memory is actually used!

---

### Solution: PagedAttention (vLLM, 2023)

Allocates memory in **fixed-size blocks (pages)** instead of one contiguous region.

Each block:

- Holds a fixed number of tokens
- Can be shared across different requests
- Allocated **on demand**

---

### Advantages

| Problem                | Solution                                  |
| ---------------------- | ----------------------------------------- |
| Internal fragmentation | Small, fixed-size blocks                  |
| External fragmentation | Equal block sizes → uniform allocation    |
| Memory waste           | Blocks reused dynamically across requests |

PagedAttention improves **GPU memory utilization** and **throughput** in large-scale LLM serving.

**Reference:** [arXiv:2309.06180](https://arxiv.org/pdf/2309.06180)

---

## ⚡ Flash Attention

### Goal

Reduce **memory access** overhead in attention computation.

---

### Hardware Background

| Memory Type | Description                     | Use               |
| ----------- | ------------------------------- | ----------------- |
| **SRAM**    | Fast, low-latency static memory | Cache             |
| **HBM**     | High-bandwidth memory on GPUs   | Matrix ops        |
| **DRAM**    | Main system memory              | General computing |

Standard attention stores intermediate matrices in **HBM**, which scales as $O(N^2)$ (with sequence length $N$).
→ **Memory-bound bottleneck** for long sequences.

---

### Standard Attention Memory Cost

$$
A = QK^\top, \quad S = \text{softmax}(A)
$$

Both $A$ and $S$ are $N \times N$ matrices.
Storing them → **quadratic memory usage**.

---

### Flash Attention: Key Ideas

**Challenges:**

1. Compute softmax reduction **without** storing the full $A$.
2. Perform backpropagation **without** saving intermediate attention maps.

**Solutions:**

- **Tiling:**
  Process attention in small _blocks_ that fit in fast SRAM.
- **Recomputation:**
  Skip storing intermediate results; recompute locally during backward pass.

---

### Block-Wise Computation

Instead of computing attention for all token pairs:

$$
\text{Attention}(Q, K, V) = \mathrm{softmax}!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Flash Attention divides $Q$, $K$, and $V$ into **blocks** and computes attention locally within each tile.
This reduces reads/writes from high-latency HBM.

---

### Blocked Softmax

- Standard softmax requires normalization across the entire sequence.
- FlashAttention computes **softmax within each block**, approximating global normalization efficiently.

---

### Benefits

✅ Lower memory usage — no need to store full attention matrix
✅ Reduced HBM I/O — fewer reads/writes
✅ Significant speedup for long sequences

**Reference:** [arXiv:2205.14135](https://arxiv.org/pdf/2205.14135)

---

## 🧩 Summary

| Technique           | Focus             | Problem Solved                 | Benefit                            |
| ------------------- | ----------------- | ------------------------------ | ---------------------------------- |
| **KV Cache**        | Computation       | Repeated attention computation | Faster autoregressive decoding     |
| **Paged Attention** | Memory management | Fragmentation & waste          | Better GPU memory utilization      |
| **Flash Attention** | Memory bandwidth  | Quadratic memory access        | Efficient long-sequence processing |

---

## 📚 Further Reading

- **KV Cache**
  [Hugging Face KV Cache Docs](https://huggingface.co/docs/transformers/main/kv_cache)
- **Paged Attention**
  [vLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html), [arXiv:2309.06180](https://arxiv.org/pdf/2309.06180)
- **Flash Attention**
  [Hugging Face Docs](https://huggingface.co/docs/text-generation-inference/conceptual/flash_attention), [arXiv:2205.14135](https://arxiv.org/pdf/2205.14135)
- **Self-Study**

  - [arXiv:2410.03065](https://arxiv.org/pdf/2410.03065)
  - [KV Cache Quantization Blog](https://huggingface.co/blog/kv-cache-quantization)
  - [arXiv:2307.08691](https://arxiv.org/pdf/2307.08691)
