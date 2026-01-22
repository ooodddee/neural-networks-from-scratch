# Makemore Part 2: Bigram Language Model

Building a character-level bigram language model from scratch.

## Overview

This project implements a bigram model to understand:
- How to extract character statistics from data
- Probability estimation with Laplace smoothing
- Autoregressive sampling for generation
- Bridging to neural networks (next phase)

## Mathematical Concepts

### Bigram Statistics
Count pairs of consecutive characters:
$$N_{ij} = \text{count of character } i \text{ followed by character } j$$

### Probability Estimation (MLE with Smoothing)
$$P(c_{i+1}|c_i) = \frac{N_{c_i, c_{i+1}} + \alpha}{\sum_j (N_{c_i, j} + \alpha)}$$

where $\alpha = 1$ (Laplace smoothing)

### Generation (Autoregressive Sampling)
Starting from initial character, sample next character from learned distribution:
$$c_{i+1} \sim \text{Categorical}(P[c_i])$$

## Files

- `bigram.ipynb`: Main implementation and experiments
- Data: `names.txt` (downloaded from Karpathy's repo)

## Implementation Phases

**Phase 1**: Data exploration & bigram statistics
- Load names dataset
- Build frequency matrix N
- Visualize with heatmap

**Phase 2**: Probability estimation
- Apply Laplace smoothing
- Normalize to get probability matrix P
- Verify row sums = 1

**Phase 3**: Generation
- Implement autoregressive sampling
- Generate sample names
- Compare with baseline

## My Insights

### Phase 2 - Laplace Smoothing Bug

**Problem**: Without smoothing, unseen bigrams have zero probability.

**What went wrong**: 
First attempt computed probabilities without adding 1:
```python
# ❌ WRONG
P = N.float()
P = P / P.sum(1, keepdim=True)
```

This caused generation to fail silently on rare character combinations.

**How I debugged it**:
1. Compared generated names with raw data - too limited vocabulary
2. Checked probability matrix - had exact zeros
3. Realized Laplace smoothing was missing

**Root cause**: Zero probabilities → generation gets stuck

**Solution**:
```python
# ✅ CORRECT
P = (N + 1).float()
P = P / P.sum(1, keepdim=True)
```

**Key takeaway**: Always handle zero probabilities in probabilistic models with smoothing.

---

**Connection to Micrograd**: Once we have working bigram model, the next phase will replace the count matrix with a neural network trained using our autograd engine.
