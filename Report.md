## Introduction

## 1. Understanding and Utilizing Causal Transformers

### 1.1 Introduction to transformers

Transformer models, introduced by Vaswani et al. (2017), constitute a breakthrough in sequence modeling. First, they were proposed for machine translation, but Transformers have become the standarts in natural language processing (NLP), speech processing, computer vision and time series analysis.

Unlike recurrent neural networks (RNNs) or long short-term memory networks (LSTMs), Transformers do not process sequences sequentially. They rely on a self-attention mechanism that allows each element of a sequence to directly attend to all other elements in parallel. This design enables more efficient training, better handling of long-range dependencies.

### 1.2 Casual transformers

Causal Transformers constitute a specific class of Transformers designed for autoregressive tasks, i.e. modeling a sequence by predicting each element conditionally on on the previously predicted elements. They are particularly well-suited for generative tasks, such as text generation, where outputs must be produced sequentially and depend only on past information. 
This causal constraint is essential for modeling natural language as a sequential stochastic process and directly determines both the training objective and the inference procedure. Two well-known examples are GPT and LLaMA.

Architecturally, Causal Transformers rely on a decoder-only design, combining masked self-attention, feedforward layers, and positional encodings to learn high-dimensional representations of token sequences.


Let $(x_1, x_2, \dots, x_T)$ denote a sequence of discrete tokens drawn from a vocabulary $\mathcal{V}$.  
A Causal Transformer models the joint probability of the sequence using an autoregressive factorization:

$$
P(x_1, \dots, x_T)
= \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})
$$

This formulation defines the Causal Language Modeling objective.

#### Causal self-attention

Given an input embedding matrix $X \in \mathbb{R}^{T \times d}$, linear projections produce queries $Q$, keys $K$, and values $V$:

$$
Q = X W_Q, \quad
K = X W_K, \quad
V = X W_V
$$

The self-attention operation is defined as:

$$
\text{Attention}(Q, K, V)
= \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} + M \right) V
$$

where $M \in \mathbb{R}^{T \times T}$ is a causal mask such that:

$$
M_{ij} =
\begin{cases}
0 & \text{if } j \le i \\
-\infty & \text{if } j > i
\end{cases}
$$

This mask ensures that the representation at position $i$ depends only on tokens  
$(x_1, \dots, x_i)$.


#### Training objective

The parameters $\theta$ of the model are learned by minimizing the negative log-likelihood of the observed sequence:

$$
\mathcal{L}(\theta)
= - \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \dots, x_{t-1})
$$

In practice, this loss is implemented as a cross-entropy loss between the predicted token distribution and the true next token.  
The training procedure is self-supervised, as the targets are derived directly from the input data.

Causal Transformers rely on a decoder-only architecture and generate sequences in an autoregressive manner, where each token is predicted conditionally on previously generated tokens, mirroring the probabilistic structure learned during training. The use of causal self-attention enforces this temporal constraint, while positional encodings such as learned embeddings or rotary positional embeddings allow the model to capture token order and long-range dependencies. These models scale efficiently with increased depth, width, and training data due to parameter sharing across sequence positions. However, their autoregressive inference and the quadratic complexity of self-attention introduce limitations in terms of generation latency and long-context efficiency.


## 2. Implementing the RLHF Pipeline

## 3. Training a Reward Model

To align the language model with human preferences, we trained a Reward Model on the Stanford Human Preferences (SHP) dataset. This model learns to assign higher scores to responses that humans prefer over alternative responses. The Reward Model is a pre-trained GPT-2 model augmented with a scalar value head, allowing it to output a single reward per sequence. This approach is a key component of the Reinforcement Learning with Human Feedback (RLHF) pipeline, providing a supervised signal to guide subsequent policy optimization.

For this experiment, we selected examples from three subreddits to cover diverse domains:

| Subreddit | #Train | #Validation | #Test | Total |
|-----------|--------|------------|-------|-------|
| explainlikeimfive | 19,592 | 1,014 | 1,070 | 21,676 |
| askscience | 13,316 | 899 | 977 | 15,192 |
| askphilosophy | 10,307 | 608 | 677 | 11,592 |

Dataset preprocessing included:
**Generating paired examples** `(chosen, rejected)` using the `labels` field.  
**Tokenization** with the GPT-2 tokenizer, combining the prompt and response for both chosen and rejected sequences.  
**Padding and truncation** to a maximum of 512 tokens for GPU efficiency.

This process resulted in a **training dataset of ~48k examples** and a small validation subset (~500 examples) for monitoring.


## 4. Optimization with Proximal Policy Optimization (PPO)

## Conclusion

