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

This project implements Reinforcement Learning from Human Feedback (RLHF) to fine-tune a GPT-2 model on the Stanford ELI5 dataset. The objective is to align the model’s responses with human preferences for simplicity, and accuracy. The pipeline consists of a preprocessing of the dataset step and two main stages: training a Reward Model to act as human judgment, and optimizing the base LLM using Proximal Policy Optimization.

**Data preparation** 

The raw dataset from _Stanfordnlp_ is not  binarized. It contains a question (the "history" or "prompt") and a list of multiple human answers, each with a specific score based on community upvotes. Firstly, we transformed these answers into pairs. For every question, we selected a "chosen" answer (higher score) and a "rejected" answer (lower score).

Secondly, we formatted the data into a structure compatible with the Hugging Face _RewardTrainer_. Through this tokenization process, the dataset was transformed into the columns _input_ids_chosen_/_attention_mask_chosen_ for the preferred responses and _input_ids_rejected_ _attention_mask_rejected_ for the non-preferred alternatives.

Finally, to manage computational limitations and ensure stable training on GPT-2, we implemented a filtering process: 
- Token Limit: We removed any responses exceeding 256 tokens. This prevents Out-of-Memory errors. There was 13,123 samples under 256 tokens.
- Subsampling for Efficiency: To optimize training time while maintaining a valid  pipeline, we utilized the following subsets:
  - Reward Model: 1,000 training pairs | 200 evaluation pairs.
  - PPO Optimization: 1,000 training prompts | 200 evaluation prompts.


## 3. Training a Reward Model

To align the language model with human preferences, we trained a Reward Model on the Stanford Human Preferences dataset. This model learns to assign higher scores to responses that humans prefer over alternative responses. The Reward Model a pre-trained GPT-2 that is transformed from a text generator into a regression model

The Reward Model provides the feedback the model needs to improve during the PPO phase. The fianl model uses these scores to learn which answers are better suited for moving from generic responses to preferred ones.

**Results**

The Training Loss showed a promising downward trend, it dropped from 2.49 at step 50 to 0.52 by step 750. However, the Validation Loss struggled to decrease below 0.71. 
The model shows an accuracy of 47.00%. 5his indicates the model  fails to differentiate between the chosen and rejected text pairs, the near-random accuracy suggests that a larger training subset (beyond 1,000 samples) and additional epochs may be required to capture the nuances of responses (it was not implemented due to computational resources). 


## 4. Optimization with Proximal Policy Optimization (PPO)

Then, we used our trained Reward Model to fine-tune the GPT-2 policy through reinforcement learning. The goal is to generate better outputs that align with human preferences. In this setup, the Reward Model provides a  score for every response the model produces. By using the PPO (Proximal Policy Optimization) algorithm, we optimized the model to maximize these reward scores.

We began with an initial LLM (GPT-2), that generates responses to prompts in the training dataset, for each response the Reward Model gives a score and we fine-tune the model to align its outputs with human preferences. To prevent generating nonsensical text, PPO applies a KL Divergence penalty, which measures how much the updated policy differs from the original model.

**Results**




## Conclusion

