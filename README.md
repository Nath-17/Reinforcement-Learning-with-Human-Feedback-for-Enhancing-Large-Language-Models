# Reinforcement-Learning-with-Human-Feedback-for-Enhancing-Large-Language-Models



## Overview

This project investigate the Transformers and how Reinforcement Learning with Human Feedback (RLHF) can be used to fine-tune Large Language Models (LLMs) using Proximal Policy Optimization (PPO). 

**Objective**

In this repository we developed a functional pipeline for RLHF. We used this pipeline to improve a Transformer-based model, here, GPT-2 by incorporating human preferences to produce better outputs. 


## Methodology

The RLHF pipeline is implemented in three main stages:

**Data Preprocessing** 

We utilized the _stanfordnlp/explain_like_im_five _dataset. To ensure computational efficiency and model stability, we applied constraints:
- Token Filtering: All sequences were capped at 256 tokens to prevent memory issues.
- Transformed raw Reddit upvote data into "chosen" and "rejected" pairs to create a preference signal.
- Data Split: We used a subset of 1,000 training samples for the Reward Model and 1,000 samples for PPO optimization to demonstrate the pipeline's functionality.

**Reward Modeling**

We trained a Reward Model based on GPT-2 on this dataset to predict human preferences by evaluating the quality of responses.

It was trained to maximize the difference between the "chosen" (preferred) text and the "rejected" text.

This model provides a score used by the PPO algorithm.

**Proximal Policy Optimization**
Using the Hugging Face TRL (Transformer Reinforcement Learning) library, we fine-tuned the base GPT-2 policy. This optimization enables to get better answers from our model. 
Environment: The Reward Model provides feedback on the quality of generated explanations.




## Key results


## Limitations

Dataset Size: Expanding the training set beyond 1,000 samples would likely push the Reward Model accuracy beyond the 50% barrier.

Compute Constraints: The 256-token limit was necessary for this experiment but could be expanded on more powerful hardware to capture more detailed explanations.
