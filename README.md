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

**Reward Model Training**

- Training loss = 0.52
- Validation loss = 0.76
- Accuracy = 0.45

**Proximal Policy Optimization**

- Loss: 4.769	to 0.194
- KL Divergence: 0.002 to 2.35
- Score: 1.434 to	1.434

The PPO training showed  convergence with a reduction in Value Loss (4.76 to 0.19), and a controlled increase in KL Divergence (2.35) confirms the model successfully adapted its policy toward human preferences. 

**Example of text generated**

PROMPT:    "The best way to learn something new is"

Base Model:  "...to create a list of things you will like about yourself. In this example, I am writing my life story and doing it in English instead ("

RLHF PPO: "...to get as much practice in it. You'll find that you will be able to do so by working with your own personal trainer and the ones"

We can see a difference here. The Base Model doesn't really answer the question, but the RLHF PPO model gives an actual advice. This shows the Reward Model did its job in making the model's outputs more practical and easier to understand, just like the 'Explain Like I'm Five' style even if there is room for improvements for the second part of the answer. 
## Limitations

Dataset Size: Expanding the training set beyond 1,000 samples would likely push the Reward Model accuracy beyond the 50% barrier.

Compute Constraints: The 256-token limit was necessary for this experiment but could be expanded on more powerful hardware to capture more detailed explanations.
