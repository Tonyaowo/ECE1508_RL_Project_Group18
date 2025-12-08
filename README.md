# ECE1508 Final Project – RLHF for Movie Review Continuation

This repository contains the code for our ECE1508 Reinforcement Learning final project.  
We fine-tune a DistilGPT-2 language model on the IMDB movie review dataset, and then apply
Reinforcement Learning from Human Feedback (RLHF-style PPO) using automatic reward signals
based on BLEU, ROUGE-L and semantic similarity.

---

## 1. Project Overview

- **Base model**: `distilgpt2` (Hugging Face Transformers)
- **Dataset**: IMDB (`datasets` library)
- **Stage 1 – Supervised Fine-Tuning (SFT)**  
  - Turn raw IMDB reviews into (prompt, continuation) pairs  
  - Fine-tune DistilGPT-2 on continuation prediction
- **Stage 2 – RLHF-style PPO**  
  - Use SFT model as initialization and reference model  
  - Define a reward based on:
    - Sentiment
    - Repetition
    - Length
    - Coherence
    - Relevence
  - Train a policy-value model with PPO and an adaptive KL controller
- **Evaluation**:
  - Automatic metrics: BLEU, ROUGE-1/2/L, METEOR, Distinct-1/2, BERTScore
  - Average reward on a held-out test set

## 2. Repository Structure
├── README.md
├── requirements.txt
├── demo.py
├── notebooks/
│   └── ECE1508RL-FinalProject.ipynb
├── models/
│   ├── sft_model/
│   └── rlhf_checkpoints_v2/
├── report/
│   └── final_report.pdf
