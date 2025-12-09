This repository contains the code for our ECE1508 Reinforcement Learning final project.  
We fine-tune a DistilGPT-2 language model on the IMDB movie review dataset, and then apply
Reinforcement Learning from Human Feedback (RLHF-style PPO) using automatic reward signals
based on BLEU, ROUGE-L and semantic similarity.

---
Project Overview

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

Repository Structure:

1.1508_project.ipynb        # Main training notebook (SFT + PPO)

2.RL_Project.py             # Python script version of PPO training

3.demo.ipynb                # Example input-output demo

4.ModelCheckpoint/          # Saved model checkpoints (SFT / PPO)

5.report.pdf                # Final written report

6.requirements.txt          # Dependencies

7.README.md                 # This file

Installation
1. Clone the repository
2. Install dependencies : pip install -r requirements.txt

Running the Code: use Jupyter Notebook
open:1508_project.ipynb
