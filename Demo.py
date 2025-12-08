import os
os.makedirs("/content/drive/MyDrive/rlhf_checkpoints", exist_ok=True)
train_ppo_rlhf(
    policy=policy,
    ref_lm=ref_lm,
    tokenizer=tokenizer,
    rl_train_loader=rl_train_loader,
    rl_valid_loader=rl_valid_loader,
    epochs=4,
    lr=1e-6,
    ppo_epochs=1,
    save_dir="/content/drive/MyDrive/rlhf_checkpoints",
)

import torch
from transformers import AutoTokenizer
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = "/content/drive/MyDrive/rlhf_checkpoints"

tokenizer = AutoTokenizer.from_pretrained("distilgpt2", padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

def load_policy(epoch, dir):
    ckpt_path = os.path.join(dir, f"ppo_policy_epoch{epoch}.pt")
    print(f"Loading: {ckpt_path}")
    policy = PolicyValueModel("distilgpt2").to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()
    return policy

@torch.no_grad()
def generate(model, prompt, max_new_tokens=40, sample=True):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.lm.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=sample,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

policy_e1 = load_policy(1, CKPT_DIR)
policy_e2 = load_policy(2, CKPT_DIR)
policy_e3 = load_policy(3, CKPT_DIR)
policy_e4 = load_policy(4, CKPT_DIR)

prompt = "The movie was"
print("=== Epoch 1 ===")
print(generate(policy_e1, prompt))

print("\n=== Epoch 2 ===")
print(generate(policy_e2, prompt))

print("\n=== Epoch 3 ===")
print(generate(policy_e3, prompt))

print("\n=== Epoch 4 ===")
print(generate(policy_e4, prompt))

ppo_policy = load_policy(4,CKPT_DIR)

ppo_metrics = eval_metrics(
    ppo_policy.lm,
    sft_test_loader,
    max_batches=200,
    use_bertscore=True
)

print("\n=== PPO Epoch 4 (RLHF) ===")
for k, v in ppo_metrics.items():
    print(f"{k:12s}: {v}")
