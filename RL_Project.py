!pip install datasets

!pip install nltk rouge-score
!pip install rouge-score bert-score

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from google.colab import drive
drive.mount('/content/drive')

SFT_DIR = "/content/drive/MyDrive/sft_model"
os.makedirs(SFT_DIR, exist_ok=True)

RLHF_DIR = "/content/drive/MyDrive/rlhf_model"
os.makedirs(RLHF_DIR, exist_ok=True)

"""# Load imdb dataset"""

from datasets import load_dataset, DatasetDict

raw = load_dataset("imdb")

train_texts = list(raw["train"]["text"])
test_texts  = list(raw["test"]["text"])

all_texts = train_texts + test_texts
len(all_texts)

"""# Split dataset (training: validation: test = 6:2:2)"""

from datasets import Dataset

all_ds = Dataset.from_dict({"text": all_texts})

splits = all_ds.train_test_split(test_size=0.4, seed=42)
train_ds = splits["train"]
tmp_ds   = splits["test"]

splits2 = tmp_ds.train_test_split(test_size=0.5, seed=42)
valid_ds = splits2["train"]
test_ds  = splits2["test"]

len(train_ds), len(valid_ds), len(test_ds)

import re

def clean_text(text):

    text = re.sub(r"<.*?>", " ", text)


    text = re.sub(r"\s+", " ", text).strip()

    return text

def build_sft_pair(text: str, max_prompt_len=40, max_target_len=64):


    text = clean_text(text)

    tokens = text.split()
    if len(tokens) <= max_prompt_len + 5:
        return None

    prompt = " ".join(tokens[:max_prompt_len])
    target = " ".join(tokens[max_prompt_len:max_prompt_len + max_target_len])

    return {
        "prompt": prompt,
        "target": target
    }

def sft_map_function(batch):
    prompts, targets = [], []

    for text in batch["text"]:
        r = build_sft_pair(text)
        if r is not None:
            prompts.append(r["prompt"])
            targets.append(r["target"])

    return {"prompt": prompts, "target": targets}

sft_train = train_ds.map(
    sft_map_function,
    batched=True,
    remove_columns=train_ds.column_names
)

sft_valid = valid_ds.map(
    sft_map_function,
    batched=True,
    remove_columns=valid_ds.column_names
)

sft_test = test_ds.map(
    sft_map_function,
    batched=True,
    remove_columns=test_ds.column_names
)

print("SFT train:", len(sft_train))
print("SFT valid:", len(sft_valid))
print("SFT test:",  len(sft_test))

sft_train[0]

def build_rl_prompt(text, max_prompt_len=20):
    text = clean_text(text)
    tokens = text.split()

    if len(tokens) == 0:
        return None

    prompt = " ".join(tokens[:max_prompt_len])

    return {"prompt": prompt}

def rl_map_function(batch):
    prompts = []
    for text in batch["text"]:
        r = build_rl_prompt(text)
        if r is not None:
            prompts.append(r["prompt"])
    return {"prompt": prompts}

rl_train = train_ds.map(
    rl_map_function,
    batched=True,
    remove_columns=train_ds.column_names
)

rl_valid = valid_ds.map(
    rl_map_function,
    batched=True,
    remove_columns=valid_ds.column_names
)

rl_test = test_ds.map(
    rl_map_function,
    batched=True,
    remove_columns=test_ds.column_names
)

print("RL train:", len(rl_train))
print("RL valid:", len(rl_valid))
print("RL test:",  len(rl_test))

rl_train[0]

"""# Create dataloader"""

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilgpt2",padding_side="right")
tokenizer.pad_token = tokenizer.eos_token

"""# STF dataloader"""

import torch

def sft_collate_fn(batch):
    """
    batch: [{"prompt": "...", "target": "..."}]
    return: input_ids, attention_mask, labels
    """

    inputs = []
    labels = []

    for sample in batch:
        prompt = sample["prompt"]
        target = sample["target"]

        full_text = prompt + " " + target

        # Tokenize
        enc = tokenizer(
            full_text,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0]


        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        prompt_len = len(prompt_ids)


        label_ids = input_ids.clone()


        label_ids[:prompt_len] = -100

        inputs.append({
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": label_ids
        })


    batch_input_ids = torch.nn.utils.rnn.pad_sequence(
        [x["input_ids"] for x in inputs],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [x["attention_mask"] for x in inputs],
        batch_first=True,
        padding_value=0
    )
    batch_labels = torch.nn.utils.rnn.pad_sequence(
        [x["labels"] for x in inputs],
        batch_first=True,
        padding_value=-100
    )

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }

from torch.utils.data import DataLoader

# Training loader
sft_train_loader = DataLoader(
    sft_train,
    batch_size=8,
    shuffle=True,
    collate_fn=sft_collate_fn
)

# Validation loader
sft_valid_loader = DataLoader(
    sft_valid,
    batch_size=8,
    shuffle=False,
    collate_fn=sft_collate_fn
)

# Test loader (最终报告使用)
sft_test_loader = DataLoader(
    sft_test,
    batch_size=8,        # or larger, e.g. 16/32
    shuffle=False,
    collate_fn=sft_collate_fn
)

batch = next(iter(sft_train_loader))

for k, v in batch.items():
    print(k, v.shape)

batch["labels"][0]

"""# SFT"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

sft_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
sft_model.resize_token_embeddings(len(tokenizer))
sft_model.to(device)

optimizer = torch.optim.AdamW(sft_model.parameters(), lr=5e-5)

num_epochs = 10
num_training_steps = num_epochs * len(sft_train_loader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * num_training_steps),
    num_training_steps=num_training_steps
)

best_val_loss = float("inf")

for epoch in range(1, num_epochs + 1):
    print(f"\n===== SFT Epoch {epoch}/{num_epochs} =====")
    sft_model.train()
    total_loss = 0.0

    pbar = tqdm(sft_train_loader)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = sft_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_train_loss = total_loss / len(sft_train_loader)
    print(f"Train Loss = {avg_train_loss:.4f}")

    sft_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in sft_valid_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = sft_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(sft_valid_loader)
    print(f"Valid Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        sft_model.save_pretrained(SFT_DIR)
        tokenizer.save_pretrained(SFT_DIR)
        print(f"New best model saved to {SFT_DIR} (val_loss = {best_val_loss:.4f})")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(SFT_DIR, padding_side="right")
tok.pad_token = tok.eos_token


base_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
base_model.eval()

sft_model = AutoModelForCausalLM.from_pretrained(SFT_DIR).to(device)
sft_model.eval()

@torch.no_grad()
def generate_one(model, prompt, max_new_tokens=25):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

prompt = "The movie was"

print("===== Base distilgpt2 =====")
for i in range(3):
    print(f"[Sample {i+1}] {generate_one(base_model, prompt)}\n")

print("===== SFT model =====")
for i in range(3):
    print(f"[Sample {i+1}] {generate_one(sft_model, prompt)}\n")

import torch
from transformers import AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def decode_example_from_batch(batch, idx):
    input_ids = batch["input_ids"][idx]
    labels    = batch["labels"][idx]


    target_mask = labels != -100


    target_ids = input_ids[target_mask]
    target_text = tokenizer.decode(target_ids, skip_special_tokens=True)


    if target_mask.any():
        first_target_pos = torch.where(target_mask)[0][0].item()
    else:
        first_target_pos = len(input_ids)

    prompt_ids = input_ids[:first_target_pos]
    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

    return prompt_text, target_text

@torch.no_grad()
def generate_continuation(model, prompt_text, max_new_tokens=50):
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    full_ids = out[0]
    prompt_len = enc["input_ids"].size(1)
    gen_cont_ids = full_ids[prompt_len:]
    gen_cont_text = tokenizer.decode(gen_cont_ids, skip_special_tokens=True)
    return gen_cont_text

from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bertscore
def eval_metrics(model, data_loader, max_batches=200, use_bertscore=False):
    model.eval()

    smoothie = SmoothingFunction().method3
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    all_references = []
    all_hypotheses = []

    rouge1_f = rouge2_f = rougel_f = 0.0
    meteor_sum = 0.0
    sample_count = 0

    gen_len_sum = 0
    rep_ratio_sum = 0
    total_unigrams = 0
    total_bigrams = 0
    unigram_set = set()
    bigram_set = set()

    ref_texts = []
    hyp_texts = []

    pbar = tqdm(data_loader, desc="Eval", total=max_batches)

    for b_idx, batch in enumerate(pbar):
        if b_idx >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        bs = batch["input_ids"].size(0)

        for i in range(bs):
            prompt_text, target_text = decode_example_from_batch(batch, i)
            if len(target_text.strip()) == 0:
                continue

            gen_text = generate_continuation(model, prompt_text)


            ref_tokens = target_text.split()
            hyp_tokens = gen_text.split()
            if len(hyp_tokens) == 0:
                continue

            # 1) BLEU
            all_references.append([ref_tokens])
            all_hypotheses.append(hyp_tokens)

            # 2) ROUGE
            scores = rouge.score(target_text, gen_text)
            rouge1_f += scores["rouge1"].fmeasure
            rouge2_f += scores["rouge2"].fmeasure
            rougel_f += scores["rougeL"].fmeasure

            # 3) METEOR
            meteor_sum += meteor_score([ref_tokens], hyp_tokens)

            # 4) average length
            L = len(hyp_tokens)
            gen_len_sum += L

            # 5)  (1 - unique / len)
            unique_tokens = len(set(hyp_tokens))
            rep_ratio = 1.0 - unique_tokens / max(L, 1)
            rep_ratio_sum += rep_ratio

            # 6) distinct-1 / distinct-2
            unigram_set.update(hyp_tokens)
            total_unigrams += L

            if L > 1:
                bigrams = list(zip(hyp_tokens, hyp_tokens[1:]))
                bigram_set.update(bigrams)
                total_bigrams += len(bigrams)

            # 7) BERTScore
            if use_bertscore:
                ref_texts.append(target_text)
                hyp_texts.append(gen_text)

            sample_count += 1

        pbar.set_postfix({"pairs": sample_count})

    bleu   = corpus_bleu(all_references, all_hypotheses, smoothing_function=smoothie)
    rouge1 = rouge1_f / max(sample_count, 1)
    rouge2 = rouge2_f / max(sample_count, 1)
    rougel = rougel_f / max(sample_count, 1)
    meteor_avg = meteor_sum / max(sample_count, 1)

    avg_len   = gen_len_sum / max(sample_count, 1)
    avg_rep   = rep_ratio_sum / max(sample_count, 1)
    distinct1 = len(unigram_set) / max(total_unigrams, 1)
    distinct2 = len(bigram_set)  / max(total_bigrams, 1)

    metrics = {
        "bleu": bleu,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougel": rougel,
        "meteor": meteor_avg,
        "avg_len": avg_len,
        "avg_rep": avg_rep,
        "distinct1": distinct1,
        "distinct2": distinct2,
        "samples": sample_count,
    }

    # ====== BERTScore ======
    if use_bertscore and len(hyp_texts) > 0:
        P, R, F1 = bertscore(hyp_texts, ref_texts, lang="en", verbose=False)
        metrics["bertscore_f1"] = F1.mean().item()

    return metrics

base_metrics = eval_metrics(base_model, sft_test_loader, max_batches=200, use_bertscore=True)
sft_metrics  = eval_metrics(sft_model,  sft_test_loader, max_batches=200, use_bertscore=True)

print("=== Base ===")
for k, v in base_metrics.items():
    print(f"{k:12s}: {v}")

print("\n=== SFT ===")
for k, v in sft_metrics.items():
    print(f"{k:12s}: {v}")

"""# RL dataloader"""

from torch.utils.data import DataLoader

def rl_collate_fn(batch):
    prompts = [ex["prompt"] for ex in batch]
    return {"prompts": prompts}

batch_size_rl = 4

rl_train_loader = DataLoader(
    rl_train,
    batch_size=batch_size_rl,
    shuffle=True,
    collate_fn=rl_collate_fn,
)

rl_valid_loader = DataLoader(
    rl_valid,
    batch_size=batch_size_rl,
    shuffle=False,
    collate_fn=rl_collate_fn,
)

rl_test_loader = DataLoader(
    rl_test,
    batch_size=batch_size_rl,
    shuffle=False,
    collate_fn=rl_collate_fn,
)

batch = next(iter(rl_train_loader))
batch["prompts"]

"""# Policy"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyValueModel(nn.Module):
    def __init__(self, lm_name="distilgpt2"):
        super().__init__()

        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        hidden_size = self.lm.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask=None):


        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        logits = outputs.logits                    # [B, T, V]
        hidden = outputs.hidden_states[-1]         # [B, T, H]
        values = self.value_head(hidden).squeeze(-1)  # [B, T]
        return logits, values

from transformers import AutoModelForCausalLM

def create_reference_model(lm_name="distilgpt2"):
    ref = AutoModelForCausalLM.from_pretrained(lm_name)
    ref.eval()
    for param in ref.parameters():
        param.requires_grad = False
    return ref

device = "cuda" if torch.cuda.is_available() else "cpu"

policy = PolicyValueModel(SFT_DIR).to(device)
ref_lm = create_reference_model(SFT_DIR).to(device)

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

test_prompt = ["This movie was surprisingly"]
enc = tokenizer(test_prompt, return_tensors="pt").to(device)

print(enc["input_ids"].shape)
print(enc["attention_mask"].shape)

logits, values = policy(enc["input_ids"], enc["attention_mask"])

print("logits:", logits.shape)
print("values:", values)

"""# Reward"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

sent_tok = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
sent_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
).to(device)
sent_model.eval()

@torch.no_grad()
def sentiment_reward(prompts, responses):
    texts = [p + " " + r for p, r in zip(prompts, responses)]
    enc = sent_tok(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)

    outputs = sent_model(**enc)
    probs = outputs.logits.softmax(dim=-1)
    pos = probs[:, 1]  # positive class probability
    return pos  # shape [B]

def repetition_penalty(responses, alpha=0.5):
    scores = []
    for txt in responses:
        toks = txt.split()
        if len(toks) == 0:
            scores.append(0.0)
            continue

        # ratio of repeated tokens (NOT n-grams)
        unique = len(set(toks))
        rep_ratio = 1 - (unique / len(toks))

        # return negative (penalty)
        scores.append(-alpha * rep_ratio)

    return torch.tensor(scores, device=device)

def completeness_reward(responses):
    scores = []
    for txt in responses:
        if txt.strip().endswith((".", "!", "?")):
            scores.append(1.0)

        else:
            scores.append(0.0)

    return torch.tensor(scores, device=device)

def short_penalty(responses, min_len=5, alpha=0.2):
    scores = []
    for txt in responses:
        L = len(txt.split())
        if L < min_len:
            scores.append(-(min_len - L) * alpha)
        else:
            scores.append(0.0)
    return torch.tensor(scores, device=device)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)
nli_model.eval()

@torch.no_grad()
def coherence_reward(responses):
    scores = []
    for txt in responses:

        sentences = [s.strip() for s in txt.split(".") if s.strip()]
        if len(sentences) < 2:
            scores.append(1.0)
            continue

        ok = 1.0
        for i in range(len(sentences)-1):
            enc = nli_tok(sentences[i], sentences[i+1], return_tensors="pt", truncation=True).to(device)
            logits = nli_model(**enc).logits
            # MNLI labels: 0=contradiction, 1=neutral, 2=entailment
            pred = torch.argmax(logits, dim=-1).item()
            if pred == 0:
                ok -= 0.3


        scores.append(max(ok, 0.0))

    return torch.tensor(scores, device=device)

from sentence_transformers import SentenceTransformer
import torch

rel_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

@torch.no_grad()
def relevance_reward(prompts, responses):
    e1 = rel_model.encode(prompts, convert_to_tensor=True, device=device)
    e2 = rel_model.encode(responses, convert_to_tensor=True, device=device)
    sim = torch.nn.functional.cosine_similarity(e1, e2)
    return sim.clamp(0,1)    # shape [B]

def compute_reward(prompts, responses):
    R_rm     = sentiment_reward(prompts, responses)
    R_rep    = repetition_penalty(responses)
    R_short  = short_penalty(responses)
    R_comp   = completeness_reward(responses)
    R_rel    = relevance_reward(prompts, responses)
    R_coh    = coherence_reward(responses)

    R = (
        0.35  * R_rm +
        0.10 * R_rep +
        0.10 * R_short +
        0.10 * R_comp +
        0.25 * R_rel +
        0.35 * R_coh
    )

    return R

@torch.no_grad()
def generate_with_logprobs(policy, tokenizer, prompts, max_new_tokens=32):

    # 1. encode prompt
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)
    prompt_input_ids = enc["input_ids"]
    prompt_attn_mask = enc["attention_mask"]

    prompt_lens = prompt_attn_mask.sum(dim=1).tolist()

    # 2. model generate
    gen_ids = policy.lm.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    # 3. forward LM to get logprobs
    input_ids = gen_ids
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    logits, _ = policy(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    log_probs = logits.log_softmax(dim=-1)

    batch_size = input_ids.size(0)
    responses = []
    gen_tokens_list = []
    old_logprobs_list = []
    for i in range(batch_size):
      pl = prompt_lens[i]
      full_ids = input_ids[i]
      L = full_ids.size(0)

      gen_tokens = full_ids[pl:]
      gen_tokens_list.append(gen_tokens)

      # logprobs slicing
      lp = log_probs[i, pl-1:L-1, :]
      lp_token = lp.gather(
          dim=-1,
          index=gen_tokens.unsqueeze(-1)
      ).squeeze(-1)

      old_logprobs_list.append(lp_token)


      resp_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
      responses.append(resp_text)

    return {
        "responses": responses,
        "gen_tokens": gen_tokens_list,
        "old_logprobs": old_logprobs_list,
        "input_ids": input_ids,
        "prompt_lens": prompt_lens,
    }



"""
  Prompt 1: "The movie was"
  Prompt 2: "I feel very"

  {
  "responses": ["amazing!", "happy today"],
  "gen_tokens": [
        tensor([21045,  2901,  50256]),
        tensor([ 1098,  3401])
  ],
  "old_logprobs": [
        tensor([-5.20, -4.11, -6.88]),
        tensor([-3.33, -7.77])
  ],
  "input_ids": [
        tensor([p0, p1, p2, p3, 21045, 2901, 50256]),
        tensor([q0, q1, q2, 1098, 3401])
  ],
  "prompt_lens": [4, 3]
}"""

batch = next(iter(rl_train_loader))
prompts = batch["prompts"]

roll = generate_with_logprobs(policy, tokenizer, prompts)

print("=== Prompt ===")
print(prompts[0])
print("=== Response ===")
print(roll["responses"][0])
print("=== Old logprobs len ===")
print(len(roll["old_logprobs"][0]))

R = compute_reward(prompts, roll["responses"])
print(R)

"""# PPO Optimization"""

def rollout_generate(policy, tokenizer, prompts):
    return generate_with_logprobs(policy, tokenizer, prompts)

def rollout_reward(prompts, responses):
    return compute_reward(prompts, responses)

def compute_values_seq(policy, tokenizer, roll):
    input_ids = roll["input_ids"]
    attn_mask = (input_ids != tokenizer.pad_token_id).long()

    logits, values_all = policy(input_ids=input_ids, attention_mask=attn_mask)

    prompt_lens = roll["prompt_lens"]
    B = input_ids.size(0)

    values_seq = []
    for i in range(B):
        pl = prompt_lens[i]
        gen_len = len(roll["gen_tokens"][i])


        v = values_all[i, pl - 1 : pl - 1 + gen_len]

        values_seq.append(v)

    return values_seq

def compute_advantages_seq(rewards, values_seq):
    advantages = []
    for R_i, V_seq in zip(rewards, values_seq):
        A_i = (R_i - V_seq).detach()
        advantages.append(A_i)
    return advantages

def compute_new_logprobs(policy, tokenizer, roll):
    input_ids = roll["input_ids"]
    attn_mask = (input_ids != tokenizer.pad_token_id).long()

    logits, _ = policy(input_ids=input_ids, attention_mask=attn_mask)
    log_probs = logits.log_softmax(dim=-1)


    prompt_lens = roll["prompt_lens"]
    batch_size = input_ids.size(0)

    new_lp_list = []

    for i in range(batch_size):
        pl = prompt_lens[i]
        gen_tokens = roll["gen_tokens"][i].to(log_probs.device)

        lp = log_probs[i, pl-1:-1, :]  # [T_gen, vocab_size]

        lp_token = lp.gather(
            dim=-1,
            index=gen_tokens.unsqueeze(-1)
        ).squeeze(-1)

        new_lp_list.append(lp_token)

    return new_lp_list

@torch.no_grad()
def compute_ref_logprobs(ref_lm, tokenizer, roll):
    input_ids = roll["input_ids"]
    attn_mask = (input_ids != tokenizer.pad_token_id).long()

    logits = ref_lm(input_ids=input_ids, attention_mask=attn_mask).logits
    log_probs = logits.log_softmax(dim=-1)

    prompt_lens = roll["prompt_lens"]
    batch_size = input_ids.size(0)

    ref_lp_list = []

    for i in range(batch_size):
        pl = prompt_lens[i]

        gen_tokens = roll["gen_tokens"][i].to(log_probs.device)


        lp = log_probs[i, pl-1:-1, :]     # [T_gen, vocab]


        lp_token = lp.gather(
            dim=-1,
            index=gen_tokens.unsqueeze(-1)
        ).squeeze(-1)

        ref_lp_list.append(lp_token)

    return ref_lp_list

class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, alpha):
        self.kl_coef = init_kl_coef
        self.target = target
        self.alpha = alpha

    def update(self, current_kl: float):
        if current_kl != current_kl:
            return self.kl_coef

        ratio = current_kl / (self.target + 1e-8)
        change = (ratio - 1.0) * self.alpha


        self.kl_coef *= (1.0 + change)


        self.kl_coef = max(1e-4, min(self.kl_coef, 1.0))
        return self.kl_coef

def compute_ppo_loss(
    old_lps, new_lps, ref_lps,
    advantages, rewards, values_seqs,
    clip_eps=0.2, value_coef=0.1, kl_coef=0.1, entropy_coef=0.01,
):
    policy_losses = []
    value_losses  = []
    kl_terms      = []
    entropies     = []

    B = len(old_lps)

    for i in range(B):
        old_lp = old_lps[i]        # [T_gen]
        new_lp = new_lps[i]        # [T_gen]
        ref_lp = ref_lps[i]        # [T_gen]

        A_seq = advantages[i]      # [T_gen]
        V_seq = values_seqs[i]     # [T_gen]
        R_i   = rewards[i]         # scalar

        # ---------- PPO ratio ----------
        ratio = (new_lp - old_lp).exp()

        obj1 = ratio * A_seq
        obj2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * A_seq
        policy_loss_i = -torch.min(obj1, obj2).mean()

        # ---------- Value loss ----------
        value_loss_i = (V_seq - R_i).pow(2).mean()

        # ---------- ★ Surrogate KL (Stable) ----------
        # KL = mean( (new_lp - ref_lp)^2 )
        kl_i = (new_lp - ref_lp).pow(2).mean()

        # ---------- Entropy ----------
        entropy_i = -new_lp.mean()

        policy_losses.append(policy_loss_i)
        value_losses.append(value_loss_i)
        kl_terms.append(kl_i)
        entropies.append(entropy_i)

    # ---------- combine ----------
    policy_loss = torch.stack(policy_losses).mean()
    value_loss  = torch.stack(value_losses).mean()
    kl_loss     = torch.stack(kl_terms).mean()
    entropy     = torch.stack(entropies).mean()

    total_loss = (
        policy_loss +
        value_coef * value_loss +
        kl_coef * kl_loss -
        entropy_coef * entropy
    )

    stats = {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_loss": kl_loss.item(),
        "entropy": entropy.item(),
    }

    return total_loss, stats

def ppo_step(policy, optimizer, total_loss):
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

def ppo_update_single_batch(
    policy,
    ref_lm,
    tokenizer,
    roll,
    rewards,
    optimizer,
    kl_coef,
    device="cuda",
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):


    # 1) old & new logprobs
    old_lps = roll["old_logprobs"]
    new_lps = compute_new_logprobs(policy, tokenizer, roll)
    ref_lps = compute_ref_logprobs(ref_lm, tokenizer, roll)

    # 2) values
    values_seqs = compute_values_seq(policy, tokenizer, roll)

    # 3) advantages (seq-level reward)
    advantages = compute_advantages_seq(rewards, values_seqs)


    advantages = [(A - A.mean()) / (A.std() + 1e-8) for A in advantages]

    # 4) PPO loss
    total_loss, stats = compute_ppo_loss(
        old_lps,
        new_lps,
        ref_lps,
        advantages,
        rewards,
        values_seqs,
        clip_eps=clip_eps,
        value_coef=value_coef,
        kl_coef=kl_coef,
        entropy_coef=entropy_coef,
    )


    input_ids = roll["input_ids"].to(device)              # [B, T]
    attn_mask = (input_ids != tokenizer.pad_token_id).long()
    prompt_lens = roll["prompt_lens"]                     # list[int]


    labels = input_ids.clone()
    labels[:] = -100
    for i, pl in enumerate(prompt_lens):

        labels[i, pl:] = input_ids[i, pl:]
    labels[attn_mask == 0] = -100

    lm_outputs = policy.lm(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
    )
    lm_loss = lm_outputs.loss


    lm_coef = 0.05
    total_loss = total_loss + lm_coef * lm_loss

    stats["lm_loss"] = lm_loss.item()
    stats["total_loss"] = total_loss.item()


    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    stats["loss"] = stats["total_loss"]
    stats["reward"] = rewards.mean().item()

    return stats

from tqdm import tqdm

def train_ppo_rlhf(
    policy,
    ref_lm,
    tokenizer,
    rl_train_loader,
    rl_valid_loader,
    epochs=6,
    lr=3e-6,
    ppo_epochs=4,
    save_dir="checkpoints",
    device="cuda"
):

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    kl_ctrl = AdaptiveKLController(
        init_kl_coef=0.01,
        target=1,
        alpha=0.05
    )

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch} / {epochs} =====")

        policy.train()
        train_stats = {"reward": 0, "kl": 0, "loss": 0}
        steps = 0

        # ---------------- add tqdm wrapper ----------------
        pbar = tqdm(rl_train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            prompts = batch["prompts"]

            # 1) Rollout
            policy.eval()
            roll = generate_with_logprobs(policy, tokenizer, prompts)
            rewards = compute_reward(prompts, roll["responses"])

            # 2) PPO epochs
            policy.train()
            for _ in range(ppo_epochs):
                stats = ppo_update_single_batch(
                    policy=policy,
                    ref_lm=ref_lm,
                    tokenizer=tokenizer,
                    roll=roll,
                    rewards=rewards,
                    optimizer=optimizer,
                    kl_coef=kl_ctrl.kl_coef,
                )

                # update KL controller
                kl_ctrl.update(stats["kl_loss"])

            # ---------------- show metrics on tqdm ----------------
            pbar.set_postfix({
                "R":   f"{stats['reward']:.3f}",
                "KL":  f"{stats['kl_loss']:.3f}",
                "Loss":f"{stats['total_loss']:.3f}",
                "KLc": f"{kl_ctrl.kl_coef:.4f}",
            })

            # accumulate stats
            train_stats["reward"] += stats["reward"]
            train_stats["kl"]     += stats["kl_loss"]
            train_stats["loss"]   += stats["total_loss"]
            steps += 1


        print(f"\n[Train] Epoch summary {epoch}/{epochs}")
        print(f" avg_reward = {train_stats['reward']/steps:.4f}")
        print(f" avg_KL     = {train_stats['kl']/steps:.4f}")
        print(f" avg_loss   = {train_stats['loss']/steps:.4f}")
        print(f" KL_coef    = {kl_ctrl.kl_coef:.4f}")

        # ---------- Validation ----------
        policy.eval()
        val_rewards = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in rl_valid_loader:
                prompts = batch["prompts"]
                roll = generate_with_logprobs(policy, tokenizer, prompts)
                rewards = compute_reward(prompts, roll["responses"])
                val_rewards += rewards.mean().item()
                val_steps += 1

        print(f"[Valid] avg reward = {val_rewards / val_steps:.4f}")

        # ---------- Save ----------
        save_path = os.path.join(save_dir, f"ppo_policy_epoch{epoch}.pt")
        torch.save(policy.state_dict(), save_path)
        print(f"✔ Model saved to {save_path}")

from transformers.utils import logging
logging.set_verbosity_error()

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

ppo_policy = load_policy(1,CKPT_DIR)

ppo_metrics = eval_metrics(
    ppo_policy.lm,
    sft_test_loader,
    max_batches=200,
    use_bertscore=True
)

print("\n=== PPO Epoch 1 (RLHF) ===")
for k, v in ppo_metrics.items():
    print(f"{k:12s}: {v}")

ppo_policy = load_policy(2,CKPT_DIR)

ppo_metrics = eval_metrics(
    ppo_policy.lm,
    sft_test_loader,
    max_batches=200,
    use_bertscore=True
)

print("\n=== PPO Epoch 2 (RLHF) ===")
for k, v in ppo_metrics.items():
    print(f"{k:12s}: {v}")

ppo_policy = load_policy(3,CKPT_DIR)

ppo_metrics = eval_metrics(
    ppo_policy.lm,
    sft_test_loader,
    max_batches=200,
    use_bertscore=True
)

print("\n=== PPO Epoch 3 (RLHF) ===")
for k, v in ppo_metrics.items():
    print(f"{k:12s}: {v}")

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

# === SFT ===
# bleu        : 0.0023955856689617926
# rouge1      : 0.21807061043818923
# rouge2      : 0.017736503192633018
# rougel      : 0.13266596028034056
# meteor      : 0.10540464840973089
# avg_len     : 42.74625
# avg_rep     : 0.18454676423067326
# distinct1   : 0.11610667602421265
# distinct2   : 0.48960984519567624
# samples     : 1600
# bertscore_f1: 0.8339571356773376

from tqdm import tqdm

@torch.no_grad()
def eval_avg_reward(lm_model, rl_loader, max_batches=None, max_new_tokens=32):
    lm_model.eval()

    total_R = 0.0
    total_n = 0

    pbar = tqdm(rl_loader, desc="Eval reward")

    for b_idx, batch in enumerate(pbar):
        if max_batches is not None and b_idx >= max_batches:
            break

        prompts = batch["prompts"]   # list[str]


        responses = [
            generate_continuation(lm_model, p, max_new_tokens=max_new_tokens)
            for p in prompts
        ]


        R = compute_reward(prompts, responses)  # tensor[B]

        total_R += R.sum().item()
        total_n += len(prompts)

        pbar.set_postfix({"avg_R": total_R / max(total_n, 1)})

    return total_R / max(total_n, 1)

from transformers import AutoModelForCausalLM


sft_lm = AutoModelForCausalLM.from_pretrained(SFT_DIR).to(device)
sft_lm.eval()

sft_test_avg_R = eval_avg_reward(
    sft_lm,
    rl_test_loader,
    max_batches=200,
    max_new_tokens=32
)

print("SFT avg reward on RL test set:", sft_test_avg_R)

ppo_e1 = load_policy(1, CKPT_DIR)     # PolicyValueModel
ppo_e1_avg_R = eval_avg_reward(
    ppo_e1.lm,
    rl_test_loader,
    max_batches=200,
    max_new_tokens=32
)
print("PPO epoch 1 avg reward on RL test set:", ppo_e1_avg_R)



ppo_e2 = load_policy(2, CKPT_DIR)
ppo_e2_avg_R = eval_avg_reward(
    ppo_e2.lm,
    rl_test_loader,
    max_batches=200,
    max_new_tokens=32
)
print("PPO epoch 2 avg reward on RL test set:", ppo_e2_avg_R)


ppo_e3 = load_policy(3, CKPT_DIR)
ppo_e3_avg_R = eval_avg_reward(
    ppo_e3.lm,
    rl_test_loader,
    max_batches=200,
    max_new_tokens=32
)
print("PPO epoch 3 avg reward on RL test set:", ppo_e3_avg_R)



ppo_e3 = load_policy(4, CKPT_DIR)
ppo_e3_avg_R = eval_avg_reward(
    ppo_e3.lm,
    rl_test_loader,
    max_batches=200,
    max_new_tokens=32
)
print("PPO epoch 4 avg reward on RL test set:", ppo_e3_avg_R)
