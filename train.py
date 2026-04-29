"""
InsightRank: XAI Resume Auditor
================================
train.py — Fine-tuning pipeline for Cross-Encoder semantic matching
Base model : cross-encoder/ms-marco-MiniLM-L-6-v2
Dataset    : netsol/resume-score-details (HuggingFace) + synthetic augmentation
Task       : Regression (0-1 relevance score) + Binary classification head
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "model_name"   : "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "max_length"   : 512,
    "batch_size"   : 8,
    "epochs"       : 5,
    "lr"           : 2e-5,
    "warmup_ratio" : 0.1,
    "weight_decay" : 0.01,
    "seed"         : 42,
    "val_split"    : 0.15,
    "save_dir"     : "./model_output",
    "device"       : "cuda" if torch.cuda.is_available() else "cpu",
    "score_threshold": 0.5,     # binary classification threshold
    "pos_threshold" : 0.70,     # strong match
    "neg_threshold" : 0.30,     # critical gap
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG["seed"])
log.info(f"Using device: {CFG['device']}")

# ── Dataset Loading ─────────────────────────────────────────────────────────────

def load_hf_dataset() -> pd.DataFrame:
    """
    Load netsol/resume-score-details from HuggingFace.
    Expected columns: resume_text, job_description, score (0-100 or float 0-1)
    Falls back to synthetic data if unavailable.
    """
    try:
        log.info("Loading HuggingFace dataset: netsol/resume-score-details")
        ds = load_dataset("netsol/resume-score-details", split="train")
        df = ds.to_pandas()
        log.info(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

        # Normalise column names
        col_map = {}
        for c in df.columns:
            lc = c.lower().replace(" ", "_")
            if "resume" in lc and "text" in lc:      col_map[c] = "resume_text"
            elif "job" in lc or "description" in lc: col_map[c] = "job_description"
            elif "score" in lc:                       col_map[c] = "score"
        df.rename(columns=col_map, inplace=True)

        # Normalise score to [0, 1]
        if "score" in df.columns:
            max_s = df["score"].max()
            if max_s > 1.0:
                df["score"] = df["score"] / 100.0
        else:
            df["score"] = 0.5

        # Drop rows missing critical columns
        req = ["resume_text", "job_description", "score"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns after mapping: {missing}")

        df = df[req].dropna()
        df = df[df["resume_text"].str.len() > 50]
        return df

    except Exception as e:
        log.warning(f"HF dataset failed ({e}). Generating synthetic training data.")
        return _generate_synthetic_data(n=800)


def _generate_synthetic_data(n: int = 800) -> pd.DataFrame:
    """
    Synthetic resume + JD pairs with realistic score distribution.
    Used as fallback or augmentation when primary dataset is unavailable.
    """
    skills_pool = [
        "Python", "Java", "React", "Node.js", "Docker", "Kubernetes",
        "AWS", "GCP", "Azure", "SQL", "MongoDB", "TensorFlow", "PyTorch",
        "NLP", "Machine Learning", "Deep Learning", "CI/CD", "Git",
        "REST API", "GraphQL", "Agile", "Spark", "Hadoop", "Linux",
        "FastAPI", "Flask", "Data Science", "Computer Vision", "LLM",
    ]
    roles = ["Software Engineer", "Data Scientist", "ML Engineer",
             "Backend Developer", "DevOps Engineer", "Full Stack Developer"]

    rows = []
    for _ in range(n):
        jd_skills  = random.sample(skills_pool, k=random.randint(5, 10))
        overlap_k  = random.randint(0, len(jd_skills))
        res_skills = jd_skills[:overlap_k] + random.sample(
            [s for s in skills_pool if s not in jd_skills],
            k=random.randint(0, 5)
        )
        score = round(overlap_k / len(jd_skills) + random.gauss(0, 0.05), 3)
        score = float(np.clip(score, 0.0, 1.0))
        role  = random.choice(roles)

        jd_text = (
            f"We are hiring a {role}. "
            f"Required skills: {', '.join(jd_skills)}. "
            f"Experience with cloud platforms and agile methodologies preferred. "
            f"Strong communication and problem-solving skills required."
        )
        res_text = (
            f"Experienced {role} with proficiency in {', '.join(res_skills)}. "
            f"{'Worked on cloud-based solutions. ' if 'AWS' in res_skills or 'GCP' in res_skills else ''}"
            f"Delivered multiple production-level projects with cross-functional teams."
        )
        rows.append({"resume_text": res_text, "job_description": jd_text, "score": score})

    log.info(f"Generated {n} synthetic training pairs.")
    return pd.DataFrame(rows)

# ── Dataset Class ──────────────────────────────────────────────────────────────

class ResumeJDDataset(Dataset):
    """
    Tokenises (job_description, resume_text) pairs for Cross-Encoder.
    Labels: continuous [0,1] for regression OR binary for classification.
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int, mode: str = "regression"):
        self.pairs      = list(zip(df["job_description"].tolist(), df["resume_text"].tolist()))
        self.labels     = df["score"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.mode       = mode

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, d = self.pairs[idx]
        enc  = self.tokenizer(
            q, d,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : label,
        }

# ── Model ──────────────────────────────────────────────────────────────────────

class CrossEncoderRegressor(nn.Module):
    """
    Wraps AutoModelForSequenceClassification (num_labels=1) for regression.
    Optionally adds a classification head for binary plus/gap prediction.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder    = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, ignore_mismatched_sizes=True
        )
        hidden_size     = self.encoder.config.hidden_size
        self.clf_head   = nn.Linear(hidden_size, 2)  # binary: match / no-match
        self.dropout    = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, output_hidden=False):
        out       = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
        reg_score = torch.sigmoid(out.logits.squeeze(-1))  # [B]
        cls_score = None
        if output_hidden:
            pooled    = out.hidden_states[-1][:, 0, :]     # [CLS] token
            cls_score = self.clf_head(self.dropout(pooled)) # [B, 2]
        return reg_score, cls_score

# ── Loss ───────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """MSE regression loss + optional cross-entropy on binary labels."""
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.mse  = nn.MSELoss()
        self.ce   = nn.CrossEntropyLoss()
        self.alpha = alpha  # weight for regression loss

    def forward(self, reg_pred, label, clf_pred=None):
        loss = self.alpha * self.mse(reg_pred, label)
        if clf_pred is not None:
            bin_label = (label > CFG["score_threshold"]).long()
            loss += (1 - self.alpha) * self.ce(clf_pred, bin_label)
        return loss

# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(preds: list, labels: list) -> dict:
    preds_arr  = np.array(preds)
    labels_arr = np.array(labels)
    bin_preds  = (preds_arr  >= CFG["score_threshold"]).astype(int)
    bin_labels = (labels_arr >= CFG["score_threshold"]).astype(int)

    acc = accuracy_score(bin_labels, bin_preds)
    f1  = f1_score(bin_labels, bin_preds, zero_division=0)
    pcc, _ = pearsonr(preds_arr, labels_arr)
    scc, _ = spearmanr(preds_arr, labels_arr)
    mse = float(np.mean((preds_arr - labels_arr) ** 2))

    return {"accuracy": acc, "f1": f1, "pearson": pcc, "spearman": scc, "mse": mse}

# ── Training Loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device, use_clf=True):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)

        optimizer.zero_grad()
        reg_pred, clf_pred = model(input_ids, attn_mask, output_hidden=use_clf)
        loss = loss_fn(reg_pred, labels, clf_pred)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(reg_pred.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device, use_clf=True):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels    = batch["label"].to(device)

        reg_pred, clf_pred = model(input_ids, attn_mask, output_hidden=use_clf)
        loss = loss_fn(reg_pred, labels, clf_pred)

        total_loss += loss.item()
        all_preds.extend(reg_pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CFG["save_dir"], exist_ok=True)

    # 1. Load data
    df = load_hf_dataset()
    log.info(f"Dataset size: {len(df)} | Score range: {df['score'].min():.3f} – {df['score'].max():.3f}")

    train_df, val_df = train_test_split(df, test_size=CFG["val_split"],
                                        random_state=CFG["seed"], shuffle=True)
    log.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # 2. Tokenizer + datasets
    tokenizer  = AutoTokenizer.from_pretrained(CFG["model_name"])
    train_ds   = ResumeJDDataset(train_df, tokenizer, CFG["max_length"])
    val_ds     = ResumeJDDataset(val_df,   tokenizer, CFG["max_length"])
    train_dl   = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=0)
    val_dl     = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=0)

    # 3. Model
    model = CrossEncoderRegressor(CFG["model_name"]).to(CFG["device"])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {total_params:,}")

    # 4. Optimizer + scheduler + loss
    optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    loss_fn   = CombinedLoss(alpha=0.8)

    # 5. Training loop
    best_val_f1 = 0.0
    history = []

    for epoch in range(1, CFG["epochs"] + 1):
        log.info(f"\n{'='*60}")
        log.info(f"EPOCH {epoch}/{CFG['epochs']}  |  LR: {scheduler.get_last_lr()}")

        train_m = train_epoch(model, train_dl, optimizer, loss_fn, CFG["device"])
        val_m   = eval_epoch(model, val_dl, loss_fn, CFG["device"])
        scheduler.step()

        log.info(f"TRAIN  loss={train_m['loss']:.4f}  acc={train_m['accuracy']:.4f}  "
                 f"f1={train_m['f1']:.4f}  pearson={train_m['pearson']:.4f}")
        log.info(f"VAL    loss={val_m['loss']:.4f}  acc={val_m['accuracy']:.4f}  "
                 f"f1={val_m['f1']:.4f}  pearson={val_m['pearson']:.4f}")

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        # Save best model
        if val_m["f1"] >= best_val_f1:
            best_val_f1 = val_m["f1"]
            model.encoder.save_pretrained(CFG["save_dir"])
            tokenizer.save_pretrained(CFG["save_dir"])
            log.info(f"✓ Best model saved (val_f1={best_val_f1:.4f})")

    # 6. Save training history
    hist_rows = []
    for h in history:
        hist_rows.append({
            "epoch"       : h["epoch"],
            "train_loss"  : h["train"]["loss"],
            "train_acc"   : h["train"]["accuracy"],
            "train_f1"    : h["train"]["f1"],
            "train_pearson": h["train"]["pearson"],
            "val_loss"    : h["val"]["loss"],
            "val_acc"     : h["val"]["accuracy"],
            "val_f1"      : h["val"]["f1"],
            "val_pearson" : h["val"]["pearson"],
        })
    pd.DataFrame(hist_rows).to_csv(f"{CFG['save_dir']}/training_history.csv", index=False)
    log.info(f"\nTraining complete. Model saved to: {CFG['save_dir']}")

if __name__ == "__main__":
    main()