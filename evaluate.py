"""
InsightRank: XAI Resume Auditor
================================
evaluate.py — Comprehensive evaluation suite
Metrics: Accuracy, Precision, Recall, F1, Pearson, Spearman
         + Sample predictions + Semantic score visualisation
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
EVAL_CFG = {
    "model_dir"       : "./model_output",
    "fallback_model"  : "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "max_length"      : 512,
    "batch_size"      : 16,
    "score_threshold" : 0.5,
    "device"          : "cuda" if torch.cuda.is_available() else "cpu",
    "seed"            : 42,
    "n_samples"       : 10,   # sample predictions to display
}


# ── Data Loading (mirrors train.py) ───────────────────────────────────────────

def load_eval_data(n_samples: int = 200) -> pd.DataFrame:
    """Load eval split from HF dataset or generate synthetic fallback."""
    try:
        ds = load_dataset("netsol/resume-score-details", split="train")
        df = ds.to_pandas()
        col_map = {}
        for c in df.columns:
            lc = c.lower().replace(" ", "_")
            if "resume" in lc and "text" in lc:       col_map[c] = "resume_text"
            elif "job" in lc or "description" in lc:  col_map[c] = "job_description"
            elif "score" in lc:                        col_map[c] = "score"
        df.rename(columns=col_map, inplace=True)
        if "score" in df.columns and df["score"].max() > 1.0:
            df["score"] = df["score"] / 100.0
        df = df[["resume_text", "job_description", "score"]].dropna()
        _, val_df = train_test_split(df, test_size=0.15, random_state=42, shuffle=True)
        return val_df.head(n_samples)
    except Exception as e:
        log.warning(f"HF dataset unavailable ({e}). Using synthetic eval data.")
        return _synthetic_eval(n_samples)


def _synthetic_eval(n: int) -> pd.DataFrame:
    import random
    random.seed(42)
    skills_pool = [
        "Python", "Java", "React", "Docker", "Kubernetes", "AWS",
        "SQL", "TensorFlow", "PyTorch", "NLP", "Machine Learning",
        "CI/CD", "Git", "REST API", "Data Science", "Linux", "FastAPI",
    ]
    rows = []
    for _ in range(n):
        jd_skills  = random.sample(skills_pool, k=random.randint(4, 8))
        overlap    = random.randint(0, len(jd_skills))
        res_skills = jd_skills[:overlap] + random.sample(
            [s for s in skills_pool if s not in jd_skills], k=random.randint(0, 3)
        )
        score = float(np.clip(overlap / len(jd_skills) + np.random.normal(0, 0.05), 0, 1))
        rows.append({
            "resume_text"    : f"Skilled in {', '.join(res_skills)}. Strong delivery track record.",
            "job_description": f"Required: {', '.join(jd_skills)}. Agile environment. Team player.",
            "score"          : round(score, 3),
        })
    return pd.DataFrame(rows)


# ── Inference ──────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, model_dir: str, fallback: str):
        src = model_dir if os.path.isdir(model_dir) else fallback
        log.info(f"Loading model from: {src}")
        self.tokenizer = AutoTokenizer.from_pretrained(src)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            src, num_labels=1, ignore_mismatched_sizes=True
        ).to(EVAL_CFG["device"])
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, pairs: list) -> np.ndarray:
        """pairs: list of (jd_text, resume_text)"""
        enc = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=EVAL_CFG["max_length"],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(EVAL_CFG["device"]) for k, v in enc.items()}
        logits = self.model(**enc).logits.squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy()

    def predict_all(self, df: pd.DataFrame) -> np.ndarray:
        pairs   = list(zip(df["job_description"], df["resume_text"]))
        preds   = []
        bs      = EVAL_CFG["batch_size"]
        for i in range(0, len(pairs), bs):
            preds.extend(self.predict_batch(pairs[i:i+bs]).tolist())
        return np.array(preds)


# ── Metrics ────────────────────────────────────────────────────────────────────

def full_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    bin_true = (y_true >= threshold).astype(int)
    bin_pred = (y_pred >= threshold).astype(int)

    acc  = accuracy_score(bin_true, bin_pred)
    prec = precision_score(bin_true, bin_pred, zero_division=0)
    rec  = recall_score(bin_true, bin_pred, zero_division=0)
    f1   = f1_score(bin_true, bin_pred, zero_division=0)
    pcc, p_pval  = pearsonr(y_pred, y_true)
    scc, s_pval  = spearmanr(y_pred, y_true)
    mse  = float(np.mean((y_pred - y_true) ** 2))
    mae  = float(np.mean(np.abs(y_pred - y_true)))

    return {
        "accuracy"        : round(acc, 4),
        "precision"       : round(prec, 4),
        "recall"          : round(rec, 4),
        "f1_score"        : round(f1, 4),
        "pearson_r"       : round(pcc, 4),
        "pearson_pval"    : round(p_pval, 6),
        "spearman_rho"    : round(scc, 4),
        "spearman_pval"   : round(s_pval, 6),
        "mse"             : round(mse, 6),
        "mae"             : round(mae, 6),
        "bin_true"        : bin_true,
        "bin_pred"        : bin_pred,
    }


def print_report(metrics: dict, df: pd.DataFrame, preds: np.ndarray):
    sep = "=" * 65

    print(f"\n{sep}")
    print("  InsightRank: XAI Resume Auditor — Evaluation Report")
    print(sep)

    print("\n📊 REGRESSION METRICS")
    print(f"   MSE          : {metrics['mse']}")
    print(f"   MAE          : {metrics['mae']}")
    print(f"   Pearson r    : {metrics['pearson_r']}  (p={metrics['pearson_pval']})")
    print(f"   Spearman ρ   : {metrics['spearman_rho']}  (p={metrics['spearman_pval']})")

    print(f"\n🎯 CLASSIFICATION METRICS (threshold={EVAL_CFG['score_threshold']})")
    print(f"   Accuracy     : {metrics['accuracy']}")
    print(f"   Precision    : {metrics['precision']}")
    print(f"   Recall       : {metrics['recall']}")
    print(f"   F1 Score     : {metrics['f1_score']}")

    print("\n📋 DETAILED CLASSIFICATION REPORT")
    print(classification_report(
        metrics["bin_true"], metrics["bin_pred"],
        target_names=["No Match", "Match"], zero_division=0
    ))

    print("🔲 CONFUSION MATRIX")
    cm = confusion_matrix(metrics["bin_true"], metrics["bin_pred"])
    print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"   FN={cm[1,0]}  TP={cm[1,1]}")

    # Score distribution
    labels = df["score"].values
    print("\n📈 SCORE DISTRIBUTION")
    for bracket, lo, hi in [("Strong (>0.70)", 0.70, 1.01),
                             ("Moderate (0.30–0.70)", 0.30, 0.70),
                             ("Weak (<0.30)", 0.00, 0.30)]:
        n_true = int(((labels >= lo) & (labels < hi)).sum())
        n_pred = int(((preds  >= lo) & (preds  < hi)).sum())
        print(f"   {bracket:<25} True: {n_true:>4}  Pred: {n_pred:>4}")

    # Sample predictions
    print(f"\n🔍 SAMPLE PREDICTIONS (n={EVAL_CFG['n_samples']})")
    print(f"   {'#':<4} {'True':>6}  {'Pred':>6}  {'Δ':>6}  {'JD snippet (50 chars)'}")
    print("   " + "-" * 60)
    idx = np.random.choice(len(df), size=min(EVAL_CFG["n_samples"], len(df)), replace=False)
    for rank, i in enumerate(idx, 1):
        true_s = labels[i]
        pred_s = preds[i]
        jd_snip = df["job_description"].iloc[i][:50].replace("\n", " ")
        delta   = pred_s - true_s
        print(f"   {rank:<4} {true_s:>6.3f}  {pred_s:>6.3f}  {delta:>+6.3f}  {jd_snip}")

    # Plus / Gap detection accuracy
    plus_mask  = labels >= 0.70
    gap_mask   = labels <= 0.30
    plus_acc   = accuracy_score((plus_mask).astype(int), (preds >= 0.70).astype(int)) if plus_mask.any() else 0.0
    gap_acc    = accuracy_score((gap_mask).astype(int),  (preds <= 0.30).astype(int)) if gap_mask.any() else 0.0
    print(f"\n✅ Plus-Point Detection Accuracy : {plus_acc:.4f}")
    print(f"❌ Critical-Gap Detection Accuracy: {gap_acc:.4f}")
    print(f"\n{sep}\n")


# ── Skill Extraction Evaluation ────────────────────────────────────────────────

SKILL_PATTERNS = [
    "python", "java", "javascript", "react", "node", "docker", "kubernetes",
    "aws", "gcp", "azure", "sql", "mongodb", "tensorflow", "pytorch", "nlp",
    "machine learning", "deep learning", "ci/cd", "git", "agile", "spark",
    "hadoop", "linux", "fastapi", "flask", "data science", "computer vision",
    "llm", "langchain", "typescript", "rest api", "graphql", "devops",
]

def extract_skills(text: str) -> set:
    text_lower = text.lower()
    return {s for s in SKILL_PATTERNS if s in text_lower}

def eval_skill_extraction(df: pd.DataFrame, n: int = 20):
    print("🛠  SKILL EXTRACTION SAMPLE")
    print(f"   {'#':<4} {'JD Skills':<35}  {'Resume Skills':<35}  Gap")
    print("   " + "-" * 90)
    sample = df.sample(n=min(n, len(df)), random_state=42)
    for rank, (_, row) in enumerate(sample.iterrows(), 1):
        jd_skills  = extract_skills(row["job_description"])
        res_skills = extract_skills(row["resume_text"])
        gaps       = jd_skills - res_skills
        jd_str  = ", ".join(sorted(jd_skills))[:33]
        res_str = ", ".join(sorted(res_skills))[:33]
        gap_str = ", ".join(sorted(gaps))[:28] or "none"
        print(f"   {rank:<4} {jd_str:<35}  {res_str:<35}  {gap_str}")
    print()


# ── Save Results ───────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, preds: np.ndarray, metrics: dict, out_dir: str = "./model_output"):
    os.makedirs(out_dir, exist_ok=True)
    results_df = df.copy()
    results_df["predicted_score"] = preds
    results_df["true_label"]      = (results_df["score"] >= EVAL_CFG["score_threshold"]).astype(int)
    results_df["pred_label"]      = (preds >= EVAL_CFG["score_threshold"]).astype(int)
    results_df["correct"]         = (results_df["true_label"] == results_df["pred_label"]).astype(int)
    results_df.to_csv(f"{out_dir}/eval_predictions.csv", index=False)

    summary = {k: v for k, v in metrics.items() if k not in ("bin_true", "bin_pred")}
    pd.DataFrame([summary]).to_csv(f"{out_dir}/eval_metrics.csv", index=False)
    log.info(f"Results saved to {out_dir}/")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InsightRank Evaluator")
    parser.add_argument("--model_dir",   default=EVAL_CFG["model_dir"])
    parser.add_argument("--n_samples",   type=int, default=200)
    parser.add_argument("--threshold",   type=float, default=EVAL_CFG["score_threshold"])
    parser.add_argument("--save",        action="store_true", default=True)
    args = parser.parse_args()

    EVAL_CFG["score_threshold"] = args.threshold

    # 1. Load eval data
    df = load_eval_data(n_samples=args.n_samples)
    log.info(f"Eval samples: {len(df)}")

    # 2. Load model & predict
    evaluator = Evaluator(args.model_dir, EVAL_CFG["fallback_model"])
    preds     = evaluator.predict_all(df)

    # 3. Compute & print metrics
    metrics = full_metrics(df["score"].values, preds, args.threshold)
    print_report(metrics, df, preds)

    # 4. Skill extraction sample
    eval_skill_extraction(df)

    # 5. Save results
    if args.save:
        save_results(df, preds, metrics)

if __name__ == "__main__":
    main()