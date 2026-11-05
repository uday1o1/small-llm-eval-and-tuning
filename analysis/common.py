"""Shared constants and helpers for summarize.py and figures.py."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BASE_MODELS = ["gpt2-124M", "SmolLM2-135M", "SmolLM2-360M", "Qwen2.5-0.5B", "Qwen3-0.6B-Base", "Qwen3.5-0.8B-Base"]
HF_ID = {"gpt2-124M": "gpt2", "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M", "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
         "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B", "Qwen3-0.6B-Base": "Qwen/Qwen3-0.6B-Base",
         "Qwen3.5-0.8B-Base": "Qwen/Qwen3.5-0.8B-Base"}
CROSSCHECK_DIR = {m: HF_ID[m].split("/")[-1] for m in BASE_MODELS}
# the score each task is reported with (as in nanolm's laptop table)
TASKS = {"arc_easy": ("acc_norm", "ARC-Easy"), "arc_challenge": ("acc_norm", "ARC-Challenge"),
         "hellaswag": ("acc_norm", "HellaSwag"), "mmlu": ("acc", "MMLU"), "gsm8k": ("exact_match", "GSM8K")}
GSM8K_FILTER = "flexible-extract"
SEEDS = [1337, 42, 1, 2, 3]
SFT_SIZES = [500, 2000, 8000]
DPO_BETAS = ["0.05", "0.1", "0.5"]


def items(cfg: str, model: str) -> pd.DataFrame:
    """Per-question results of one model; MMLU subjects are pooled into task 'mmlu'."""
    d = pd.read_csv(RUNS / "benchmarks" / cfg / model / "items.csv.gz")
    d["sub"] = d["task"]                                   # MMLU subject (doc ids restart per subject)
    d["task"] = d["task"].where(~d["task"].str.startswith("mmlu_"), "mmlu")
    d = d[(d.task != "gsm8k") | (d["filter"] == GSM8K_FILTER)]
    return d


def task_scores(d: pd.DataFrame) -> dict:
    """{task: (per-question correctness array, per-question chance array)} in a fixed question order."""
    out = {}
    for t, (metric, _) in TASKS.items():
        s = d[d.task == t].sort_values(["sub", "doc_id"])
        if not len(s):
            continue
        y = s[metric].to_numpy(float)
        chance = (1.0 / s["n_choices"]).to_numpy(float) if t != "gsm8k" else np.zeros(len(s))
        out[t] = (y, chance)
    return out


def se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def vs_chance(y, chance) -> dict:
    """Accuracy against random guessing. The number of hits under guessing is a sum of independent
    Bernoulli(1/K_i) (exact for equal K); z uses its mean and variance (n >= 100 here)."""
    n, hits = len(y), float(y.sum())
    mu, var = float(chance.sum()), float((chance * (1 - chance)).sum())
    z = (hits - mu) / math.sqrt(var) if var > 0 else float("inf") if hits > mu else 0.0
    return {"n": n, "acc": hits / n, "se": se(hits / n, n), "chance": mu / n, "z_vs_chance": z}


def mcnemar(a, b) -> dict:
    """Paired comparison of two per-question correctness vectors (same questions, same order):
    exact two-sided binomial test on the discordant questions."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = len(a)
    p = binomtest(n01, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    d = (n01 - n10) / n
    sed = math.sqrt(max(n10 + n01 - (n01 - n10) ** 2 / n, 0)) / n          # SE of the paired difference
    return {"diff": d, "se_diff": sed, "a_only": n10, "b_only": n01, "p": float(p)}


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=False, default=float) + "\n")
