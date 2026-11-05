"""Training tokens of every SFT and DPO run (no GPU): nanolm's data selection and batch order, replayed.

    python scripts/token_budget.py runs/token_budget.json

For each run, the examples nanolm's training loop draws (finetune.py: a Random(seed) permutation of the
training set, popped from the end, a new permutation appended whenever fewer than one batch is left) are
replayed with nanolm's own data code, and their tokens counted: all tokens of each sequence (prompt and
answer, before padding) and the answer tokens alone (the tokens with a loss). DPO counts the chosen and
the rejected sequence of every pair.
"""
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

from nanolm.config import FTRunConfig, load_config
from nanolm.finetune import IGNORE, load_examples

SEEDS = [1337, 42, 1, 2, 3]
RUNS = {"sft": [("n500", ["data.n_train=500"]), ("n2000", ["data.n_train=2000"]), ("n8000", ["data.n_train=8000"])],
        "dpo": [("beta0.05", []), ("beta0.1", []), ("beta0.5", [])]}     # beta does not change the data


def drawn(n_train, tc, seed):
    """Indices of the training examples nanolm's loop uses, in order (finetune.py)."""
    rng, order, out = random.Random(seed), [], []
    for _ in range(tc.steps):
        for _ in range(tc.grad_accum):
            if len(order) < tc.batch_size:
                order += rng.sample(range(n_train), n_train)
            out += [order.pop() for _ in range(tc.batch_size)]
    return out


def seqs(ex, kind):
    return [ex] if kind == "sft" else [ex["chosen"], ex["rejected"]]


def main(out):
    res = {}
    for kind, runs in RUNS.items():
        for name, over in runs:
            cfg = load_config(f"configs/finetune/{kind}.yaml", over, cls=FTRunConfig)
            tok = AutoTokenizer.from_pretrained(cfg.model.name)
            train, _ = load_examples(cfg, tok)
            for seed in SEEDS:
                idx = drawn(len(train), cfg.train, seed)
                s = [q for i in idx for q in seqs(train[i], kind)]
                res[f"{kind}/{name}_s{seed}"] = {
                    "examples_drawn": len(idx), "distinct_examples": len(set(idx)), "n_train": len(train),
                    "tokens": sum(len(q["input_ids"]) for q in s),
                    "answer_tokens": sum(sum(t != IGNORE for t in q["labels"]) for q in s)}
            print(kind, name, res[f"{kind}/{name}_s{SEEDS[0]}"])
    Path(out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
