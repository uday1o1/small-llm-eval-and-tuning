"""Answer lengths in the DPO preference pairs (no GPU): how often is the chosen answer the longer one?

    python scripts/pair_lengths.py runs/dpo/pair_lengths.json

Uses nanolm's own data code with the DPO config, so the pairs are exactly the training pairs and the
held-out pairs of the runs (the held-out set as in posthoc.py: every pair the filter keeps).
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from nanolm.config import FTRunConfig, load_config
from nanolm.finetune import IGNORE, load_examples


def answer_tokens(ex):
    return sum(t != IGNORE for t in ex["labels"])


def stats(pairs):
    c = [answer_tokens(p["chosen"]) for p in pairs]
    r = [answer_tokens(p["rejected"]) for p in pairs]
    n = len(pairs)
    return {"n_pairs": n, "chosen_longer": sum(a > b for a, b in zip(c, r)) / n,
            "rejected_longer": sum(a < b for a, b in zip(c, r)) / n,
            "mean_tokens_chosen": sum(c) / n, "mean_tokens_rejected": sum(r) / n}


def main(out):
    cfg = load_config("configs/finetune/dpo.yaml", ["data.n_val=500"], cls=FTRunConfig)
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    train, val = load_examples(cfg, tok)
    res = {"train": stats(train), "heldout": stats(val), "heldout_first100": stats(val[:100])}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
