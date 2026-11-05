"""Per-choice log-likelihoods on ARC-Easy (test, 2,376 questions), scored exactly as
nanolm.benchmarks.mc_accuracy scores them, with both normalizations recorded (GPU).

    python scripts/arc_scores.py HuggingFaceTB/SmolLM2-360M runs/crosscheck/SmolLM2-360M

For every question and choice the CSV holds the summed log-probability of the choice tokens, the
number of characters of the choice (without the separating space, as mc_accuracy and lm_eval
count them) and the number of tokens the choice adds after the context. Before writing the JSON
summary (the CSV is written first), the script checks that its accuracies on the first 300
questions equal nanolm's own mc_accuracy.
"""
import csv
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanolm.benchmarks import arc_items, mc_accuracy


@torch.no_grad()
def choice_scores(model, tok, it, dev):
    """The loop body of mc_accuracy, returning every quantity it uses."""
    out = []
    n_ctx = len(tok(it["context"], add_special_tokens=False)["input_ids"])
    for ch in it["choices"]:
        full = tok(it["context"] + ch, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([full], device=dev)
        logp = model(ids).logits.float().log_softmax(-1)[0, n_ctx - 1:-1]
        s = logp.gather(-1, ids[0, n_ctx:, None]).sum().item()
        out.append({"loglik": s, "chars": len(ch.lstrip(" ")), "tokens": len(full) - n_ctx,
                    "ctx_prefix_ok": int(full[:n_ctx] == tok(it["context"], add_special_tokens=False)["input_ids"])})
    return out


def main(name, out_dir):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
    items = arc_items()
    rows = []
    for q, it in enumerate(items):
        for c, r in enumerate(choice_scores(model, tok, it, dev)):
            rows.append({"question": q, "choice": c, "label": it["label"], **r})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "arc_easy_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def acc(n, norm):
        hits = 0
        for q in range(n):
            rs = by_q[q]
            key = [r["loglik"] / (r[norm] if norm else 1) for r in rs]
            hits += int(max(range(len(key)), key=key.__getitem__) == rs[0]["label"])
        return hits / n

    by_q = {}
    for r in rows:
        by_q.setdefault(r["question"], []).append(r)
    ref = mc_accuracy(model, tok, items[:300], dev)
    mine = {"acc": acc(300, None), "acc_norm": acc(300, "chars")}
    assert mine["acc"] == ref["acc"] and mine["acc_norm"] == ref["acc_norm"], (mine, ref)
    summary = {"model": name, "n_questions": len(items), "first300_mc_accuracy": ref,
               "full": {"acc": acc(len(items), None), "acc_norm_chars": acc(len(items), "chars"),
                        "acc_norm_tokens": acc(len(items), "tokens")},
               "first300_acc_norm_tokens": acc(300, "tokens")}
    (out / "arc_easy_scores.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:3])
