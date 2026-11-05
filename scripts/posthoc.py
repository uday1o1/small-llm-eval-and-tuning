"""Measurements on the tuned models after training (GPU), with nanolm's own losses and data code.

    python scripts/posthoc.py sft runs/sft/n2000_s1337      # one adapter
    python scripts/posthoc.py sft runs/sft/base             # the untuned base model (no adapter)

For every run directory it writes posthoc.json and generations.jsonl:
  * FineWeb bits per byte of the tuned model (nanolm.finetune.eval_bpb, as for the base models);
  * greedy answers to held-out prompts (the first 50 held-out examples of the run's own split, never
    trained on), with the answer length and whether the answer ended with the end-of-turn token;
  * DPO only (the untuned model is its own reference, so its rewards are zero and only the policy's
    own preference is informative): implicit-reward accuracy
    and margin on every held-out pair that nanolm's filter keeps from the test split (asking for 500
    returns all 240: the run's 100 validation pairs first, then the other 140), and how far the
    policy moved the per-token log-probability of the chosen and of the rejected answers away from
    the reference.
"""
import json
import sys
from pathlib import Path

import torch

from nanolm.config import FTRunConfig, load_config
from nanolm.device import get_device
from nanolm.finetune import IGNORE, collate, dpo_terms, eval_bpb, load_examples, token_logprobs

N_PAIRS, N_PROMPTS, GEN_TOKENS, BS = 500, 50, 256, 25   # N_PAIRS: more than the split has


def load(kind, run_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg_file = Path("configs/finetune") / f"{kind}.yaml"
    cfg = load_config(str(cfg_file), [], cls=FTRunConfig)
    adapter = Path(run_dir) / "adapter"
    if adapter.exists():
        cfg = load_config(str(Path(run_dir) / "config.yaml"), [], cls=FTRunConfig)
    dinfo = get_device(cfg.device, cfg.precision)
    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, dtype=torch.float32).to(dinfo.device)
    if adapter.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return cfg, dinfo, tok, model.eval()


def prompt_ids(tok, p):
    """The prompt exactly as nanolm.finetune formats it for training and sampling."""
    if tok.chat_template:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)
    return tok.encode(f"### Instruction:\n{p}\n\n### Response:\n")


@torch.no_grad()
def generate(model, tok, prompts, dev):
    eot = tok.eos_token_id
    out = []
    for i in range(0, len(prompts), BS):
        batch = [prompt_ids(tok, p) for p in prompts[i: i + BS]]
        T = max(map(len, batch))
        ids = torch.tensor([[tok.pad_token_id] * (T - len(b)) + b for b in batch], device=dev)   # left padding
        att = torch.tensor([[0] * (T - len(b)) + [1] * len(b) for b in batch], device=dev)
        gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=GEN_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id)[:, T:].tolist()
        for p, g in zip(prompts[i: i + BS], gen):
            ended = eot in g
            g = g[: g.index(eot)] if ended else g
            out.append({"prompt": p, "answer": tok.decode(g, skip_special_tokens=True), "tokens": len(g), "ended": ended})
    return out


def prompts_of(kind, examples, tok):
    out = []
    for e in examples[:N_PROMPTS]:
        if kind == "dpo":
            out.append(e["prompt"])
        else:   # SFT examples hold token ids only: recover the instruction text from the prompt tokens
            n = next(i for i, l in enumerate(e["labels"]) if l != IGNORE)
            text = tok.decode(e["input_ids"][:n])
            out.append(text.split("### Instruction:\n", 1)[1].rsplit("\n\n### Response:", 1)[0])
    return out


@torch.no_grad()
def dpo_heldout(model, tok, cfg, val, dev, tuned):
    rows = []
    for i in range(0, len(val), 4):
        chunk = val[i: i + 4]
        w = collate([e["chosen"] for e in chunk], tok.pad_token_id, dev)
        l = collate([e["rejected"] for e in chunk], tok.pad_token_id, dev)
        pw, nw = token_logprobs(model, *w)
        pl, nl = token_logprobs(model, *l)
        if tuned:                          # the reference is the same network with the adapter off
            with model.disable_adapter():
                rw, _ = token_logprobs(model, *w)
                rl, _ = token_logprobs(model, *l)
        else:                              # the untuned model is its own reference
            rw, rl = pw, pl
        for k in range(len(chunk)):
            rows.append({"logratio_chosen": (pw[k] - rw[k]).item(), "logratio_rejected": (pl[k] - rl[k]).item(),
                         "tokens_chosen": nw[k].item(), "tokens_rejected": nl[k].item(),
                         "policy_logp_chosen": pw[k].item(), "policy_logp_rejected": pl[k].item()})
    b = cfg.train.dpo_beta
    margins = [b * (r["logratio_chosen"] - r["logratio_rejected"]) for r in rows]
    n = len(rows)
    return {"n_pairs": n, "beta": b,
            "reward_acc": sum((m > 0) if abs(m) >= 1e-3 else 0.5 for m in margins) / n,
            "reward_margin": sum(margins) / n,
            "logratio_margin": sum(m / b for m in margins) / n,
            "first100_reward_acc": sum((m > 0) if abs(m) >= 1e-3 else 0.5 for m in margins[:100]) / 100,
            "first100_reward_margin": sum(margins[:100]) / 100,
            "dlogp_per_token_chosen": sum(r["logratio_chosen"] / r["tokens_chosen"] for r in rows) / n,
            "dlogp_per_token_rejected": sum(r["logratio_rejected"] / r["tokens_rejected"] for r in rows) / n,
            "policy_prefers_chosen": sum(r["policy_logp_chosen"] / r["tokens_chosen"] > r["policy_logp_rejected"] / r["tokens_rejected"]
                                         for r in rows) / n}


def main(kind, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg, dinfo, tok, model = load(kind, run_dir)
    tuned = (run_dir / "adapter").exists()
    if kind == "dpo":
        cfg.data.n_val = N_PAIRS
    _, val = load_examples(cfg, tok)
    res = {"run_dir": str(run_dir), "model": cfg.model.name, "adapter": tuned, "n_val_examples": len(val)}
    bpb_cfg = load_config(None, ["data.format=golf_val", "train.method=eval_bpb"], cls=FTRunConfig)
    res.update({"fineweb_" + k: v for k, v in eval_bpb(model, tok, bpb_cfg, dinfo).items()})
    if kind == "dpo":
        res.update(dpo_heldout(model, tok, cfg, val, dinfo.device, tuned))
    gens = generate(model, tok, prompts_of(kind, val, tok), dinfo.device)
    res.update({"gen_n": len(gens), "gen_mean_tokens": sum(g["tokens"] for g in gens) / len(gens),
                "gen_ended_frac": sum(g["ended"] for g in gens) / len(gens)})
    with open(run_dir / "generations.jsonl", "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    (run_dir / "posthoc.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main(*sys.argv[1:3])
