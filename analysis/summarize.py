"""Every summary number and paired test, computed from the run records into results/summary.json.

    python analysis/summarize.py          (no GPU)
"""
import glob
import json
import math
import re

import numpy as np
import pandas as pd

from scipy.stats import ttest_1samp, ttest_rel

from common import (DPO_BETAS, SEEDS, SFT_SIZES, BASE_MODELS, CROSSCHECK_DIR, GSM8K_FILTER, HF_ID, ROOT, RUNS, TASKS, items, mcnemar, save,
                    task_scores, vs_chance)

# values quoted from nanolm's tutorial (section 3.5) and laptop quickstart (step 6), for comparison only
TUTORIAL_BPB = {"gpt2-124M": 1.079, "SmolLM2-135M": 1.010, "SmolLM2-360M": 0.928, "Qwen2.5-0.5B": 0.951,
                "Qwen3-0.6B-Base": 0.953, "Qwen3.5-0.8B-Base": 0.988}
QUICKSTART_BPB = {"gpt2-124M": 1.0750, "Qwen2.5-0.5B": 0.9511}
QUICKSTART_LAPTOP = {"SmolLM2-135M": {"arc_challenge": 0.277, "arc_easy": 0.553, "hellaswag": 0.437, "mmlu": 0.243, "gsm8k": 0.02},
                     "SmolLM2-360M": {"arc_challenge": 0.363, "arc_easy": 0.663, "hellaswag": 0.553, "mmlu": 0.262, "gsm8k": 0.05},
                     "Qwen2.5-0.5B": {"arc_challenge": 0.327, "arc_easy": 0.570, "hellaswag": 0.497, "mmlu": 0.495, "gsm8k": 0.34}}


# ------------------------------------------------------------------------------- FineWeb bpb
def bpb():
    out = {}
    for m in BASE_MODELS:
        r = json.loads(open(RUNS / "bpb" / CROSSCHECK_DIR[m] / "metrics.json").read())
        out[m] = {k: r[k] for k in ("val_bpb", "bytes_per_token", "eval_docs", "eval_bytes", "eval_tokens", "params",
                                    "eval_time_sec", "device", "dtype")}
        out[m]["tutorial"] = TUTORIAL_BPB[m]
        if m in QUICKSTART_BPB:
            out[m]["quickstart_mac"] = QUICKSTART_BPB[m]
    return out


# -------------------------------------------------------------------------------- benchmarks
def aggregate(cfg, model):
    """lm_eval's own aggregate scores (results_*.json) for one model, primary metric per task."""
    out = {}
    for f in glob.glob(str(RUNS / "benchmarks" / cfg / model / "*" / "*" / "results_*.json")):
        r = json.loads(open(f).read())["results"]
        for t, (metric, _) in TASKS.items():
            key = f"{metric},{GSM8K_FILTER}" if t == "gsm8k" else f"{metric},none"
            if t in r and key in r[t]:
                out[t] = r[t][key]
    return out


def bench_table(cfg, models):
    out = {}
    for m in models:
        sc = task_scores(items(cfg, m))
        agg = aggregate(cfg, m)
        out[m] = {}
        for t, (y, ch) in sc.items():
            v = vs_chance(y, ch)
            assert abs(v["acc"] - agg[t]) < 1e-9, (cfg, m, t, v["acc"], agg[t])   # per-question records = lm_eval aggregate
            out[m][t] = v
    return out


def pairwise(cfg, models):
    """McNemar test for every pair of models on every task (same questions)."""
    sc = {m: task_scores(items(cfg, m)) for m in models}
    out = {}
    for t in TASKS:
        rows = []
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                r = mcnemar(sc[a][t][0], sc[b][t][0])
                rows.append({"a": a, "b": b, **r})
        out[t] = {"pairs": rows, "resolved_p05": sum(r["p"] < 0.05 for r in rows), "n_pairs": len(rows)}
    # Holm's correction over all tests of the table (every pair on every task)
    allp = sorted((r["p"], t, i) for t in TASKS for i, r in enumerate(out[t]["pairs"]))
    m, stop = len(allp), False
    for k, (pv, t, i) in enumerate(allp):
        stop = stop or pv > 0.05 / (m - k)
        out[t]["pairs"][i]["holm_reject"] = not stop
    for t in TASKS:
        out[t]["resolved_holm"] = sum(r["holm_reject"] for r in out[t]["pairs"])
    return out


def laptop_reproduction():
    out = {}
    for m, ref in QUICKSTART_LAPTOP.items():
        ours = aggregate("laptop", m)
        logged = aggregate("models", m)
        out[m] = {t: {"quickstart_mac": ref[t], "ours": ours[t], "ours_logged_run": logged[t]} for t in ref}
    return out


def cloze_pairs():
    """McNemar on cloze-MMLU acc_norm for every pair of base models (same questions, same order)."""
    y = {m: items("cloze", m).sort_values(["sub", "doc_id"]).acc_norm.to_numpy(float) for m in BASE_MODELS}
    return [{"a": a, "b": b, **mcnemar(y[a], y[b])} for i, a in enumerate(BASE_MODELS) for b in BASE_MODELS[i + 1:]]


def cloze():
    out = {}
    for m in BASE_MODELS:
        f = glob.glob(str(RUNS / "benchmarks" / "cloze" / m / "*" / "*" / "results_*.json"))
        if not f:
            continue
        r = json.loads(open(f[0]).read())["results"]["mmlu_continuation"]
        d = pd.read_csv(RUNS / "benchmarks" / "cloze" / m / "items.csv.gz")
        assert abs(d.acc.mean() - r["acc,none"]) < 1e-9, m      # lm_eval's group acc = mean over all questions
        # lm_eval aggregates only acc for the group; acc_norm is the same mean of the per-question values
        out[m] = {"acc": r["acc,none"], "acc_norm": float(d.acc_norm.mean()), "n": len(d),
                  "acc_vs_chance": vs_chance(d.acc.to_numpy(float), (1 / d.n_choices).to_numpy(float)),
                  "acc_norm_vs_chance": vs_chance(d.acc_norm.to_numpy(float), (1 / d.n_choices).to_numpy(float))}
    return out


# ------------------------------------------------------------------ scoring-rule check (ARC-Easy)
def lm_eval_results(path_glob):
    f = sorted(glob.glob(str(path_glob)))
    assert len(f) == 1, (path_glob, f)
    return json.loads(open(f[0]).read())


def crosscheck():
    out = {}
    for m in BASE_MODELS:
        d = RUNS / "crosscheck" / CROSSCHECK_DIR[m]
        line = open(d / "crosscheck.log").read()
        acc, acc_norm = map(float, re.search(r"acc ([\d.]+)\s+acc_norm ([\d.]+)", line).groups())
        le = lm_eval_results(d / "lm_eval" / "*" / "results_*.json")["results"]["arc_easy"]
        sc = pd.read_csv(d / "arc_easy_scores.csv")
        # lm_eval's per-choice log-likelihoods for the same 300 questions
        samples = [json.loads(l) for l in open(sorted(glob.glob(str(d / "lm_eval" / "*" / "samples_arc_easy_*.jsonl")))[0])]
        le_ll = {(int(s["doc_id"]), c): float(r[0][0] if isinstance(r[0], list) else r[0])
                 for s in samples for c, r in enumerate(_resps(s))}
        first = sc[sc.question < 300]
        diffs = [abs(r.loglik - le_ll[(r.question, r.choice)]) for r in first.itertuples()]
        picks = {}
        sc["chars_with_space"] = sc.chars + 1          # the tutorial's first version counted the separating space
        for norm in ("raw", "chars", "tokens", "chars_with_space"):
            key = sc.loglik / (sc[norm] if norm != "raw" else 1)
            picks[norm] = sc.assign(k=key).loc[lambda x: x.groupby("question").k.idxmax()].set_index("question")
        lab = sc.groupby("question").label.first()
        le_pred = {int(s["doc_id"]): (float(s["acc"]), float(s["acc_norm"])) for s in samples}
        res = {"nanolm_crosscheck": {"acc": acc, "acc_norm": acc_norm},
               "lm_eval": {"acc": le["acc,none"], "acc_norm": le["acc_norm,none"]},
               "max_abs_loglik_diff_first300": max(diffs),
               "per_question_agree_first300": {
                   "acc": int(sum((picks["raw"].choice[q] == lab[q]) == bool(le_pred[q][0]) for q in range(300))),
                   "acc_norm": int(sum((picks["chars"].choice[q] == lab[q]) == bool(le_pred[q][1]) for q in range(300)))}}
        ps = picks["chars_with_space"].choice.loc[list(range(300))].to_numpy()
        pc3 = picks["chars"].choice.loc[list(range(300))].to_numpy()
        y3 = lab.loc[list(range(300))].to_numpy()
        res["with_space_first300"] = {"acc_norm": float((ps == y3).mean()), "differ_from_chars": int((ps != pc3).sum())}
        for scope, n in (("first300", 300), ("full", sc.question.nunique())):
            qs = range(n)
            pc = picks["chars"].choice.loc[list(qs)].to_numpy()
            pt = picks["tokens"].choice.loc[list(qs)].to_numpy()
            y = lab.loc[list(qs)].to_numpy()
            flip = pc != pt
            # characters per token of the choice each rule picked, on the questions where they differ
            cpt = (sc.chars / sc.tokens).to_numpy().reshape(-1)
            idx = {(r.question, r.choice): i for i, r in enumerate(sc.itertuples())}
            cpt_c = np.array([cpt[idx[(q, pc[q])]] for q in qs])
            cpt_t = np.array([cpt[idx[(q, pt[q])]] for q in qs])
            res[scope] = {"n": n, "acc_norm_chars": float((pc == y).mean()), "acc_norm_tokens": float((pt == y).mean()),
                          "flips": int(flip.sum()), "right_to_wrong": int((flip & (pc == y)).sum()),
                          "wrong_to_right": int((flip & (pt == y)).sum()),
                          "flips_to_fewer_chars_per_token": int((flip & (cpt_t < cpt_c)).sum()),
                          "mcnemar": mcnemar(pc == y, pt == y)}
        out[m] = res
    return out


def _resps(s):
    r = s["resps"]
    if isinstance(r, str):
        import ast
        r = ast.literal_eval(r)
    return r


# ------------------------------------------------------------------------- tuning studies
def describe(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": v.tolist()}


def paired(a, b):
    """b - a over matched seeds, with the paired t-test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    p = float(ttest_rel(b, a).pvalue) if d.std() > 0 else (0.0 if d.mean() else 1.0)
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)), "p": p,
            "all_same_sign": bool((d > 0).all() or (d < 0).all())}


def all_pairs(recs, keys, get):
    """Every pair of settings: paired change over seeds, and the noise floor of the two settings compared
    (the median, i.e. the mean, of their two within-setting stds)."""
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            va, vb = [get(x) for x in recs[a]], [get(x) for x in recs[b]]
            d = paired(va, vb)
            d["floor"] = float(np.median([np.std(va, ddof=1), np.std(vb, ddof=1)]))
            d["resolved"] = bool(abs(d["mean_diff"]) >= 2 * d["floor"])
            out[f"{b}-{a}"] = d
    return out


def run_records(kind, key):
    out = []
    for sd in SEEDS:
        r = RUNS / kind / f"{key}_s{sd}"
        m = json.loads(open(r / "metrics.json").read())
        ph = json.loads(open(r / "posthoc.json").read())
        ev = [json.loads(l) for l in open(r / "eval_log.jsonl")]
        gens = [json.loads(l) for l in open(r / "generations.jsonl")]
        out.append({"seed": sd, "metrics": m, "posthoc": ph, "curve": ev, "gens": gens})
    return out


def gen_stats(gens):
    a = [g["answer"] for g in gens]
    return {"ended": float(np.mean([g["ended"] for g in gens])), "tokens": float(np.mean([g["tokens"] for g in gens])),
            "new_section": float(np.mean(["###" in x for x in a])),
            "markdown_bold": float(np.mean(["**" in x for x in a])),
            "headers": float(np.mean([bool(re.search(r"(^|\n)#{1,4} ", x)) for x in a])),
            "list": float(np.mean([bool(re.search(r"(^|\n)\s*([-*•]|\d+\.) ", x)) for x in a])),
            "opener_certainly_sure": float(np.mean([x.strip().lower().startswith(("certainly", "sure", "of course", "absolutely")) for x in a]))}


def vs_base_gens(recs, base_gens, short=20):
    """Per prompt, tuned against untuned (same prompts, same order): the share of answers that got longer;
    and on the prompts the untuned model answers in at most `short` tokens, how often the tuned answer's
    first line starts with the untuned answer (its first 20 characters), how often such a kept answer got
    longer, and how often the tuned answer is over twice as long."""
    first = lambda a: a.strip().split("\n")[0].strip()
    idx = [i for i, g in enumerate(base_gens) if g["tokens"] <= short]
    longer, kept, kept_longer, doubled = [], [], [], []
    for x in recs:
        assert [g["prompt"] for g in x["gens"]] == [g["prompt"] for g in base_gens]
        longer.append(float(np.mean([a["tokens"] > b["tokens"] for a, b in zip(x["gens"], base_gens)])))
        k = [i for i in idx if first(x["gens"][i]["answer"]).startswith(first(base_gens[i]["answer"])[:20])]
        kept.append(len(k))
        kept_longer.append(sum(x["gens"][i]["tokens"] > base_gens[i]["tokens"] for i in k))
        doubled.append(sum(x["gens"][i]["tokens"] > 2 * base_gens[i]["tokens"] for i in idx))
    return {"longer": describe(longer), "n_short": len(idx), "short_kept": kept, "short_kept_longer": kept_longer,
            "short_doubled": doubled}


def bench_deltas(cfg, base_cfg, base_model, names, n_boot=2000):
    """Tuned minus base on every task (same questions): per seed with McNemar, and the seed-mean change with a
    95% interval from resampling the questions (paired bootstrap: per question, the tuned-minus-base
    correctness averaged over the seeds, resampled with replacement)."""
    base = task_scores(items(base_cfg, base_model))
    rng = np.random.default_rng(0)
    out = {}
    for name in names:
        per = {t: [] for t in TASKS}
        dq = {t: [] for t in TASKS}
        for sd in SEEDS:
            tuned = task_scores(items(cfg, f"{name}_s{sd}"))
            agg = aggregate(cfg, f"{name}_s{sd}")
            assert all(abs(tuned[t][0].mean() - agg[t]) < 1e-9 for t in TASKS), (name, sd)   # per-question records = lm_eval aggregate
            for t in TASKS:
                per[t].append(mcnemar(base[t][0], tuned[t][0]))
                dq[t].append(tuned[t][0] - base[t][0])
        out[name] = {}
        for t, v in per.items():
            q = np.mean(dq[t], axis=0)                                   # per question, mean over seeds
            boot = np.array([q[rng.integers(0, len(q), len(q))].mean() for _ in range(n_boot)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            dm = float(np.mean([r["diff"] for r in v]))
            passes = bool(abs(dm) >= 2 * float(np.std([r["diff"] for r in v], ddof=1)) and (lo > 0 or hi < 0))
            out[name][t] = {"diff": describe([r["diff"] for r in v]), "mcnemar_p": [r["p"] for r in v],
                            "passes_rule": passes,
                            "se_diff": [r["se_diff"] for r in v], "n_questions": int(len(q)),
                            "boot_ci95": [float(lo), float(hi)], "boot_se": float(boot.std(ddof=1)),
                            "base_acc": float(base[t][0].mean())}
    return out


def sft():
    out, keys = {}, [f"n{n}" for n in SFT_SIZES]
    recs = {k: run_records("sft", k) for k in keys}
    for k in keys:
        r = recs[k]
        befores = [x["metrics"]["before_val_response_nll"] for x in r]    # the same model; differs only by floating-point nondeterminism
        curve = {}
        for x in r:
            for e in x["curve"]:
                curve.setdefault(e["step"], []).append(e["val_response_nll"])
        best = [min(x["curve"], key=lambda e: e["val_response_nll"]) for x in r]
        out[k] = {"n_train": r[0]["metrics"]["n_train_examples"], "before_nll": describe(befores),
                  "before_nll_spread": max(befores) - min(befores),
                  "best_nll": describe([e["val_response_nll"] for e in best]), "best_step": [e["step"] for e in best],
                  "after_nll": describe([x["metrics"]["after_val_response_nll"] for x in r]),
                  "after_ppl": describe([x["metrics"]["after_val_response_ppl"] for x in r]),
                  "curve": {int(st): describe(v) for st, v in sorted(curve.items())},
                  "fineweb_bpb": describe([x["posthoc"]["fineweb_val_bpb"] for x in r]),
                  "gen": {g: describe([gen_stats(x["gens"])[g] for x in r]) for g in gen_stats(r[0]["gens"])},
                  "train_time_sec": describe([x["metrics"]["train_time_sec"] for x in r]),
                  "peak_mem_mb": describe([x["metrics"]["peak_mem_mb"] for x in r]),
                  "trainable_params": r[0]["metrics"]["trainable_params"], "params": r[0]["metrics"]["params"]}
    ref = "n2000"
    out["vs_n2000"] = {k: paired([x["metrics"]["after_val_response_nll"] for x in recs[ref]],
                                 [x["metrics"]["after_val_response_nll"] for x in recs[k]]) for k in keys if k != ref}
    out["noise_floor"] = float(np.median([out[k]["after_nll"]["std"] for k in keys]))
    out["pairs"] = all_pairs(recs, keys, lambda x: x["metrics"]["after_val_response_nll"])
    base = json.loads(open(RUNS / "sft" / "base" / "posthoc.json").read())
    bg = [json.loads(l) for l in open(RUNS / "sft" / "base" / "generations.jsonl")]
    out["base"] = {"fineweb_bpb": base["fineweb_val_bpb"], "gen": gen_stats(bg)}
    # nanolm's own before/after samples (5 fixed prompts): runs whose answer places Paris in Normandy
    out["paris_in_normandy_runs"] = sum("Normandy" in (RUNS / "sft" / f"{k}_s{sd}" / "samples_after.txt").read_text()
                                        for k in keys for sd in SEEDS)
    out["paris_in_normandy_before"] = "Normandy" in (RUNS / "sft" / "n2000_s1337" / "samples_before.txt").read_text()
    for k in keys:
        out[k]["longer_than_base"] = vs_base_gens(recs[k], bg)
    out["benchmarks"] = bench_deltas("sft", "models", "SmolLM2-360M", keys)
    return out


def dpo():
    out, keys = {}, [f"beta{b}" for b in DPO_BETAS]
    recs = {k: run_records("dpo", k) for k in keys}
    for k in keys:
        r = recs[k]
        curve = {}
        for x in r:
            for e in x["curve"]:
                c = curve.setdefault(e["step"], {"acc": [], "margin": [], "prefers": []})
                c["acc"].append(e["val_reward_acc"]); c["margin"].append(e["val_reward_margin"])
                c["prefers"].append(e["val_policy_prefers_chosen"])
        beta = r[0]["posthoc"]["beta"]
        # the held-out set starts with the run's own 100 validation pairs; re-evaluated in batches of 4 instead of
        # nanolm's 2, they agree up to floating-point differences (different padding per batch; a pair at margin ~0
        # can flip: 0.01 of accuracy)
        rc_acc = max(abs(x["posthoc"]["first100_reward_acc"] - x["metrics"]["after_val_reward_acc"]) for x in r)
        rc_margin = max(abs(x["posthoc"]["first100_reward_margin"] - x["metrics"]["after_val_reward_margin"]) for x in r)
        assert rc_acc <= 0.01 + 1e-9 and rc_margin < 0.005, (k, rc_acc, rc_margin)
        out[k] = {"beta": beta, "n_heldout_pairs": r[0]["posthoc"]["n_pairs"],
                  "recheck_first100_max_diff": {"reward_acc": rc_acc, "reward_margin": rc_margin},
                  "before": {m: describe([x["metrics"][f"before_val_{m}"] for x in r]) for m in ("reward_acc", "reward_margin", "policy_prefers_chosen")},
                  "after_acc": describe([x["metrics"]["after_val_reward_acc"] for x in r]),
                  "after_margin": describe([x["metrics"]["after_val_reward_margin"] for x in r]),
                  "after_margin_over_beta": describe([x["metrics"]["after_val_reward_margin"] / beta for x in r]),
                  "after_prefers": describe([x["metrics"]["after_val_policy_prefers_chosen"] for x in r]),
                  "curve": {int(st): {m: describe(v) for m, v in c.items()} for st, c in sorted(curve.items())},
                  "heldout": {m: describe([x["posthoc"][m] for x in r]) for m in
                                 ("reward_acc", "reward_margin", "logratio_margin", "dlogp_per_token_chosen",
                                  "dlogp_per_token_rejected", "policy_prefers_chosen", "first100_reward_acc", "first100_reward_margin")},
                  "fineweb_bpb": describe([x["posthoc"]["fineweb_val_bpb"] for x in r]),
                  "gen": {g: describe([gen_stats(x["gens"])[g] for x in r]) for g in gen_stats(r[0]["gens"])},
                  "train_time_sec": describe([x["metrics"]["train_time_sec"] for x in r]),
                  "peak_mem_mb": describe([x["metrics"]["peak_mem_mb"] for x in r]),
                  "trainable_params": r[0]["metrics"]["trainable_params"], "params": r[0]["metrics"]["params"]}
    ref = "beta0.1"
    out["vs_beta0.1"] = {k: {m: paired([x["metrics"][f"after_val_{m}"] for x in recs[ref]],
                                       [x["metrics"][f"after_val_{m}"] for x in recs[k]])
                             for m in ("reward_acc", "reward_margin", "policy_prefers_chosen")} for k in keys if k != ref}
    out["vs_beta0.1_heldout"] = {k: {m: paired([x["posthoc"][m] for x in recs[ref]], [x["posthoc"][m] for x in recs[k]])
                                        for m in ("reward_acc", "logratio_margin", "policy_prefers_chosen")} for k in keys if k != ref}
    out["noise_floor_reward_acc"] = float(np.median([out[k]["after_acc"]["std"] for k in keys]))
    out["pairs"] = {m: all_pairs(recs, keys, g) for m, g in (
        ("reward_acc_100", lambda x: x["metrics"]["after_val_reward_acc"]),
        ("reward_margin_100", lambda x: x["metrics"]["after_val_reward_margin"]),
        ("policy_prefers_chosen_100", lambda x: x["metrics"]["after_val_policy_prefers_chosen"]),
        ("reward_acc_heldout", lambda x: x["posthoc"]["reward_acc"]),
        ("logratio_margin_heldout", lambda x: x["posthoc"]["logratio_margin"]),
        ("policy_prefers_chosen_heldout", lambda x: x["posthoc"]["policy_prefers_chosen"]),
        ("answer_tokens", lambda x: gen_stats(x["gens"])["tokens"]))}
    out["noise_floor_reward_acc_heldout"] = float(np.median([out[k]["heldout"]["reward_acc"]["std"] for k in keys]))
    base = json.loads(open(RUNS / "dpo" / "base" / "posthoc.json").read())
    bg = [json.loads(l) for l in open(RUNS / "dpo" / "base" / "generations.jsonl")]
    out["base"] = {"fineweb_bpb": base["fineweb_val_bpb"], "policy_prefers_chosen_heldout": base["policy_prefers_chosen"],
                   "gen": gen_stats(bg)}
    # the policy's own preference after tuning against the untuned model's (a fixed value): one-sample t-test
    out["prefers_vs_untuned"] = {k: {"mean_diff": float(np.mean([x["posthoc"]["policy_prefers_chosen"] for x in recs[k]]) - base["policy_prefers_chosen"]),
                                     "p": float(ttest_1samp([x["posthoc"]["policy_prefers_chosen"] for x in recs[k]], base["policy_prefers_chosen"]).pvalue)}
                                 for k in keys}
    for k in keys:
        out[k]["longer_than_base"] = vs_base_gens(recs[k], bg)
    out["pair_lengths"] = json.loads(open(RUNS / "dpo" / "pair_lengths.json").read())
    out["dlogp_rejected_over_chosen"] = {k: out[k]["heldout"]["dlogp_per_token_rejected"]["mean"] /
                                         out[k]["heldout"]["dlogp_per_token_chosen"]["mean"] for k in keys}
    out["benchmarks"] = bench_deltas("dpo", "models", "Qwen2.5-0.5B-Instruct", keys)
    return out


# --------------------------------------------------------------- first 300 questions vs full set
def subset_vs_full():
    """Is the first-300 subset (first 300 per MMLU subject, first 100 of GSM8K) representative of the
    whole evaluation set (HellaSwag: validation)? Accuracy on both, and the subset's distance from the full value in units of the
    subset's standard error (finite-population corrected: the subset is part of the full set)."""
    out = {}
    for m in BASE_MODELS:
        a, b = task_scores(items("models", m)), task_scores(items("full", m))
        out[m] = {}
        for t in TASKS:
            n, N = len(a[t][0]), len(b[t][0])
            pa, pb = float(a[t][0].mean()), float(b[t][0].mean())
            fpc = math.sqrt((N - n) / (N - 1)) if N > 1 else 0.0
            s = math.sqrt(pb * (1 - pb) / n) * fpc
            out[m][t] = {"n_subset": n, "n_full": N, "subset": pa, "full": pb, "diff": pa - pb,
                         "z": (pa - pb) / s if s > 0 else 0.0}
    return out


def mmlu_letters():
    """MMLU (letters A to D after the listed choices): how often each model picks its most frequent letter,
    and how often that letter is the correct one. A model that always picks one letter scores that letter's
    share of the gold answers, which can sit below 1/4."""
    out = {}
    for cfg, models in (("full", BASE_MODELS), ("models", ["SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct"])):
        for m in models:
            d = items(cfg, m)
            d = d[d.task == "mmlu"]
            assert ((d.pred == d.gold) == (d.acc == 1)).all(), m        # pred = the choice lm_eval's acc scores
            share = d.pred.value_counts(normalize=True)
            top = int(share.idxmax())
            out[m] = {"scope": cfg, "n": int(len(d)), "acc": float(d.acc.mean()), "top_letter": "ABCD"[top],
                      "n_correct": int(d.acc.sum()), "n_gold_top_letter": int((d.gold == top).sum()),
                      "top_letter_share": float(share.max()),
                      "pred_share": {"ABCD"[int(k)]: float(v) for k, v in share.sort_index().items()},
                      "gold_share": {"ABCD"[int(k)]: float(v) for k, v in d.gold.value_counts(normalize=True).sort_index().items()}}
    return out


def hellaswag_sources():
    """HellaSwag acc_norm by source corpus. The first 300 validation questions are all ActivityNet;
    the full set adds the WikiHow questions."""
    out = {}
    for m in BASE_MODELS:
        d = items("full", m)
        h = d[d.task == "hellaswag"].sort_values("doc_id")
        out[m] = {"first300_sources": sorted(h.head(300).source.unique().tolist()),
                  **{s: {"n": int(len(g)), "acc_norm": float(g.acc_norm.mean())} for s, g in h.groupby("source")},
                  "activitynet_after_first300": float(h.iloc[300:].query("source == 'activitynet'").acc_norm.mean())}
    return out


# ------------------------------------------------------------------- publisher-reported scores
# Copied from the model cards and technical reports (percent), with the settings the source states.
# SmolLM2: HF model cards (lighteval, 0-shot except GSM8K 5-shot; the cards' "ARC (Average)", compared with
# the mean of our ARC-Easy and ARC-Challenge; "MMLU (cloze)"). Qwen2.5-0.5B: Qwen2.5 technical report
# (arXiv:2412.15115v2) Table 5; Qwen2.5-0.5B-Instruct: the same report's Table 10. Qwen3-0.6B-Base: Qwen3
# technical report (arXiv:2505.09388v1) Table 8. GPT-2 and Qwen3.5-0.8B-Base publish none of these.
MODEL_CARDS = {
    "SmolLM2-135M": {"arc_avg": (43.9, "0-shot"), "hellaswag": (42.1, "0-shot"), "mmlu_cloze": (31.5, "0-shot, cloze"), "gsm8k": (1.4, "5-shot")},
    "SmolLM2-360M": {"arc_avg": (53.0, "0-shot"), "hellaswag": (54.5, "0-shot"), "mmlu_cloze": (35.8, "0-shot, cloze"), "gsm8k": (3.2, "5-shot")},
    "SmolLM2-360M-Instruct": {"arc_avg": (43.7, "0-shot"), "hellaswag": (52.1, "0-shot"), "mmlu_cloze": (32.8, "0-shot, cloze"), "gsm8k": (7.43, "5-shot")},
    "Qwen2.5-0.5B": {"arc_challenge": (35.6, "25-shot"), "hellaswag": (52.1, "10-shot"), "mmlu": (47.5, "5-shot"), "gsm8k": (41.6, "4-shot")},
    "Qwen2.5-0.5B-Instruct": {"gsm8k": (49.6, "shots not stated")},
    "Qwen3-0.6B-Base": {"mmlu": (52.81, "5-shot"), "gsm8k": (59.59, "4-shot, chain of thought")},
}


def model_cards(full, subset, cl):
    """Our score next to each published one: every question for base models (the first-300 subset for
    the instruct models, which ran only there); ARC average = mean of Easy and Challenge; MMLU cloze
    = lm_eval's mmlu_continuation acc_norm."""
    out = {}
    for m, pub in MODEL_CARDS.items():
        src = full.get(m) or subset[m]
        out[m] = {}
        for k, (v, setting) in pub.items():
            if k == "arc_avg":
                ours = (src["arc_easy"]["acc"] + src["arc_challenge"]["acc"]) / 2
            elif k == "mmlu_cloze":
                ours = cl[m]["acc_norm"] if m in cl else None
            else:
                ours = src[k]["acc"]
            scope = "full" if m in full else "first100" if k == "gsm8k" else "first300"
            out[m][k] = {"published": v / 100, "setting": setting,
                         "ours": ours, "ours_scope": "full (cloze)" if k == "mmlu_cloze" else scope,
                         "gap": None if ours is None else ours - v / 100}
    return out


# ------------------------------------------------- publisher settings, run on the same questions
def cards():
    """The publishers' settings, and the chat template with and without chain of thought for the instruct models
    (configs/benchmarks/cards.yaml), against our setting on
    the same questions (McNemar). Each model's groups in cards.yaml run different tasks, so one per-question
    table per model holds them all; GSM8K is scored with lm_eval's flexible-extract filter throughout."""
    def gsm(cfg, m, task, n=None):
        d = pd.read_csv(RUNS / "benchmarks" / cfg / m / "items.csv.gz")
        d = d[(d.task == task) & (d["filter"] == GSM8K_FILTER)].sort_values("doc_id")
        return d.exact_match.to_numpy(float)[:n]
    tests = {"Qwen2.5-0.5B ARC-Challenge 25-shot vs 0-shot": ("Qwen2.5-0.5B", "arc_challenge", ("cards", "arc_challenge"), ("full", None)),
             "Qwen3-0.6B-Base MMLU 5-shot vs 0-shot": ("Qwen3-0.6B-Base", "mmlu", ("cards", "mmlu"), ("full", None)),
             "SmolLM2-360M-Instruct GSM8K chat template vs none": ("SmolLM2-360M-Instruct", "gsm8k", ("cards", "gsm8k"), ("models", None)),
             "Qwen2.5-0.5B-Instruct GSM8K chat template vs none": ("Qwen2.5-0.5B-Instruct", "gsm8k", ("cards", "gsm8k"), ("models", None))}
    for m in ("SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct"):
        tests[f"{m} GSM8K chat template with chain of thought vs none"] = (m, "gsm8k", ("cards", "gsm8k_cot"), ("models", None))
    for m in ("Qwen2.5-0.5B", "Qwen3-0.6B-Base"):
        tests[f"{m} GSM8K 4-shot vs 5-shot"] = (m, "gsm8k", ("cards", "gsm8k"), ("full", 300))
        tests[f"{m} GSM8K 4-shot chain of thought vs 5-shot"] = (m, "gsm8k", ("cards", "gsm8k_cot"), ("full", 300))
    out = {}
    for name, (m, t, (c1, t1), (c0, n0)) in tests.items():
        if t == "gsm8k":
            y1, y0 = gsm(c1, m, t1), gsm(c0, m, "gsm8k", n0)
        else:
            y1, y0 = task_scores(items(c1, m))[t][0], task_scores(items(c0, m))[t][0]
        assert len(y1) == len(y0), name
        out[name] = {"model": m, "task": t, "n": int(len(y1)), "publisher_setting": float(y1.mean()),
                     "ours": float(y0.mean()), **mcnemar(y0, y1)}
    return out


def gsm8k_filters():
    """GSM8K exact match under lm_eval's two answer filters (strict-match needs the '#### <number>' format,
    flexible-extract takes the last number), for the base and instruct models and every tuned adapter."""
    def acc(cfg, m, flt):
        d = pd.read_csv(RUNS / "benchmarks" / cfg / m / "items.csv.gz")
        return float(d[(d.task == "gsm8k") & (d["filter"] == flt)].exact_match.mean())
    out = {}
    for cfg, names in (("models", BASE_MODELS + ["SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct"]),
                       ("sft", [f"n{n}_s{s}" for n in SFT_SIZES for s in SEEDS]),
                       ("dpo", [f"beta{b}_s{s}" for b in DPO_BETAS for s in SEEDS])):
        for m in names:
            out[f"{cfg}/{m}"] = {f: acc(cfg, m, f) for f in ("strict-match", "flexible-extract")}
    return out


# ------------------------------------------------------------------------------- token budget
def token_budget():
    """Training tokens per run (scripts/token_budget.py) and 6ND with N = all parameters."""
    tb = json.loads(open(RUNS / "token_budget.json").read())
    out = {}
    for kind, keys in (("sft", [f"n{n}" for n in SFT_SIZES]), ("dpo", [f"beta{b}" for b in DPO_BETAS])):
        for k in keys:
            r = [tb[f"{kind}/{k}_s{sd}"] for sd in SEEDS]
            n_params = json.loads(open(RUNS / kind / f"{k}_s{SEEDS[0]}" / "metrics.json").read())["params"]
            out[f"{kind}/{k}"] = {"tokens": describe([x["tokens"] for x in r]),
                                  "answer_tokens": describe([x["answer_tokens"] for x in r]),
                                  "distinct_examples": describe([x["distinct_examples"] for x in r]),
                                  "examples_drawn": r[0]["examples_drawn"], "params": n_params,
                                  "flops_6ND": 6 * n_params * float(np.mean([x["tokens"] for x in r]))}
    return out


# ------------------------------------------------------------------------- architecture
def architecture():
    r = json.loads(open(ROOT / "results" / "architecture" / "architecture.json").read())
    out = {}
    for name, v in r.items():
        c = v["config"]
        kv = c["kv_bytes_per_token_bf16"]
        meas = v["measured"]
        # what the cache holds beyond the KV of the full-attention layers: the fixed linear-attention state
        const = [x["state_bytes"] - kv * x["T"] for x in meas]
        assert max(const) == min(const), (name, const)            # measured cache = kv * T + constant, at every T
        fixed = const[0]
        state = lambda T: kv * T + fixed
        if c["n_linear_attention"]:
            # transformers keeps the Gated DeltaNet recurrent state in fp32 and the convolution state at the
            # full kernel width in bf16; with that, config.json predicts the measured constant exactly
            f = glob.glob(str(ROOT / "results" / "architecture" / f"{name.replace('/', '__')}@*.config.json"))[0]
            tc = json.loads(open(f).read()).get("text_config")
            conv_dim = 2 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"] + tc["linear_num_value_heads"] * tc["linear_value_head_dim"]
            per_layer = c["recurrent_state_elems_per_layer"] * 4 + conv_dim * tc["linear_conv_kernel_dim"] * 2
            assert per_layer * c["n_linear_attention"] == fixed, (name, per_layer, fixed)
            fixed_detail = {"recurrent_bytes_per_layer_fp32": c["recurrent_state_elems_per_layer"] * 4,
                            "conv_bytes_per_layer_bf16": conv_dim * tc["linear_conv_kernel_dim"] * 2,
                            "conv_kernel": tc["linear_conv_kernel_dim"], "conv_dim": conv_dim}
        else:
            fixed_detail = {}
        p = v["params_meta"]
        assert p == v["params_safetensors"], name                 # meta-device model = released weights
        out[name] = {"sha": v["sha"], "params": p, "params_total": sum(p.values()),
                     "params_non_embedding": sum(p.values()) - p["embedding"] - p["lm_head"],
                     "config": c, "kv_bytes_per_token": kv, "fixed_state_bytes": fixed, "fixed_state_detail": fixed_detail,
                     "measured": meas,
                     "state_bytes_32k": state(32768), "state_bytes_262k": state(262144),
                     "attention_flops_per_token_per_context_token": c["attention_flops_per_token_per_context_token"]}
    h, f = out["Qwen/Qwen3.5-2B"], out["Qwen/Qwen3-1.7B"]
    out["ratio_full_over_hybrid"] = {"kv_per_token": f["kv_bytes_per_token"] / h["kv_bytes_per_token"],
                                     "state_32k": f["state_bytes_32k"] / h["state_bytes_32k"],
                                     "state_262k": f["state_bytes_262k"] / h["state_bytes_262k"],
                                     "attention_flops": f["attention_flops_per_token_per_context_token"] / h["attention_flops_per_token_per_context_token"]}
    # context length at which the hybrid's growing KV cache equals its fixed linear-attention state
    out["hybrid_kv_equals_fixed_state_at_T"] = h["fixed_state_bytes"] / h["kv_bytes_per_token"]
    # the hybrid with all its layers full attention (same heads): KV per token x n_layers / n_full
    c = h["config"]
    out["hybrid_if_all_full"] = {"kv_bytes_per_token": h["kv_bytes_per_token"] * c["n_layers"] // c["n_full_attention"]}
    out["hybrid_if_all_full"]["state_bytes_262k"] = out["hybrid_if_all_full"]["kv_bytes_per_token"] * 262144
    out["hybrid_if_all_full"]["ratio_to_hybrid_262k"] = out["hybrid_if_all_full"]["state_bytes_262k"] / h["state_bytes_262k"]
    # per generated token at context T: weight matmuls (2 FLOPs per weight of every layer and of the
    # output head; the input embedding is a lookup) against the attention over the cache (4 x heads x
    # head dim per full-attention layer per cached token: q.k and the weighted sum of v)
    for r in (h, f):
        mm = 2 * (r["params_non_embedding"] + r["params"]["embedding"])     # output head = V x d either way
        r["decode_flops_per_token"] = {str(T): {"weights": mm, "attention": r["attention_flops_per_token_per_context_token"] * T}
                                       for T in (32768, 262144)}
    return out


# ------------------------------------------------------------------------------- compute
def compute():
    """Recorded GPU time: nanolm's eval/train times and lm_eval's total evaluation time per run (runs
    shared the GPU up to four at a time, so these overlap and sum to more than the elapsed time)."""
    ft = {k: sum(json.loads(open(f).read())["train_time_sec"] for f in glob.glob(str(RUNS / k / "*_s*" / "metrics.json")))
          for k in ("sft", "dpo")}
    n_ft = {k: len(glob.glob(str(RUNS / k / "*_s*" / "metrics.json"))) for k in ("sft", "dpo")}
    lm = [float(json.loads(open(f).read())["total_evaluation_time_seconds"])
          for f in glob.glob(str(RUNS / "benchmarks" / "*" / "*" / "*" / "*" / "results_*.json"))]
    lm += [float(json.loads(open(f).read())["total_evaluation_time_seconds"])
           for f in glob.glob(str(RUNS / "crosscheck" / "*" / "lm_eval" / "*" / "results_*.json"))]
    bp = sum(json.loads(open(f).read())["eval_time_sec"] for f in glob.glob(str(RUNS / "bpb" / "*" / "metrics.json")))
    return {"finetune_runs": n_ft, "finetune_hours": {k: v / 3600 for k, v in ft.items()},
            "lm_eval_runs": len(lm), "lm_eval_hours": sum(lm) / 3600, "bpb_hours": bp / 3600}


# ----------------------------------------------------------------------------- provenance
def provenance():
    """Model commit, lm_eval version and chat-template use of every benchmark run (lm_eval's results files)."""
    shas, versions, templates = {}, set(), set()
    for f in glob.glob(str(RUNS / "benchmarks" / "*" / "*" / "*" / "*" / "results_*.json")):
        r = json.loads(open(f).read())
        shas.setdefault(r["model_name"], set()).add(r["config"]["model_sha"])
        versions.add(r["lm_eval_version"])
        templates.add(r["chat_template"] is not None)
    return {"model_sha": {k: sorted(v) for k, v in sorted(shas.items())}, "lm_eval_version": sorted(versions),
            "any_chat_template": any(templates)}


def main():
    models = BASE_MODELS + ["SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct"]
    summary = {"bpb": bpb(), "crosscheck": crosscheck(),
               "benchmarks": bench_table("models", models), "benchmarks_full": bench_table("full", BASE_MODELS),
               "pairwise_subset": pairwise("models", BASE_MODELS), "pairwise_full": pairwise("full", BASE_MODELS),
               "laptop_reproduction": laptop_reproduction(), "mmlu_cloze": cloze(), "mmlu_cloze_pairs": cloze_pairs(),
               "subset_vs_full": subset_vs_full(), "hellaswag_sources": hellaswag_sources(), "mmlu_letters": mmlu_letters(), "architecture": architecture(), "provenance": provenance(), "compute": compute(), "cards": cards(), "gsm8k_filters": gsm8k_filters(), "token_budget": token_budget(),
               "sft": sft(), "dpo": dpo()}
    summary["model_cards"] = model_cards(summary["benchmarks_full"], summary["benchmarks"], summary["mmlu_cloze"])
    # benchmark changes after tuning that pass the resolution rule, against a 95% interval's chance rate
    from scipy.stats import binom
    flags = [v[t]["passes_rule"] for k in ("sft", "dpo") for v in summary[k]["benchmarks"].values() for t in v]
    n, n_pass = len(flags), sum(flags)
    summary["rule_passes"] = {"n_changes": n, "n_pass": n_pass, "expected_by_chance_95": 0.05 * n,
                              "p_at_least_n_pass_if_independent": float(binom.sf(n_pass - 1, n, 0.05))}
    save(summary, ROOT / "results" / "summary.json")


if __name__ == "__main__":
    main()
