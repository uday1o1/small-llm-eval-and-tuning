"""All figures, from results/summary.json (no GPU).

    python analysis/figures.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BASE_MODELS, DPO_BETAS, ROOT, SFT_SIZES, TASKS

S = json.loads((ROOT / "results" / "summary.json").read_text())
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = plt.get_cmap("tab10").colors
SHORT = {"gpt2-124M": "GPT-2 124M", "SmolLM2-135M": "SmolLM2 135M", "SmolLM2-360M": "SmolLM2 360M",
         "Qwen2.5-0.5B": "Qwen2.5 0.5B", "Qwen3-0.6B-Base": "Qwen3 0.6B", "Qwen3.5-0.8B-Base": "Qwen3.5 0.8B"}


def evaluation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 2.6]})
    x = np.arange(len(BASE_MODELS))
    v = [S["bpb"][m]["val_bpb"] for m in BASE_MODELS]
    a.scatter(x, v, color=[COL[i] for i in range(len(x))], s=28, zorder=2)
    a.scatter(x, [S["bpb"][m]["tutorial"] for m in BASE_MODELS], marker="_", s=120, color="k", zorder=3,
              label="tutorial value")
    a.set_xticks(x, [SHORT[m] for m in BASE_MODELS], rotation=40, ha="right")
    a.set_ylim(0.85, 1.1)
    a.set_ylabel("FineWeb validation bits per byte")
    a.set_title("(a) FineWeb bpb (200 documents)", loc="left")
    a.legend(loc="upper right", frameon=False)
    tasks = list(TASKS)
    w = 0.8 / len(BASE_MODELS)
    for i, m in enumerate(BASE_MODELS):
        r = S["benchmarks"][m]
        xs = np.arange(len(tasks)) + (i - (len(BASE_MODELS) - 1) / 2) * w
        b.bar(xs, [r[t]["acc"] for t in tasks], w, yerr=[r[t]["se"] for t in tasks], color=COL[i], capsize=1.2,
              error_kw={"lw": 0.6}, label=SHORT[m])
        b.scatter(xs, [S["benchmarks_full"][m][t]["acc"] for t in tasks], marker="_", s=30, color="k", lw=1,
                  zorder=3, label="full set" if i == 0 else None)
    for j, t in enumerate(tasks):
        c = S["benchmarks"][BASE_MODELS[0]][t]["chance"]
        b.hlines(c, j - 0.45, j + 0.45, color="k", ls="--", lw=0.8, label="chance" if j == 0 else None)
    b.set_xticks(range(len(tasks)), [f"{TASKS[t][1]}\n({'acc_norm' if TASKS[t][0] == 'acc_norm' else 'acc' if t != 'gsm8k' else 'exact match'})" for t in tasks])
    b.set_ylabel("accuracy (± SE)")
    b.set_ylim(0, 0.8)
    b.set_title("(b) Benchmarks: bars on the first 300 questions, marks on all", loc="left")
    b.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "evaluation.png", dpi=200)
    plt.close(fig)


def bench_delta_panel(ax, block, keys, labels, title):
    tasks = list(TASKS)
    w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        xs = np.arange(len(tasks)) + (i - (len(keys) - 1) / 2) * w
        d = [block[k][t]["diff"] for t in tasks]
        m = np.array([100 * x["mean"] for x in d])
        ci = np.array([[100 * block[k][t]["boot_ci95"][0] for t in tasks], [100 * block[k][t]["boot_ci95"][1] for t in tasks]])
        ax.bar(xs, m, w, yerr=[m - ci[0], ci[1] - m], color=COL[i], capsize=1.5, error_kw={"lw": 0.6}, label=labels[i])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(tasks)), [TASKS[t][1].replace("ARC-Challenge", "ARC-C").replace("ARC-Easy", "ARC-E") for t in tasks])
    ax.set_ylabel("tuned minus untuned (points)")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=6.5)


def sft():
    T = S["sft"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for i, n in enumerate(SFT_SIZES):
        k = f"n{n}"
        steps = [0] + list(map(int, T[k]["curve"]))
        mean = [T[k]["before_nll"]["mean"]] + [c["mean"] for c in T[k]["curve"].values()]
        std = [0] + [c["std"] for c in T[k]["curve"].values()]
        ex = np.array(steps) * 16
        a.plot(ex, mean, marker="o", ms=2.5, color=COL[i], label=f"{n:,} training examples")
        a.fill_between(ex, np.array(mean) - std, np.array(mean) + std, color=COL[i], alpha=0.2)
    a.set_xlabel("training examples seen (16 per step)")
    a.set_ylabel("held-out answer loss (nats per token)")
    a.set_title("(a) Held-out answer loss, 5 seeds", loc="left")
    a.legend(frameon=False)
    bench_delta_panel(b, T["benchmarks"], [f"n{n}" for n in SFT_SIZES], [f"{n:,} examples" for n in SFT_SIZES],
                      "(b) Benchmark change (95% intervals)")
    fig.tight_layout()
    fig.savefig(OUT / "sft.png", dpi=200)
    plt.close(fig)


def dpo():
    D = S["dpo"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for i, b in enumerate(DPO_BETAS):
        k = f"beta{b}"
        steps = [0] + list(map(int, D[k]["curve"]))
        for ax, m, before in ((axes[0], "acc", 0.5), (axes[1], "margin", 0.0)):
            mean = [before] + [c[m]["mean"] for c in D[k]["curve"].values()]
            std = [0] + [c[m]["std"] for c in D[k]["curve"].values()]
            px = np.array(steps) * 8
            ax.plot(px, mean, marker="o", ms=2.5, color=COL[i], label=f"β = {b}")
            ax.fill_between(px, np.array(mean) - std, np.array(mean) + std, color=COL[i], alpha=0.2)
        mean = [0] + [c["margin"]["mean"] / D[k]["beta"] for c in D[k]["curve"].values()]
        axes[2].plot(np.array(steps) * 8, mean, marker="o", ms=2.5, color=COL[i], label=f"β = {b}")
    axes[0].axhline(0.5, color="k", ls="--", lw=0.7)
    axes[0].set_ylabel("reward accuracy")
    axes[1].set_ylabel("reward margin β·Δ")
    axes[2].set_ylabel("Δ = margin / β")
    for ax, t in zip(axes, ["(a) reward accuracy", "(b) reward margin", "(c) margin / β"]):
        ax.set_xlabel("preference pairs seen")
        ax.set_title(t, loc="left")
    axes[0].legend(frameon=False)
    fig.suptitle("DPO of Qwen2.5-0.5B-Instruct on 100 held-out pairs, mean over 5 seeds (± std shaded in a, b)", fontsize=8.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "dpo.png", dpi=200)
    plt.close(fig)


def architecture():
    R = S["architecture"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    T = np.logspace(np.log10(1024), np.log10(262144), 50)
    for i, name in enumerate(["Qwen/Qwen3.5-2B", "Qwen/Qwen3-1.7B"]):
        r = R[name]
        ax.plot(T, (r["kv_bytes_per_token"] * T + r["fixed_state_bytes"]) / 2 ** 30, color=COL[i],
                label=name.split("/")[1])
        ax.scatter([m["T"] for m in r["measured"]], [m["state_bytes"] / 2 ** 30 for m in r["measured"]],
                   color=COL[i], s=12, zorder=3)
    for x in (32768, 262144):
        ax.axvline(x, color="k", lw=0.5, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("inference state per sequence (GiB)")
    ax.set_title("Lines: from config.json; dots: measured", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "architecture.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    for fn in (evaluation, sft, dpo, architecture):
        fn()
