"""Compact per-question results from lm_eval's --log_samples records (no GPU).

    python scripts/extract_items.py runs/benchmarks/models

For every model directory under the given roots, all samples_<task>_*.jsonl files are reduced to
one gzipped CSV, items.csv.gz: task, doc_id, filter, number of choices, the source corpus of a
HellaSwag question (ActivityNet or WikiHow), the index of the correct choice and of the choice
with the highest log-likelihood, and the per-question
metrics (acc, acc_norm for multiple choice; exact_match for GSM8K, one row per answer filter).
These tables are what the paired comparisons between models use; the full records (prompts,
log-likelihoods, generated answers) are large and kept out of the repository.
"""
import ast
import csv
import gzip
import json
import re
import sys
from pathlib import Path

FIELDS = ["task", "doc_id", "filter", "n_choices", "source", "gold", "pred", "acc", "acc_norm", "exact_match"]


def rows_of(f: Path):
    task = re.match(r"samples_(.+)_\d{4}-\d\d-\d\dT", f.name).group(1)
    for line in open(f):
        r = json.loads(line)
        resps = r["resps"]
        if isinstance(resps, str):
            resps = ast.literal_eval(resps)
        out = {"task": task, "doc_id": r["doc_id"], "filter": r.get("filter", "none"),
               "n_choices": len(resps) if "acc" in r else "",
               "source": str(r.get("doc", {}).get("source_id", "")).split("~")[0],   # HellaSwag: activitynet or wikihow
               "gold": "", "pred": ""}
        if "acc" in r:   # multiple choice: index of the correct choice, and of the highest raw log-likelihood
            ll = [float(x[0][0] if isinstance(x[0], list) else x[0]) for x in resps]
            out["pred"] = ll.index(max(ll))
            out["gold"] = int(r["target"])
        for k in ("acc", "acc_norm", "exact_match"):
            out[k] = float(r[k]) if k in r else ""
        yield out


def main(roots):
    for root in map(Path, roots):
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted(model_dir.rglob("samples_*.jsonl"))
            if not files:
                continue
            with gzip.open(model_dir / "items.csv.gz", "wt", newline="") as g:
                w = csv.DictWriter(g, fieldnames=FIELDS)
                w.writeheader()
                for f in files:
                    w.writerows(rows_of(f))
            print(model_dir, len(files), "files")


if __name__ == "__main__":
    main(sys.argv[1:])
