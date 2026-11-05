"""`python -m nanolm.benchmarks`, with lm_eval's per-question records switched on.

    python scripts/bench.py run -c configs/benchmarks/models.yaml --slots 4
    python scripts/bench.py summary runs/benchmarks/models

The jobs are exactly the ones nanolm builds (nanolm.benchmarks.jobs_for, unmodified), with
`--log_samples` added so that every question's log-likelihoods and correctness are saved. The
per-question records give the paired tests between models; the aggregate scores are unchanged.
"""
import nanolm.benchmarks as nb

_jobs_for = nb.jobs_for


def jobs_for(cfg):
    jobs = _jobs_for(cfg)
    for j in jobs:
        j["cmd"].insert(j["cmd"].index("--device"), "--log_samples")
    return jobs


nb.jobs_for = jobs_for

if __name__ == "__main__":
    nb.main()
