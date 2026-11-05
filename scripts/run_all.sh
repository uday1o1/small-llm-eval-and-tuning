#!/usr/bin/env bash
# Every run behind the results (one GPU). Finished runs are skipped, so the script can be re-run to
# resume. SLOTS = how many training or evaluation jobs share the GPU at a time; BSLOTS = the same for
# lm_eval jobs (fp32 logits over a 150k to 250k-token vocabulary are large), and any that fail,
# for example out of memory, are retried one at a time.
#   bash scripts/run_all.sh                 # all stages
#   bash scripts/run_all.sh sft dpo         # selected stages
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
NL=external/DeepDataMiningLearning/DeepDataMiningLearning/llm/nanolm
SLOTS=${SLOTS:-4}
BSLOTS=${BSLOTS:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SEEDS="1337 42 1 2 3"
BASE="gpt2 HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B Qwen/Qwen3-0.6B-Base Qwen/Qwen3.5-0.8B-Base"
short() { basename "$1"; }
# run_jobs: reads "<done-file> <command>" lines, runs the commands whose done-file is missing
run_jobs() { while read -r done cmd; do [ -e "$done" ] || echo "$cmd"; done | xargs -P "$SLOTS" -I{} bash -c {}; }

prepare() { python -m nanolm.prepare --golf-shards 1 --dest data/golf; }

bpb() {   # FineWeb bits per byte of each base model (nanolm's P1_eval_bpb, fp32)
  for m in $BASE; do s=$(short $m); d=runs/bpb/$s; mkdir -p $d
    echo "$d/metrics.json python -m nanolm.finetune -c $NL/configs/part2/P1_eval_bpb.yaml model.name=$m model.dtype=fp32 name=bpb_$s run_dir=$d > $d/stdout.log 2>&1"
  done | run_jobs
}

crosscheck() {   # nanolm's scorer and lm_eval on the first 300 ARC-Easy questions; all 2,376 with both normalizations
  for m in $BASE; do s=$(short $m); d=runs/crosscheck/$s; mkdir -p $d
    echo "$d/crosscheck.log python -m nanolm.benchmarks crosscheck $m --limit 300 > $d/crosscheck.log.tmp 2>&1 && mv $d/crosscheck.log.tmp $d/crosscheck.log"
    echo "$d/lm_eval/DONE python -m lm_eval --model hf --model_args pretrained=$m,dtype=float32 --tasks arc_easy --num_fewshot 0 --limit 300 --batch_size 8 --device cuda --seed 1234 --log_samples --output_path $d/lm_eval > $d/lm_eval.log 2>&1 && touch $d/lm_eval/DONE"
    echo "$d/arc_easy_scores.json python scripts/arc_scores.py $m $d > $d/arc_scores.log 2>&1"
  done | run_jobs
}

benchmarks() {   # lm_eval through nanolm.benchmarks: nanolm's laptop config (output directory changed only), then ours
  for n in "$BSLOTS" 1; do python -m nanolm.benchmarks run -c configs/benchmarks/laptop.yaml --slots $n; done
  for c in models full cloze cloze_qwen3.5 cards; do for n in "$BSLOTS" 1; do python scripts/bench.py run -c configs/benchmarks/$c.yaml --slots $n; done; done
}

sft() {   # training-set size 500 / 2000 / 8000, 250 steps x 16 examples each
  for s in $SEEDS; do for n in 500 2000 8000; do d=runs/sft/n${n}_s$s; mkdir -p $d
    echo "$d/metrics.json python -m nanolm.finetune -c configs/finetune/sft.yaml data.n_train=$n seed=$s name=sft_n$n run_dir=$d > $d/stdout.log 2>&1"
  done; done | run_jobs
}

dpo() {   # beta 0.05 / 0.1 / 0.5, 100 steps x 8 pairs each
  for s in $SEEDS; do for b in 0.05 0.1 0.5; do d=runs/dpo/beta${b}_s$s; mkdir -p $d
    echo "$d/metrics.json python -m nanolm.finetune -c configs/finetune/dpo.yaml train.dpo_beta=$b seed=$s name=dpo_beta$b run_dir=$d > $d/stdout.log 2>&1"
  done; done | run_jobs
}

tuned_benchmarks() {   # the same benchmarks for every SFT and DPO adapter
  for c in sft dpo; do for n in "$BSLOTS" 1; do python scripts/bench.py run -c configs/benchmarks/$c.yaml --slots $n; done; done
}

posthoc() {   # FineWeb bpb, held-out generations and (DPO) all held-out pairs, for every tuned model and its base
  for k in sft dpo; do for d in runs/$k/base runs/$k/*_s*; do
    mkdir -p $d; echo "$d/posthoc.json python scripts/posthoc.py $k $d > $d/posthoc.log 2>&1"
  done; done | run_jobs
  [ -e runs/dpo/pair_lengths.json ] || python scripts/pair_lengths.py runs/dpo/pair_lengths.json > /dev/null
  [ -e runs/token_budget.json ] || python scripts/token_budget.py runs/token_budget.json > /dev/null
}

architecture() {   # parameters per component and inference-state memory: Qwen3.5-2B against Qwen3-1.7B
  [ -e results/architecture/architecture.json ] || python scripts/architecture.py results/architecture
}

analyze() {   # per-question tables, nanolm.analyze per study (paired p against the middle setting), the environment
  python scripts/extract_items.py runs/benchmarks/models runs/benchmarks/full runs/benchmarks/cloze runs/benchmarks/cards runs/benchmarks/sft runs/benchmarks/dpo
  # nanolm.analyze writes summary.csv/md and then stops at its curve plot (fine-tuning eval logs have no
  # train_tokens column), so its exit status is ignored and the summary files are checked instead
  python -m nanolm.analyze runs/sft --metric after_val_response_nll --baseline sft_n2000 > runs/sft/analyze_stdout.txt 2>&1 || true
  python -m nanolm.analyze runs/dpo --metric after_val_reward_acc --baseline dpo_beta0.1 > runs/dpo/analyze_stdout.txt 2>&1 || true
  ls runs/sft/summary.csv runs/sft/summary.md runs/dpo/summary.csv runs/dpo/summary.md > /dev/null
  for c in laptop models full cloze cards sft dpo; do r=runs/benchmarks/$c
    [ -d $r ] && python scripts/bench.py summary $r > $r/summary_stdout.txt; done
  { echo "# Environment of the runs (recorded on the machine that ran them; the publisher-setting runs of cards.yaml used other machines with the same GPU model and packages)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo
    lscpu | grep -E "^(CPU\(s\)|Model name)"; echo
    python -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version())"; echo
    echo "nanolm: $(git -C external/DeepDataMiningLearning rev-parse HEAD) (clean: $([ -z "$(git -C external/DeepDataMiningLearning status --porcelain)" ] && echo yes || echo no))"; echo
    python -m pip list --format=freeze 2>/dev/null | grep -iE "^($(grep -v '^#' requirements.txt | cut -d= -f1 | paste -sd'|' -))=="
  } > results/environment.txt
}

STAGES=${*:-prepare bpb crosscheck benchmarks sft dpo tuned_benchmarks posthoc architecture analyze}
for st in $STAGES; do echo "[run_all] $st"; $st; done
echo "[run_all] done"
