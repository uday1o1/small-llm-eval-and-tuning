# Evaluating and post-training small language models

What do benchmark scores, instruction tuning and preference tuning say about language models of at most 0.8B parameters, and why do newer models mix linear and full attention? Five measurements:

1. **Small models.** Six base models (GPT-2 124M, SmolLM2-135M and -360M, Qwen2.5-0.5B, Qwen3-0.6B-Base, Qwen3.5-0.8B-Base) and two instruct models: FineWeb bits per byte and five lm_eval benchmarks, with standard errors, chance levels, paired tests on the same questions, and the publishers' own settings run on the same questions.
2. **The scoring rule.** nanolm's own multiple-choice scorer against lm_eval, and what changes when a choice's log-likelihood is normalized by tokens instead of characters.
3. **Instruction tuning (SFT).** LoRA SFT of SmolLM2-360M with 500, 2,000 or 8,000 training examples at a fixed budget of 4,000 examples.
4. **Preference tuning (DPO).** LoRA DPO of Qwen2.5-0.5B-Instruct with β = 0.05, 0.1 and 0.5.
5. **An architecture.** Parameters per component and inference memory of Qwen3.5-2B (Gated DeltaNet plus full attention) against Qwen3-1.7B (full attention only).

The bpb, fine-tuning and benchmark runs go through [nanolm](https://github.com/lkk688/DeepDataMiningLearning/tree/8418fc1bae85657190f11749483bf2e729b9702c/DeepDataMiningLearning/llm) at commit `8418fc1`, unmodified, and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.11; the architecture measurement and the post-training measurements are scripts in `scripts/`. Every measured number below is computed from the records in `runs/` and `results/architecture/` by `analysis/summarize.py` into `results/summary.json`.

## Results

Means ± std are over 5 seeds (1337, 42, 1, 2, 3). Differences between settings are paired over seeds (t-test). A difference below twice the noise floor of the two settings compared (the mean of their two within-setting stds) is treated as not resolved.

### Small models

FineWeb bpb (nanolm's `part2/P1_eval_bpb.yaml`, 200 documents) and accuracy ± SE (√(p(1−p)/n)) on the first 300 questions per task (MMLU: per subject, 10,908 questions; GSM8K: the first 100, 5-shot), 0-shot, fp32, no chat template. ° = within 2 standard deviations of chance; † = at least 2 below chance. ARC and HellaSwag: acc_norm; MMLU: acc; GSM8K: exact match (flexible-extract).

| model | FineWeb bpb | ARC-Easy | ARC-Challenge | HellaSwag | MMLU | GSM8K |
|---|---:|---:|---:|---:|---:|---:|
| gpt2-124M | 1.075 | 0.380 ± 0.028 | 0.223 ± 0.024 ° | 0.427 ± 0.029 | 0.228 ± 0.004 † | 0.010 ± 0.010 |
| SmolLM2-135M | 1.010 | 0.553 ± 0.029 | 0.277 ± 0.026 ° | 0.437 ± 0.029 | 0.243 ± 0.004 ° | 0.020 ± 0.014 |
| SmolLM2-360M | 0.928 | 0.663 ± 0.027 | 0.363 ± 0.028 | 0.553 ± 0.029 | 0.262 ± 0.004 | 0.050 ± 0.022 |
| Qwen2.5-0.5B | 0.951 | 0.570 ± 0.029 | 0.327 ± 0.027 | 0.497 ± 0.029 | 0.495 ± 0.005 | 0.340 ± 0.047 |
| Qwen3-0.6B-Base | 0.953 | 0.543 ± 0.029 | 0.370 ± 0.028 | 0.530 ± 0.029 | 0.526 ± 0.005 | 0.520 ± 0.050 |
| Qwen3.5-0.8B-Base | 0.988 | 0.660 ± 0.027 | 0.403 ± 0.028 | 0.533 ± 0.029 | 0.551 ± 0.005 | 0.400 ± 0.049 |
| SmolLM2-360M-Instruct | | 0.493 ± 0.029 | 0.343 ± 0.027 | 0.543 ± 0.029 | 0.242 ± 0.004 ° | 0.020 ± 0.014 |
| Qwen2.5-0.5B-Instruct | | 0.583 ± 0.028 | 0.353 ± 0.028 | 0.490 ± 0.029 | 0.475 ± 0.005 | 0.340 ± 0.047 |
| chance | | 0.250 | 0.250 | 0.250 | 0.250 | 0 |

On every question of each evaluation set (HellaSwag: the validation set), with MMLU also in its cloze form (each answer's text scored after the question):

| model | ARC-Easy | ARC-Challenge | HellaSwag | MMLU | GSM8K | MMLU cloze (acc_norm) |
|---|---:|---:|---:|---:|---:|---:|
| gpt2-124M | 0.395 | 0.227 ° | 0.311 | 0.229 † | 0.007 | 0.268 |
| SmolLM2-135M | 0.586 | 0.296 | 0.431 | 0.243 ° | 0.019 | 0.314 |
| SmolLM2-360M | 0.680 | 0.384 | 0.563 | 0.262 | 0.034 | 0.356 |
| Qwen2.5-0.5B | 0.587 | 0.324 | 0.521 | 0.477 | 0.353 | 0.338 |
| Qwen3-0.6B-Base | 0.580 | 0.381 | 0.538 | 0.506 | 0.499 | 0.364 |
| Qwen3.5-0.8B-Base | 0.673 | 0.404 | 0.548 | 0.529 | 0.401 | 0.353 |

- **Reproduction.** The six bpb values are within 0.004 of nanolm's tutorial, and nanolm's `part3/B2_laptop.yaml`, with only its output directory changed, reproduces its quickstart table to every digit printed there.
- **Below chance is systematic.** GPT-2 answers A to 99.4% of MMLU questions; A is the correct letter for 3,222 of the 14,042, and GPT-2 gets 3,219 right.
- **The answer format reorders models.** In MMLU's letter format SmolLM2-360M trails Qwen2.5-0.5B by 0.215. In the cloze form it is ahead by 0.019 (McNemar p < 0.0001) and not resolved from Qwen3.5-0.8B-Base (−0.004, p = 0.34).
- **The first 300 questions are not always representative.** The first 300 HellaSwag validation questions all come from ActivityNet: GPT-2 scores 0.427 on them, 0.311 on all 10,042, and 0.275 on the 6,799 WikiHow ones.
- **Publisher numbers.** SmolLM2's base-model cards (0-shot lighteval; GSM8K 5-shot) agree within 0.018 on ARC (the cards' average, against the mean of our ARC-Easy and ARC-Challenge), HellaSwag and GSM8K. The publishers' own settings, run on the same questions (`configs/benchmarks/cards.yaml`):

| test (same questions) | questions | our setting | publisher's setting | published |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B ARC-Challenge, 25-shot vs 0-shot | 1,172 | 0.324 | 0.348 (p = 0.0499) | 0.356 |
| Qwen3-0.6B-Base MMLU, 5-shot vs 0-shot | 14,042 | 0.506 | 0.525 (p < 0.0001) | 0.528 |
| Qwen2.5-0.5B GSM8K, 4-shot vs 5-shot | 300 | 0.350 | 0.340 (p = 0.79) | 0.416 |
| Qwen2.5-0.5B GSM8K, 4-shot chain of thought vs 5-shot | 300 | 0.350 | 0.413 (p = 0.034) | 0.416 |
| Qwen3-0.6B-Base GSM8K, 4-shot vs 5-shot | 300 | 0.530 | 0.507 (p = 0.40) | 0.596 |
| Qwen3-0.6B-Base GSM8K, 4-shot chain of thought vs 5-shot | 300 | 0.530 | 0.610 (p = 0.0071) | 0.596 |
| SmolLM2-360M-Instruct GSM8K, chat template vs none | 100 | 0.020 | 0.070 (p = 0.18) | 0.0743 |
| Qwen2.5-0.5B-Instruct GSM8K, chat template vs none | 100 | 0.340 | 0.360 (p = 0.85) | 0.496 |
| SmolLM2-360M-Instruct GSM8K, chat template and 4-shot chain of thought vs none | 100 | 0.020 | 0.060 (p = 0.22) | 0.0743 |
| Qwen2.5-0.5B-Instruct GSM8K, chat template and 4-shot chain of thought vs none | 100 | 0.340 | 0.390 (p = 0.44) | 0.496 |

GSM8K uses lm_eval's prompts on the first 300 questions (base models) or 100 (instruct models). Qwen3 states a chain-of-thought prompt, Qwen2.5's report gives only "4-shot", so lm_eval's standard prompt and its chain-of-thought prompt are run, both at 4 shots. Qwen states no setting for its instruct model, so that model runs with the chat template, as SmolLM2's evaluation does, with lm_eval's 5-shot prompt and with its 4-shot chain-of-thought prompt. Few-shot prompts move ARC-Challenge and MMLU toward the published values and the chain-of-thought prompt does the same for the base models' GSM8K (McNemar p < 0.05, uncorrected); the shot count alone and the chat template change nothing with p < 0.05, and with Holm's correction over these 10 tests only the MMLU change is resolved. Qwen2.5-0.5B-Instruct's published GSM8K stays unexplained (0.39 with the chat template and chain of thought, p = 0.44, against 0.496).

### The scoring rule (ARC-Easy)

| model | acc / acc_norm, first 300 (both tools) | flips, first 300 | flips, all 2,376 | acc_norm, chars → tokens (all 2,376) | McNemar (all 2,376) |
|---|---:|---:|---:|---:|---:|
| gpt2-124M | 0.4200 / 0.3800 | 133 | 1,108 | 0.3948 → 0.3817 | p = 0.26 |
| SmolLM2-135M | 0.5833 / 0.5533 | 115 | 813 | 0.5859 → 0.5943 | p = 0.45 |
| SmolLM2-360M | 0.6833 / 0.6633 | 76 | 609 | 0.6801 → 0.6658 | p = 0.13 |
| Qwen2.5-0.5B | 0.5967 / 0.5700 | 90 | 764 | 0.5867 → 0.5787 | p = 0.45 |
| Qwen3-0.6B-Base | 0.6100 / 0.5433 | 101 | 806 | 0.5795 → 0.5715 | p = 0.46 |
| Qwen3.5-0.8B-Base | 0.6767 / 0.6600 | 72 | 624 | 0.6734 → 0.6612 | p = 0.21 |

`python -m nanolm.benchmarks crosscheck <model> --limit 300` and the matching lm_eval run give the same acc and acc_norm for all six models, and `scripts/arc_scores.py`, which repeats nanolm's `mc_accuracy` computation per choice (and asserts nanolm's acc and acc_norm), gives lm_eval's verdict on each of the 300 questions, with per-choice log-likelihoods within 3.6e-4 nats. Dividing by tokens instead of characters multiplies each (negative) score by the choice's characters per token, so every flip moves to the choice made of shorter tokens. That is a property of the tokenizer, not of the answer, and right and wrong flips nearly cancel.

### Instruction tuning: training-set size (SmolLM2-360M, LoRA r 16, 250 steps × 16 examples)

| training examples | passes | held-out answer loss | vs 2,000 | FineWeb bpb | answers ended | answer tokens |
|---:|---:|---:|---:|---:|---:|---:|
| untuned | | 1.556 | | 0.9279 | 4% | 253 |
| 500 | 8 | 1.3674 ± 0.0029 | +0.0465 (p < 0.0001) | 0.9390 ± 0.0002 | 87% | 103 |
| 2,000 | 2 | 1.3208 ± 0.0005 | | 0.9321 ± 0.0001 | 82% | 101 |
| 8,000 | 0.5 | 1.3211 ± 0.0009 | +0.0003 (p = 0.64, not resolved) | 0.9315 ± 0.0002 | 82% | 104 |

- 4,000 examples are drawn either way (500 eight times, 2,000 twice, or 4,000 different ones of 8,000 once; 0.58M to 0.60M tokens per run, `runs/token_budget.json`). 8,000 examples are no better than 2,000. 500 examples reach their lowest held-out loss at step 100 (3.2 passes) on every seed and then overfit.
- SFT changes the form of the answers (they end, and stop starting new `###` sections). It does not make them factual: on nanolm's fixed sample prompts, 11 of the 15 tuned models place Paris in Normandy, which the untuned model does not.
- A benchmark change after tuning counts as resolved when it exceeds twice its seed std and its 95% interval from resampling the questions (paired bootstrap of the seed-mean change) excludes 0. Of the 15 SFT changes, only MMLU at 500 and 2,000 examples (+1.0 and +0.8 points) passes, the 2,000 interval's lower end at +0.05 points and the 500 one's at +0.001 (borderline).

### Preference tuning: β (Qwen2.5-0.5B-Instruct, LoRA r 16, 100 steps × 8 pairs)

| β | reward accuracy (100 pairs) | reward margin | margin / β | reward accuracy (all 240 held-out pairs) | policy prefers chosen (240) | answer tokens |
|---:|---:|---:|---:|---:|---:|---:|
| untuned | 0.500 | 0 | | 0.500 | 0.650 | 114 |
| 0.05 | 0.682 ± 0.018 | 0.494 ± 0.029 | 9.88 | 0.686 ± 0.005 | 0.683 | 145 |
| 0.1 | 0.696 ± 0.030 | 0.656 ± 0.069 | 6.56 | 0.690 ± 0.024 | 0.674 | 144 |
| 0.5 | 0.692 ± 0.029 | 1.302 ± 0.237 | 2.60 | 0.708 ± 0.011 | 0.655 | 129 |

- The reward accuracy starts at 0.5 by construction (the reference is the untuned model, so every margin is 0 and a tie counts half). On nanolm's 100 held-out pairs no two β values differ beyond the noise; on all 240 held-out pairs (every test-split pair nanolm's filter keeps) β 0.5 beats β 0.05 by 0.0225 (p = 0.006).
- The margin grows with β although it is β times a log-ratio difference that shrinks as β grows (9.88, 6.56, 2.60): a larger β holds the policy closer to the reference. Both answers lose probability under the policy; per token the rejected answer loses 2.0 to 3.1 times as much as the chosen one.
- The policy's own preference for the chosen answer moves little: from 0.650 to 0.683 at β 0.05 (p < 0.0001) and to 0.655 at β 0.5 (p = 0.07).
- The answers get longer. On the 14 held-out prompts the untuned model answers in at most 20 tokens, the tuned model starts with the same answer in 8 to 11 per run and makes it longer in 6 to 9, usually by appending an explanation. The chosen answer is the longer one in 58.3% of the held-out pairs.
- Of the 15 benchmark changes only HellaSwag at β 0.5 (−1.1 points) passes the rule. GSM8K falls on every seed at β 0.05 and 0.1 (by 3 to 6 points of 100 questions), with intervals too wide to resolve it. 3 of the 30 SFT and DPO changes pass; a 95% interval alone would exclude 0 for about 1.5 of 30 changes by chance (3 or more with probability 0.19 if they were independent).

### Qwen3.5-2B against Qwen3-1.7B

| | Qwen3.5-2B (hybrid) | Qwen3-1.7B (full attention) |
|---|---:|---:|
| layers (Gated DeltaNet / full attention) | 24 (18 / 6) | 28 (0 / 28) |
| embedding (tied output head) | 508,559,360 | 311,164,928 |
| full attention | 88,083,456 | 352,328,704 |
| linear attention | 379,112,256 | 0 |
| MLP | 905,969,664 | 1,056,964,608 |
| norms | 100,352 | 116,736 |
| total | 1,881,825,088 | 1,720,574,976 |
| KV cache per token (bf16) | 12,288 B | 114,688 B |
| fixed Gated DeltaNet state | 19,759,104 B | 0 |
| inference state at 32k / 262k tokens | 0.39 / 3.02 GiB | 3.50 / 28.00 GiB |

Parameter counts are of the language model (Qwen3.5-2B's vision encoder and multi-token-prediction head are left out), from the model built on PyTorch's meta device, and equal the shapes in the released safetensors. The cache measured after a bf16 forward pass at 1k, 4k, 16k and 32k tokens equals the config's prediction to the byte, counting each Gated DeltaNet layer's recurrent state in fp32 and its convolution state at the full kernel width, as transformers stores them. Qwen3-1.7B's config allows 40,960 positions, so its 262k value is its per-token cache times 262,144. At 262k tokens Qwen3.5-2B keeps 9.3× less state than Qwen3-1.7B, and 4.0× less than it would with all 24 layers full attention.

## Layout

| path | contents |
|---|---|
| `configs/finetune/` | `sft.yaml`, `dpo.yaml`: nanolm's `P2_sft_alpaca.yaml` and `P3_dpo.yaml` with fp32 weights and no wall-clock cap |
| `configs/benchmarks/` | `laptop.yaml` (nanolm's `B2_laptop.yaml` verbatim, comments included, with only the output directory changed), `models.yaml` (all eight models), `full.yaml` (every question), `cloze.yaml` and `cloze_qwen3.5.yaml` (MMLU cloze form), `cards.yaml` (the publishers' settings), `sft.yaml` and `dpo.yaml` (every adapter) |
| `runs/bpb/`, `runs/crosscheck/` | FineWeb bpb records; crosscheck logs, the lm_eval run with its per-question records, and per-choice scores for all 2,376 ARC-Easy questions (`arc_easy_scores.csv`) |
| `runs/benchmarks/<config>/<model>/` | lm_eval results per task group and `stdout.log`; except under `laptop/`, `items.csv.gz` (one row per question, and per answer filter for GSM8K: correct and chosen answer, scores); `summary.csv` per config |
| `runs/sft/`, `runs/dpo/` | nanolm's record of every run (`config.yaml`, `meta.json`, `train_log.jsonl`, `eval_log.jsonl`, `metrics.json`, `samples_before.txt`, `samples_after.txt`, `stdout.log`), `posthoc.json` and `generations.jsonl`; `base/` for the untuned model; `summary.csv`/`summary.md` from `python -m nanolm.analyze` (it labels every metric "lower is better" and its printed noise floor "bpb", which for DPO's reward accuracy read the other way and in the metric's own units; it then stops at its curve plot, and `analyze_stdout.txt` holds that error ahead of the printed table); `runs/dpo/pair_lengths.json` |
| `runs/token_budget.json` | training tokens of every SFT and DPO run |
| `results/` | `summary.json`, `environment.txt`, `architecture/` (pinned config.json files, `architecture.json`) |
| `analysis/` | `summarize.py`, `figures.py`, `common.py` |
| `figures/` | `evaluation.png`, `sft.png`, `dpo.png`, `architecture.png` |
| `scripts/` | `setup.sh`, `run_all.sh`, `bench.py` (nanolm.benchmarks with lm_eval's `--log_samples`), `arc_scores.py`, `posthoc.py`, `pair_lengths.py`, `token_budget.py`, `architecture.py`, `extract_items.py`, `env.sh` |

lm_eval's full per-question records (prompts, log-likelihoods, generated answers) are attached to the `v1.0` release: `lm_eval_samples.tar.gz` (323 MB) and `lm_eval_samples_cards.tar.gz` (13 MB, the publisher-setting runs). LoRA adapters are not included; `run_all.sh` retrains them once the committed `runs/` is moved aside (see Reproducing).

## Reproducing

Requires Python 3.11 and one NVIDIA GPU with 24 GB. All runs used one RTX 4090 machine (`results/environment.txt`), except the publisher-setting runs (`cards.yaml`), which ran later on rented RTX 4090 machines with the same packages (their lm_eval results files record three CPU and driver configurations, one of them the main machine's); PyTorch 2.8.0 with CUDA 12.8, and fp32 weights for every evaluation and tuning run (bf16 for the architecture cache measurement). nanolm enables TF32 matrix products for bpb, SFT and DPO; lm_eval does not. nanolm's run records label the device "precision=bfloat16", but its fine-tuning code never casts.

```bash
bash scripts/setup.sh          # clone nanolm at commit 8418fc1 into external/, install requirements
hf auth login                  # models and datasets download from the Hugging Face Hub on first use
bash scripts/run_all.sh        # every stage; finished runs are skipped, so it resumes
python analysis/summarize.py   # results/summary.json (no GPU)
python analysis/figures.py     # figures/*.png (no GPU)
```

The committed records count as finished runs, so on a fresh clone `run_all.sh` only rebuilds the tables. To rerun everything, move them aside first: `mv runs runs.committed && mv results/architecture results/architecture.committed`.

- `run_all.sh` stages: `prepare bpb crosscheck benchmarks sft dpo tuned_benchmarks posthoc architecture analyze` (any subset can be named). `SLOTS` (default 4) jobs of the bpb, crosscheck, sft, dpo and posthoc stages and `BSLOTS` (default 2) benchmark jobs share the GPU; a benchmark job that fails, for example out of memory, is retried alone.
- Qwen3.5 runs on transformers' PyTorch implementation of Gated DeltaNet: flash-linear-attention and causal-conv1d are not installed.
- **Time.** Recorded GPU time is 2.3 hours of fine-tuning (30 runs) and 16.4 hours of lm_eval (116 runs); jobs shared the GPU, so these times overlap.
- **Without a GPU.** `summarize.py` and `figures.py` rebuild every number and figure from the committed `runs/` and `results/architecture/`, with the analysis packages of `requirements.txt` (numpy, pandas, scipy, matplotlib); other versions or platforms can change the last floating-point digits and the figures' layout and tick labels.

## Limitations

- **Scope.** One SFT and one DPO configuration and data set each, at nanolm's settings; LoRA only.
- **Evaluation.** 0-shot log-likelihood scoring (GSM8K: 5-shot generation, exact match with lm_eval's flexible-extract filter), without chat templates or chain of thought except in the publisher-setting test; every question of each evaluation set and the cloze form only for the base models; GSM8K comparisons of the instruct models and tuned adapters on 100 questions.
- **Samples.** Judged by counts over 50 held-out prompts and by reading, not by a reward model or human raters.
- **Statistics.** Paired t-tests over 5 seeds, McNemar tests and question-bootstrap intervals, without correction for multiple comparisons except where stated (`summary.json` also gives Holm-corrected counts for the model pairs and Holm decisions for the publisher-setting tests).
- **Models.** Below 0.4B parameters the models are GPT-2 and SmolLM2, from 0.49B up Qwen, so size and model family are confounded.
