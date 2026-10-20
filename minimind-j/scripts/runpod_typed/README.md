# LocalLLaMA/typed-decisions on Runpod

This runner trains the existing parallel-question MiniMind-J without adding a
separate decision head. The native LM projection predicts answer codes for all
five questions in one packed forward.

- Source: `LocalLLaMA/typed-decisions`, `all`, pinned revision
  `d9eda7f31ba227333213a1787f2c78fdaa4090c8`.
- Retains state, instructions, choice descriptions, score rubrics and noul meanings.
- Gold probability distributions are normalized for rounding and trained with soft
  cross entropy. Hard labels are used only for validation and readable outputs.
- Official train: 1,200 records. Seed-42 workflow-stratified split: 1,080 fit +
  120 validation records. Official test: all 400 records / 2,000 questions.
- Selects the best checkpoint by validation KL, with a maximum of 12 epochs and
  patience 3. Test results never select the epoch or hyperparameters.
- Default full finetuning: BF16 autocast, microbatch 4, accumulation 4, LR 2e-5
  cosine to 2e-6, AdamW weight decay .01, gradient clipping 1.0. No dropout in base.
- Starts from the exact config/tokenizer/native state_dict in `--base-root`.
  The requested run uses the user's existing MiniMind 768/8-layer SFT checkpoint,
  not the randomly initialized 115k educational example.
- Validates all packed lengths without truncating or dropping questions. Executes
  a real longest-record optimizer smoke, then restores the original model and
  creates a fresh optimizer before production training.
- Reports strict first-argmax accuracy, tie-aware accuracy, KL(gold || predicted),
  vector Brier (sum over options), 10-bin hard-accuracy ECE, and type/workflow
  breakdown. A prior baseline is fitted from fit rows only. These explicit metrics
  can differ from external scorers that use different tie/ECE conventions.

Run after transferring this entire project to a CUDA 12.8 / PyTorch 2.8 Pod:

```bash
python -m venv --system-site-packages /tmp/minimind-j-runtime
/tmp/minimind-j-runtime/bin/python -m pip install \
  transformers==4.57.6 pyarrow==21.0.0 'pytest>=8,<9'
/tmp/minimind-j-runtime/bin/python -m pytest -q
/tmp/minimind-j-runtime/bin/python scripts/runpod_typed/launch.py \
  --root /shared-volume/minimind-j-typed-decisions-20261020 \
  --base-root /shared-volume/minimind-sft-ultrachat-200k-20261020
```

The launcher writes `exit_code`; completion requires `exit_code=0`, `SUCCESS`,
`report.json`, a standalone `model/` and a re-read verified `SHA256.json`. The
native base is read-only and its hash must stay unchanged. This job requires the
requested 30–40 GiB GPU slice and refuses to silently run on CPU.

`model/` can be loaded using ordinary `load_checkpoint()` or `decide.py`.
To infer on an adapted record, remove each question's `answer` and optionally
`target_probabilities` and pass the JSON with `--request`. Labels/probabilities
are never serialized into model input regardless of whether you remove them.

The control-plane credentials stay local; no Runpod API key is sent to the Pod.
Task-specific monitoring verifies all shared-volume file hashes, downloads the
small reports and source/data/tokenizer bundle, verifies those local hashes, then
removes only the task's created Pod and checks that GET returns 404. Weights stay
on shared-volume. Existing Pods and the existing shared volume are preserved.

## Question-weighted accumulation and robustness checks

Accumulation now backpropagates each microbatch's **sum** of question losses,
then divides gradients by the actual question count in that accumulation window
before clipping and stepping. Unequal question counts and final short batches
therefore have the same weighting as a single combined question batch. Existing
saved checkpoints are unchanged; this applies to future training runs.

To evaluate the trained checkpoint on the shared volume without retraining:

```bash
python scripts/evaluate_robustness.py \
  --checkpoint /shared-volume/minimind-j-typed-decisions-20261020/model \
  --data /shared-volume/minimind-j-typed-decisions-20261020/data/test.jsonl \
  --device cuda --batch-size 4 \
  --output /tmp/minimind-j-typed-robustness.json
```

This evaluator also supports CPU. It uses FP32 model loading and inference;
original training reports used BF16 autocast, so its freshly computed baseline
is the appropriate comparison for its two variants. No remote resources are
created by this command.

`scripts/typed_paraphrases.json` explicitly maps the 20 original instructions to
alternate wording. Only the instruction changes; the rubric, state and gold
labels stay fixed. Missing mappings raise an error. The other variant reverses
choice lists and their soft target vectors, retaining the answer value. Score
levels and boolean order stay fixed. The script evaluates each variant separately,
without combining interventions or selecting a model using these test results.

Outputs include accuracy, tie-aware accuracy, KL, Brier, ECE, type/workflow
breakdowns, answer change rate, choice-only answer change rate, and mean total
variation distance after aligning probabilities by answer value. Answer changes
use actual first-argmax decoding, including ties. Accuracy metrics follow the
existing runner's first-argmax gold convention in each presented option order;
use tie-aware accuracy for ambiguous gold distributions. Hard-label datasets
use one-hot gold distributions. Smaller sensitivity is better, but unchanged
wrong answers are still wrong; inspect accuracy and KL together.

A single fixed reversal and one paraphrase per instruction measure limited
robustness, not unseen-schema generalization or all possible permutations.
The paraphrases are explicit authored fixtures, not a certified semantic benchmark.
