# GSM8K GRPO reward comparison

Run dated 2026-10-25. Each policy starts independently from the pinned
`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` or corresponding MoE checkpoint.
User selected both architectures. Order: Dense rule, Dense Skywork, MoE rule,
MoE Skywork. The two reward modes use rule reward and
`Skywork/Skywork-Reward-V2-Qwen3-1.7B`. The second experiment does not continue
from the first experiment's policy. `run.py` records all revisions and settings.

Data: `caspar/MiniMind-Mixed-SFT-Training-v1.0`, revision
`542384d27d374d668413d8ade5b9973b7b063bee`, retained GSM8K train 6,851 full
question/answer pairs. Official validation GSM8K 1,319 questions are used only
for before/after evaluation, never reward optimization or model selection.
`prepare_local.py` checks the source Parquet SHA256 against the release manifest.
Questions already carry the step-by-step/final-answer instruction. No SFT token
windows are used. No questions are truncated and gold answers never enter the
policy or Skywork prompt.

One full epoch for each reward, batch 32 questions, four rollouts per question,
backpropagation microbatch 8, FP32 weights with BF16 autocast, AdamW LR 1e-6
cosine to 1e-7, weight decay .01, gradient clip 1, reference KL beta .04,
GRPO clip .2, seed 42. Length-sorted batches are shuffled identically in both
runs; the incomplete final batch is included. Expected 215 updates per run.
Generation is unwarped sampling (temperature 1, top-p 1, top-k 0), at most 256
new tokens, and at most 768 prompt tokens. The source SFT used 340-token
windows; this experiment uses longer contexts, which must be reported.

Rule rewards use `trainer/gsm8k_reward.py` directly. Skywork is frozen, uses its
own chat template without system messages and returns raw logits. No format,
length, repetition bonus, or correctness reward is added to the Skywork score.
Group normalization is applied by GRPO. Equal-reward groups give no task-reward
policy gradient; KL still applies. Logs retain correctness/format diagnostics,
reward variance, KL, gradient norm, sampling ratio checks and length exhaustion.

Rollout EOS masks include the first EOS and exclude tokens after it. Policy,
reference likelihoods use identical masks, precision, microbatch
shape and dropout-free computation. For one optimizer update per rollout,
old-policy log probabilities are detached from the same current-policy forward.
MoE retains its native router auxiliary loss during the gradient forward. Every initial ratio must be within 1e-3 of
one; the initial reference KL must be below 1e-4. Checkpoints include optimizer
and RNG state, are hash-verified after durable copy and retain the last two.

Baseline and final evaluation use greedy decoding on all 1,319 held-out
questions with the same strict rule scorer. Evaluation batches are length-sorted
and left-padded. Report both correctness and format/budget failures. No claims
of model success are made from training reward alone. Final package verification
requires exact weight and forward-logit equality after reload; final evaluation
uses the reloaded package. Model cards and reports contain full settings and
sample responses. Published model artifacts exclude optimizer state, account
credentials and infrastructure metadata.

Run artifacts: `outputs/gsm8k-grpo-20261025` locally and
`/shared-volume/minimind-gsm8k-grpo-20261025` remotely. `control.py` and `remote.py`
are local-only helpers. Only explicitly packed source/data are sent to the Pod.
The newly created task Pod is removed after outputs are exported and verified;
the existing shared volume is preserved.

## Recovery

Dense rule completed215updates. Dense Skywork hit a redundant likelihood
recomputation mismatch and resumes from hash-verified update50 with model,
optimizer and RNG restored. Uncommitted metric rows after50 are archived as
rollback diagnostics and excluded from the final run history. The stopped
original Pod could not reacquire a GPU; it was removed with a fresh404 read.
A new A40 at the same $0.59/hour rate mounts the same preserved volume.
No source weights or completed Dense rule results are overwritten.

## Completed results

All four full-epoch runs and 1,319-question evaluations completed. Dense correct
counts: baseline14, rule18, Skywork12. MoE: baseline16, rule26, Skywork32.
Public model links, pinned commits, complete samples and recovery records are in
`../../outputs/gsm8k-grpo-20261025/report.md`. `completion.json` and the four
`*-published.json` receipts verify publication. Both task Pods were deleted;
the existing shared volume was preserved.
