# GSM8K continuation SFT and math evaluation

Start from the Dense and MoE **UltraChat SFT** final checkpoints under
`/shared-volume/minimind-ultrachat-sft-20261024`, checking their completion-record
SHA256 before loading. Preserve the original 6400-token tokenizer and model
configuration. Each job runs on its own RTX 5090; full parameter SFT, 2 epochs,
BF16, batch64 x accumulation2, AdamW LR5e-5, 3% warmup then cosine to5e-6,
gradient clipping1, seed42. No API credentials are sent to the Pods.

Dataset: https://huggingface.co/datasets/openai/gsm8k (`main`), revision
`740312add88f781978c0658806c59bc2815b9866`. All7473 training rows are used;
all1319 official test rows are held out and never drive checkpoint selection.
No extra data augmentation, calculator, or external solver.

Use original tokenizer chat template with `open_thinking=False`. The user prompt
is the original question plus `Solve step by step. End your answer with ####
followed by the final number.` Targets are the unmodified original rationale
and final answer. Native `SFTDataset.generate_labels` masks user/padding tokens.
To respect the existing 340-token maximum without losing final answers, freeze
340-token windows with stride256, 84-token overlap. Assistant targets in the
overlap are masked after their first occurrence. An assertion proves every
shifted assistant target is included exactly once. Continuation windows may
contain a suffix of the question or rationale rather than the whole question.
Hashes, exact row/window/target counts and split-disjointness are recorded.

Before SFT and after reloading the final saved checkpoint, evaluate all1319
questions with zero-shot greedy generation, at most256 new tokens, with total
context bounded by340. Prompts are never truncated; questions leaving no output
room are recorded as failures in the full denominator. Batches group exact
prompt lengths, without padding. Report context overflows, missing final-answer
format, and exhausted token budgets. No test answers enter model prompts.
Primary accuracy requires the last valid `#### number` line (Decimal-normalized).
A secondary permissive metric falls back to the last generated numeric token.
This is a custom zero-shot, 340-token protocol, not directly comparable with
leaderboard results using other prompts/context lengths/few-shot examples.
Also report assistant-only test loss/PPL (all heldout answer tokens in windows),
six handwritten elementary-math probes and complete generated answers.

The final inference weights, resumable optimizer state and each epoch's weights
remain under `/shared-volume/minimind-gsm8k-sft-20261024/{dense,moe}`. Final checks
require finite weights/optimizer, strict reload and exact checkpoint/resume
weight equality and expected update counts. Do not infer success from scripts.

`watch.py` exports verified result archives, locally synchronizes offline W&B
records to `mind4u/MiniMind-GSM8K-SFT`, verifies server receipt and then terminates
only the two task Pods, verifying fresh404 reads. Other resources are untouched.
Local artifacts: `outputs/minimind/gsm8k-sft-20261024`. See scenario003 for actual
execution evidence when available. Never upload `.env` or local SSH keys.

## Environment verification

Use the catalog's official PyTorch2.8/cu128 image. Pin PyArrow19.0.1 alongside
NumPy1.26.4; unconstrained newer PyArrow requires NumPy2 and failed import
before any training. `test_common.py` verifies target identity/order across
window boundaries and numeric extraction edge cases. Run from repository root:
`python3 -m unittest discover -s minimind/scripts/runpod_gsm8k -p test_common.py -v`.

## Checkpoint publication recovery

Dense completed and verified its original shared-volume checkpoints. The first
MoE attempt finished training but reading `resume.pth` still returned update100
instead of126, so final validation stopped and no success was claimed. The same
Pod retrained **from the original UltraChat base** for2epochs, reusing only the
already measured baseline for that exact hash-verified base. This is not4epochs
of continuation. `recovery.json` records the initial W&B ID and reason.

`run.py` now stages inference and optimizer checkpoints in `/tmp/minimind-gsm8k`,
checks the exact final epoch/update/weights locally, and copies them to fresh
`*_verified.pth` / `resume_verified.pth` files on shared-volume. Source and
published SHA256 must match before evaluation. The final MoE completion record
names the verified file; the original attempt's files are diagnostic artifacts.
The successful rerun uses a separate offline W&B directory and
`metrics_verified.jsonl`; only that W&B stream is exported as the completed run.
`GSM8K_REUSE_BASELINE=1` is only for this recovery with unchanged source hashes,
dataset revision, prompt and evaluation conditions.

## Completed run

Both jobs completed2epochs/126optimizer updates and saved5276 main evaluation
generations (2models x before/after x1319questions), independently re-scored
locally. Strict final-answer accuracy: Dense0→23/1319 (1.74%), MoE0→18/1319
(1.36%). Permissive numeric extraction: Dense16→32/1319 (2.43%), MoE10→27/1319
(2.05%). Both failed all6 handwritten arithmetic probes. Report actual
math reliability as low despite reduced assistant-token PPL.

Final MoE weights are `moe/out/gsm8k_sft_768_moe_verified.pth`; use that path
from completion.json rather than files from the initial failed verification.
W&B completed runs: Dense8i4a0sa6, MoEm8pcou50 in MiniMind-GSM8K-SFT.
Task Pods m0e7atpx27qeh4 and45tgcv3dmj6xz5 were terminated with fresh404
reads; shared-volume is preserved. See local report.md, completion.json,
independent_scoring_verified.json, final_verification.json and scenario003.
