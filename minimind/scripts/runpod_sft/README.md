# Parallel UltraChat SFT of the trained Dense and MoE models

Requested run: 2026-10-24, Asia/Seoul. Two independent RTX 5090 32 GB Pods,
Secure Cloud, $1.19/hour each (Dense EUR-NO-1, MoE EU-RO-1):

- Dense: `combumjy25nsrs`.
- MoE: `yi83nssnsjubae`, EU-RO-1 (replaces terminated `io4ugdtgyt1gy7` and `vttdsbn46qhj3d`, which failed CUDA initialization on the same host before training).

Image: `runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404`, observed on the
account before deployment. Global Volume `cmuxpmy6p000007l12uzr990b`, named
`shared-volume`, is mounted on both Pods at `/shared-volume`.

Base weights are the newly trained two-epoch checkpoints under
`/shared-volume/minimind-pretrain-rtx5090-20261023/{dense,moe}/out/`.
Their SHA256 is checked against their training completion records before loading.
The same original 6400-token tokenizer is reused, with strict model reload.
This is full-parameter SFT, not LoRA or a fresh model.

Output root: `/shared-volume/minimind-ultrachat-sft-20261024`.
Local results: `outputs/minimind/ultrachat-sft-20261024` from the repository root.

## Data and masking

Dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/tree/main/data
Revision: `8049631c405ae6576f93f445c6b8166f76f5505a`.
Only the three `train_sft` shards train the models. `test_sft` is never used for
training. `train_gen`/`test_gen` are not SFT targets and are excluded.

`prepare.py` records raw shard SHA256, checks train/test prompt-ID disjointness,
and freezes the native SFTDataset template, assistant-only loss mask, seeded
system/think augmentation, and truncation into shared NumPy arrays. The user's
maximum input length **340** is retained. Conversations whose first 340 tokens
contain no shifted assistant target are excluded and counted explicitly; long
conversations are truncated, not claimed to be fully consumed. A fixed seeded
512-document subset of valid official `test_sft` rows is used for evaluation.
Both jobs consume the exact same arrays and hash-check their local copies.

## Training and testing

Two epochs, BF16, LR 5e-5, 3% warmup followed by cosine decay to 5e-6, effective
batch 128, gradient clipping 1. Each model benchmarks batches 32/64/128 with
real forward/backward and AdamW state; the fastest below 85% VRAM is selected.
The benchmark weights are discarded; production reloads the exact pretrain
checkpoint. Training uses native `trainer.train_full_sft.train_epoch`.

First-update and periodic resumable checkpoints are persisted on the Global
Volume, with a separate inference checkpoint for each epoch. Actual settings,
progress, validation loss and tests live in `{dense,moe}/`.
Final verification checks expected epoch/update counts, finite model/optimizer
tensors, strict reload, and inference/resumable weight equality. Tests after
SFT reload the saved FP16 checkpoint into FP32 parameters.

Before/after English tests compare assistant-only loss/PPL on the fixed official
holdout; twelve handwritten factual likelihood probes; and twelve chat prompts
in greedy and sampled modes, covering explanations, extraction, formatting,
arithmetic, reasoning, short writing and summarization. Generation uses seed 42,
128 new tokens maximum, temperature 0.8, top-p 0.9, top-k 40, repetition penalty
1.05. Full answers are kept for human review. The handwritten probes are not a
standardized benchmark and PPL alone does not measure instruction quality.

## W&B, results and cleanup

No user API credentials are copied to either Pod. W&B writes offline records
on the Global Volume; `watch.py` downloads completed result bundles, verifies
archive and file hashes, then syncs W&B from the local machine using its existing
`.env` key. Project: `mind4u/MiniMind-UltraChat-SFT`.

The watcher verifies the completed W&B run summary on the server. If an offline
sync is not visible there, sync_verify.py replays the original offline history
with its original run ID and training timestamps before verifying again.

The watcher terminates only the two named task Pods, after tests, verified result
export and server-verified W&B sync, and verifies 404 from fresh Pod reads. Checkpoints
and raw results remain on the Global Volume. Each Pod may be terminated when its
own job is complete; the other job continues independently. Other account Pods
and volumes are preserved. Failures are recorded in status/log files for diagnosis.

`completion.json` means training, checkpoint verification and before/after tests
finished. `export_ready.json` means the result bundle is ready. Local
`cleanup_verified.json` confirms actual Pod termination and W&B synchronization.
The presence of these scripts does not establish a successful run.
