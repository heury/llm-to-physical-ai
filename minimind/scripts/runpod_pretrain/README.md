# FineWeb Dense then MoE pretraining on RTX 5090

Task Pod: `g1ol0tfb8yk3ik`, RTX 5090 32 GB, EU-RO-1, $1.19/hour.
Image: official PyTorch 2.8 template image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
Global Volume `cmuxpmy6p000007l12uzr990b` is named `shared-volume` and mounted
at `/shared-volume`. It survives task Pod termination.

Output root: `/shared-volume/minimind-pretrain-rtx5090-20261023`.
Source: `/shared-volume/fineweb-edu-10BT-score-3.5/pretrain-score-3.5.jsonl`.
The script verifies its complete SHA256 against the conversion manifest and
reuses the source's existing 6400-token tokenizer. Both models start from fresh
seeded weights. A fixed 256-document validation set is excluded; every remaining
document is used for two epochs per model. Native PretrainDataset truncates each
text to 338 tokens, adds BOS/EOS, and pads to the user-requested **340** tokens.

Order: Dense (768 × 8), verify final checkpoint, then MoE (768 × 8, four experts,
top-1 routing), verify final checkpoint, finish W&B, terminate only the task Pod.
Both use BF16, LR 3e-4, 2% warmup capped at 1000 microsteps followed by cosine
decay to 3e-5, effective batch 256, clipping 1, and four data workers.
Each model benchmarks batches 32/64/128 with real forward/backward and AdamW;
the fastest candidate using less than 85% VRAM is selected. Benchmark weights
are discarded before production training. Actual settings, timings and memory
use are recorded separately in `{dense,moe}/training_settings.json`.

The native `trainer.train_pretrain.train_epoch` performs training. Runtime wrappers
route checkpoints to the global volume, save after the first optimizer update
and every 250 microsteps, record progress and use the warmup schedule.
Final verification checks full-epoch step/update counts, finite model and optimizer
tensors, strict reload and equality of inference and resumable weights.

W&B project: `MiniMind-FineWeb-Pretrain`, separate online runs for Dense/MoE.
Run URLs: `{dense,moe}/wandb_run.json`; local W&B records persist under `wandb/`.
Final weights: `dense/out/pretrain_768.pth`, `moe/out/pretrain_768_moe.pth`.
Resumable checkpoints: `{dense,moe}/checkpoints/`.
`status.json`, `training.log`, `{dense,moe}/progress.json` record live state. Resumption restores optimizer/scaler and the same W&B run; an uncommitted partial gradient accumulation window is replayed.
`completion.json` is written only after both models pass final verification.
`cleanup.json` records the termination request; external read-back is needed to
confirm deletion because a Pod can disappear before writing its own receipt.
On failure the Pod remains available with logs/checkpoints for diagnosis.
Credentials are staged only in a mode-700 directory on ephemeral container disk.

This is a long-running launch: status files and live observations, rather than the
presence of these scripts, determine whether training has started or completed.
