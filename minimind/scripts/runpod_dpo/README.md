# UltraFeedback DPO run

This run starts from the completed UltraChat SFT checkpoint and freezes a second
copy as the DPO reference. `train_dpo.py` is a run-specific copy of MiniMind's
trainer: it uses actual W&B logging, computes log-softmax in float32, frees
reference logits before the policy forward, and records reward/gradient metrics.

`prepare.py` pins the dataset revision, converts all `train_prefs`/`test_prefs`
rows, and checks both native empty-think variants at length 1024. It excludes
invalid pairs, mismatched prompts, fewer than 32 visible response tokens,
masked padding, and identical tokenized/truncated responses. Official labels,
including score ties, are preserved. Prompt IDs cannot overlap across splits.
A fixed hash-selected 512-pair test subset is used for before/after evaluation.

Configuration: hidden size 768, 8 layers, BF16, sequence length 1024, one epoch,
32 preference pairs per batch, accumulation 2, learning rate 5e-7, beta 0.1.

The launcher runs preparation, real-batch loss/backpropagation smoke testing,
baseline evaluation, native DPO training, final evaluation, and packaging.
`evaluation_smoke.json` checks identical policy/reference loss is ln(2) and that
gradients are finite without taking an optimizer step. Final evaluation checks
all weight tensors, strict model loading, resume completion, preference metrics,
and four greedy text generations. The reference-relative reward accuracy is
zero by definition before training; it is not a baseline win-rate measurement.
Raw sequence preference accuracy is also reported and is length-sensitive.

Remote output: `/shared-volume/minimind-dpo-ultrafeedback-20261020`.
Local output: `out/ultrafeedback-dpo-20261020`.
The local downloader verifies every file against `SHA256.json` and the model
against `evaluation_after.json`. Optimizer resume state and converted datasets
remain on the volume; the local bundle includes model, tokenizer, configuration,
manifest, environment versions, and evaluation reports.

The existing user-owned Pod is preserved after completion.
