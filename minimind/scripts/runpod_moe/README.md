# MiniMind MoE on B300 MIG 34GB

Uses the existing `trainer/train_pretrain.py::train_epoch` and native MoE implementation.
Fresh model: hidden 768, 8 layers, 4 experts per layer, top-1 routing, native
straight-through router gradient and auxiliary coefficient 5e-4. Existing trained
6400-token FineWeb tokenizer is verified and reused.

Source: `/shared-volume/fineweb-edu-10BT-score-3.5/pretrain-score-3.5.jsonl`.
1,288,228 documents; score >= 3.5; source revision
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9; SHA256
4787645d1b783b98f724d7a9f83bf591ea591538d9f86e6f7371fcb3511821a7.
The complete local copy must match this hash. A seeded 256-document holdout is
excluded from training. All remaining 1,287,972 documents are used for one epoch,
with the native dataset's first 1022 tokens plus BOS/EOS (sequence length 1024).

BF16, LR 3e-4, 2% warmup capped at 1000 microsteps, cosine decay to 3e-5,
effective batch 128, gradient clipping 1. Candidate batches 16 and 32 are tested
with real data, backward, router/expert gradients and AdamW optimizer allocations.
The fastest candidate below 85% GPU capacity is chosen. Production model is freshly
seeded after the disposable benchmark. Actual settings are training_settings.json.
Native checkpoints every 250 microsteps and at the last step; an additional first
optimizer-update checkpoint validates that training and persistence work early.

Pod: yj7u51c5os0wng, runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404,
B300 MIG 1g.34gb, US-WA-2. Shared volume cmuxpmy6p000007l12uzr990b.
Remote root: `/shared-volume/minimind-moe-fineweb-20261020`.
Runtime: `/tmp/minimind-moe-20261020`. Local result:
`outputs/minimind/fineweb-moe-20261020`.

run.py evaluates the fixed holdout before/after training, verifies full epoch and
optimizer update count, all finite model/optimizer tensors, strict model reload,
weights matching the resumable checkpoint, and two text continuations. It packages
weights, optimizer checkpoint, tokenizer, configs, source, resolved dependencies,
logs, evaluations and SHA256. The detached shell writes exit_code.

watch.py waits for successful final packaging, downloads and verifies all hashes,
then terminates ONLY the above task-created Pod and verifies its absence. Other
Pods and the shared volume are preserved. Local cleanup.json is the final receipt.
The local watcher uses existing /tmp/moe-pod.json, /tmp/moe_proxy.py,
/tmp/minimind-runpod-cli.py and SSH key locator; no secrets are copied to artifacts.
