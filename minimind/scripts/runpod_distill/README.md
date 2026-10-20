# UltraChat distillation on Runpod B300 MIG 34GB

Runs `trainer/train_distillation.py::train_epoch` without changing the repository trainer.
Frozen teacher: completed UltraChat SFT, hidden 768, 8 layers. Student: hidden 768,
4 layers, initialized from teacher layers 0, 2, 4, 6 (other tensors copied exactly).
Both share the existing trained 6400-token tokenizer.

Dataset: HuggingFaceH4/ultrachat_200k, revision
8049631c405ae6576f93f445c6b8166f76f5505a. Reuses the SHA256-verified SFT conversion:
207865 source train_sft rows, 197944 retained with at least 32 assistant tokens
visible in the 1024-token window; 9921 excluded without sufficient visible targets.
Fixed 256-document validation from test_sft. No train_gen or test_gen examples.

One epoch; BF16; batch 32; accumulation 2; cosine learning rate 5e-6 to 5e-7;
CE coefficient .5; KL coefficient .5; temperature 1.5; gradient clipping 1.
GPU memory must pass an actual batch-32 forward/backward check before training.
The native trainer logs every 10 batches and checkpoints every 250 and the last batch.
Logging: training.log and metrics.jsonl; this task does not require W&B.

Remote root: `/shared-volume/minimind-distill-ultrachat-20261020`.
Runtime: `/tmp/minimind-distill-20261020/repo` with inherited-system-torch venv.
Execute `DISTILL_ROOT=<remote-root> <venv-python> -u scripts/runpod_distill/run.py`.
The detached launch shell records the final process exit code.

`watch.py` is a task-specific local watcher for Pod zkekvrhwoi7xff. It downloads the
final bundle only after successful training/evaluation, verifies every SHA256,
checks full epoch completion and optimizer update count, then terminates only that
Pod and verifies its absence. The shared volume and all other Pods are preserved.
It uses `/tmp/distill-pod.json`, `/tmp/distill_proxy.py`, the existing credential
loader `/tmp/minimind-runpod-cli.py` and SSH key locator; no credentials are stored
in source or artifacts. The local watcher is detached; caffeinate -i -w holds idle sleep until it exits.

Local result: `outputs/minimind/ultrachat-distill-20261020`.
Artifacts include student weights, resumable optimizer checkpoint, tokenizer,
model configuration, source snapshots, dependency versions, evaluations and hashes.
A result is complete only when `cleanup.json` reports `pod_deleted: true`.
