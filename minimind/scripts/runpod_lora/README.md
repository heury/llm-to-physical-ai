ChatDoctor LoRA training for the existing 768-hidden, 8-layer MiniMind SFT model.

Source: `lavita/ChatDoctor-HealthCareMagic-100k`, revision
`505443eac4e99ccedeffbb6f640061223d1d4bb3` (112,165 source rows).

`run.py prepare` downloads the pinned Parquet, removes repeated questions and
examples with fewer than 32 visible answer tokens, and persists token arrays and
the conversion manifest. The original instruction becomes the system message.
Assistant generation prefixes must exactly match the tokenized training prefix.
Only the answer and its end marker are supervised; prompts and padding use -100.
Validation questions are selected by SHA256 and never appear in training.

`run.py smoke` uses a disposable model to check a real 16-example backward pass,
finite gradients, adapter updates, frozen base parameters, and GPU memory.

`run.py train` evaluates the unmodified SFT base, trains native MiniMind rank-16
LoRA on square linear projections for one complete epoch (batch 16, accumulation
2, BF16 autocast, cosine LR 1e-4 to 1e-5), and evaluates the same 256 validation
examples. Only adapter parameters enter AdamW. Training uses an explicit CUDA
allocator limit of 17% to share the existing A40 with a separate GRPO process.
The existing Pod and other jobs must be preserved.

W&B project: `MiniMind-ChatDoctor-LoRA`. The real `wandb` package records loss,
gradient norm, progress, learning rate, validation NLL/perplexity and the final
adapter artifact. Authentication is supplied through the environment, never
through source or configuration artifacts.

Volume root: `/shared-volume/minimind-lora-chatdoctor-20261020`.
Runtime: `/tmp/minimind-lora-chatdoctor-20261020`.
Base: `/shared-volume/minimind-sft-ultrachat-200k-20261020/out/full_sft_768.pth`.
Set `PYTHONPATH` to the isolated runtime `repo` before running these commands.
The base SHA256, exact tokenizer, split statistics, settings, source, and final
evaluation accompany the artifacts. Final outputs include the FP32 adapter,
its optimizer checkpoint, a strict-load-verified merged FP16 model, and a SHA256
manifest. The local downloader verifies every packaged file against that manifest.

These training and validation losses measure prediction of held-out dataset
answers, not clinical reliability or factual correctness.
