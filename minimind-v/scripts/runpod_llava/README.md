# MiniMind-V: LLaVA 558K alignment → Instruct 150K SFT

This is the 2026-10-25 job, separate from the older ShareGPT4V experiment.
User selected LLaVA-Pretrain 558K and LLaVA-Instruct-150K, replacing the proposed ShareGPT4V 1.2M/100K job before that job began.

- Language base: completed Dense mixed SFT (`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`), SHA256 `a1d17222e66cf02cd2601d0b06b9cb76a455e185bee1b3dbc3725ca9c364a211`.
- Vision: frozen `google/siglip2-base-patch32-256`, pinned revision in `common.py`, 64 image tokens.
- Pretrain data: `liuhaotian/LLaVA-Pretrain`, `blip_laion_cc_sbu_558k.json` and original `images.zip`. This is the LAION/CC/SBU mixture, not CC3M alone.
- SFT: `liuhaotian/LLaVA-Instruct-150K`, `llava_instruct_150k.json` only; do not combine with its overlapping component JSONs or the 665K mixture. Original COCO train2017 images via the official S3 HTTPS endpoint.
- Revisions, source hashes, actual counts, exclusions, windows and supervised tokens are recorded. Missing/corrupt images abort preparation instead of silently dropping rows.

## Training and split policy

One projector-only alignment epoch at 4e-4, then one SFT epoch at 1e-5 updating the projector and language layers 0 and 7. All other language parameters and the entire vision encoder remain frozen. BF16, sequence 768, batch 64, accumulation 2, effective batch 128, cosine decay to 10%, AdamW, gradient clipping 1.0, seed 42.

Full multi-turn conversations are tokenized using the matching 6,400-token tokenizer. Empty thinking blocks are removed consistently for training and inference. Every window retains the initial image prefix. Long text is divided into overlapping windows; overlap labels are masked and short tails padded. Assistant answer tokens and EOS are supervised exactly once. Later windows may lack earlier conversation context; this is not unlimited-context training.

Decoded/resized RGB image hashes define groups. 256 image groups each form SFT validation and test, and are excluded from alignment. An additional 256 pretrain validation groups are excluded from both training phases. All records sharing a held-out hash are assigned together. Near duplicates and pre-existing language-pretraining leakage are not comprehensively detected.

## Verification and evaluation

`test_prepare.py` checks real-tokenizer short/long multi-turn target coverage, EOS and repeated image-prefix behavior:

```bash
cd minimind-v
LLAVA_TEST_TOKENIZER=/absolute/path/to/tokenizer python scripts/runpod_llava/test_prepare.py
```

Before training, production-size CUDA forward/backward smoke checks both freezing policies. Each stage verifies that frozen parameters stayed bitwise unchanged. Checkpoints are staged locally, copied under immutable stage/step names to shared-volume and SHA256-verified. The final FP16 checkpoint is reloaded before evaluation.

Evaluation: assistant-only validation PPL before/after, final held-out SFT test PPL, black-image ablation on the same text, and eight held-out image examples with greedy/sampling/black-image generations. These are diagnostics, not a general visual-QA accuracy benchmark. Samples retain their images, source IDs, prompts and reference responses for visual assessment.

## Artifacts and operation

- Pod: `afnr4hjc8io32k`, RTX 5090, dedicated to this job. Do not touch the separate MoE training Pod.
- Remote: `/shared-volume/minimind-v-llava-20261025`.
- Local: `outputs/minimind-v/llava-20261025` from repository root.
- `prepare.py`: download, image checks, held-out splits and window preparation.
- `train.py`: alignment, SFT, checkpoint checks, evaluation, samples and export.
- `watch.py`: durable local W&B logging, artifact download, hashes, publication and termination of only this task Pod. On a native preparation/training error it preserves diagnostics and stops this Pod; it never silently restarts partial training.
- `publish.py`: local public HF release after final tests and checksum verification.
- `inference.py`: release loader and single-image CLI. Requires the packaged `minimind_v/` source directory and `vision_encoder/`.

W&B and Hugging Face credentials remain local. Models, data, checkpoints and result archives are persistent on shared-volume. Raw datasets and training state are excluded from HF.

Planned public release: `caspar/MiniMind-V-Dense-64M-LLaVA-558K-150K-v1.0`, tag `v1.0`. Publication is not complete until `published.json` confirms public access and anonymous downloaded-file hashes. Only after publication, local export and W&B completion are verified will the supervisor terminate the task Pod and verify its absence. Shared-volume remains.

This task is asynchronous. A script's presence is not evidence that training, tests or publication succeeded. Check current `status.json`, `completion.json`, `published.json`, and `cleanup_verified.json` receipts.
