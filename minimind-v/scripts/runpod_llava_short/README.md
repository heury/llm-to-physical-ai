# Short-caption MiniMind-V experiment — 2026-10-25

This job replaces the stopped full LLaVA experiment. It starts from the verified Dense mixed-SFT language checkpoint, not a partially trained VLM. Frozen SigLIP2 provides 64 image tokens.

## Filters

`filter_data.py` records source rows/turns, selected JSON hashes and rejection counts.

- Pretrain: official `blip_laion_cc_sbu_558k.json`, pinned revision in `common.py`. Count whitespace-delimited words in the assistant caption; require 1–19 words. Uniformly select exactly 50,000 from 548,088 eligible rows using seed 42. The selection includes held-out rows, so actual training count is smaller.
- SFT: screen `conversation_58k.json` and `detail_23k.json` in the pinned LLaVA-Instruct-150K repository. Do not load `complex_reasoning_77k.json` as training data; use it only to reject matching image/question/answer triples.
- Extract independent QA turns. Answers must have 1–2 sentences by a conservative punctuation heuristic and at most 40 words; questions at most 30 words. Allow direct identification/location/count/yes-no questions; reject reasoning/speculation terms, unresolved reference words, long/list answers and uncertainty phrases. Original answer text is retained, not summarized or truncated.
- Final SFT selection: **150,607 QA pairs from 55,507 images**. All selected pairs came from conversation_58k; no detail_23k row passed. QA count is not original conversation count: one source conversation can yield multiple independent pairs.
- This is an auditable rule-based filter, not human semantic labeling. It cannot guarantee every selected answer is visually correct.

All selected images are checked. Decoded 256×256 RGB image hashes define held-out groups: 256 SFT validation, 256 SFT test, 256 pretrain validation. These hashes are excluded from the respective training phases. Exact resized duplicates are grouped; arbitrary near duplicates are not detected. Original shared-volume image archives from the stopped experiment are reused and SHA256 verified without modifying them.

## Training / evaluation

Length 340; BF16; batch 64, accumulation 2 (effective 128). One projector-only alignment epoch at 4e-4, followed by one SFT epoch at 1e-5 updating projector and language layers 0/7. Vision and other language weights remain frozen. Full assistant targets/EOS are retained; long overflow windows repeat image prefix and mask overlapping labels.

Tests: `python scripts/runpod_llava_short/test_filter.py`; then with the real tokenizer:

```bash
cd minimind-v
LLAVA_TEST_TOKENIZER=/absolute/path/to/tokenizer python scripts/runpod_llava_short/test_prepare.py
```

Evaluation after final FP16 reload: held-out text PPL with original images versus black-image controls, plus eight actual test-image samples (greedy, sampled, black controls). PPL is not visual QA accuracy. Images, prompts, references and complete outputs remain in the local result bundle for review.

## Operation

New task Pod: `4d4iep7ygw7otm`, RTX 5090, EU-RO-1. The earlier stopped Pod `afnr4hjc8io32k` remains stopped and is never modified by this supervisor.

- Remote: `/shared-volume/minimind-v-llava-short-20261025`
- Local: `outputs/minimind-v/llava-short-20261025`
- W&B project: `MiniMind-V-LLaVA-Short`
- Planned public model: `caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0`, tag `v1.0`

The local supervisor exports and verifies the final bundle, publishes only inference files, anonymously verifies the upload and then terminates **only the new task Pod**. It stops the task Pod on native failure and preserves diagnostics. No credentials are copied to the Pod. Shared-volume assets are retained.

Check `status.json` for live progress. `completion.json`, `published.json` and `cleanup_verified.json` are separate success receipts; their absence means that step is not yet complete. No result is implied by this README or the presence of scripts.
