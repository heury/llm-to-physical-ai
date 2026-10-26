# Dense MiniMind UltraFeedback short-context DPO

Base: `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` at the recorded immutable revision.
Source: `argilla/ultrafeedback-binarized-preferences-cleaned`, also revision pinned.
This experiment is independent of the older `runpod_dpo` UltraChat/1024 run.

1. `prepare.py` scans all 60,917 original pairs using the actual 6,400-token base
   tokenizer. Both full prompt+response sequences must be strictly shorter than
   340 tokens, including chat markers and EOS. No response truncation is used.
2. Hash-seeded selection creates exactly 3,000 unique prompt groups: 2,700 train,
   150 validation, 150 test. Labels/scores are retained. The empty think block is
   removed deterministically, matching the base model. Only final-assistant
   completion tokens and EOS/newline contribute to sequence log probabilities.
3. `check_data.py` verifies every selected input and response mask, split
   disjointness, and DPO gradient direction.
4. `train.py --root /shared-volume/minimind-dense-dpo-short-20261025` uses a frozen
   copy of the base as reference. Cached reference scores are checked against a
   real backward batch with identical policy/reference loss = ln(2). Full-parameter
   BF16-autocast DPO runs for two epochs, 32 pairs/batch, LR 5e-7, beta 0.1.
   FP32 master/export weights preserve small updates. The best trained epoch is
   selected by validation DPO loss; test data is not used for model selection.
5. Results include before/after preference likelihood metrics, 16 fixed greedy
   generation comparisons, reference immutability, weight-difference checks and
   exact weight/score comparison after reloading the exported model.

Reference-relative reward accuracy is zero/tied for an identical policy/reference
and is not a baseline win rate. Raw sequence likelihood comparison favors shorter
responses; token-normalized comparison is also reported. These 150 heldout pairs
are a small DPO-only evaluation, not a decontaminated general benchmark.

Local artifacts: `outputs/minimind/dpo-short-20261025/`. The raw upstream source is
cached locally; only the selected dataset and necessary model files are published.
Model/data/results persist on the existing global volume. Only this task's Pod is
removed after public upload, verification and local artifact capture.

Global Volume does not implement chmod. The initial Transformers custom-code
export hit that filesystem limitation after training/evaluation completed.
`recover_export.py` reloads the already-trained best checkpoint, verifies all150
stored test scores, exports on local container disk, reloads and verifies exact
weights/scores, then copies bytes to the global volume. It never retrains.
`train.py` now uses the same local-export/copy procedure for future runs.
The recovered timing is the elapsed time through the last optimizer update,
including inter-epoch validation; original peak training GPU memory was not saved.
