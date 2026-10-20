# Typed Decisions GPU Training Plan

Run the existing parallel-question MiniMind-J on the requested B300 MIG 34GB.
Initialize from the user's existing native MiniMind SFT 768 checkpoint and exact
saved tokenizer/config, without modifying that base. Preserve all five questions,
rubric descriptions and soft gold distributions from pinned public dataset revision
d9eda7f31ba227333213a1787f2c78fdaa4090c8. Train/validation split comes only from
`all/train`; official `all/test` is final evaluation only. Full finetuning with
BF16 autocast, small real-GPU memory smoke before training, validation KL selection,
and maximum 12 epochs with patience 3. Report accuracy, KL, vector Brier, 10-bin
ECE and type/workflow breakdown, plus a train-only prior baseline. Save standalone
model, tokenizer, exact source, data/provenance, logs and SHA256 manifest on existing
global shared-volume. Verify reload and file hashes, download small reports, and
remove only this task's newly created Pod. Preserve all other Pods and volumes.
