# GSM8K MoE to Dense soft distillation

Run from the `minimind/` project. `sources.json` pins the requested Hugging Face
teacher, student, and released mixed-SFT dataset. Both models have identical
6400-token tokenizers. The source release exports GSM8K first; select only
`dataset == gsm8k` from its first train parquet and assert completeness against
its provenance: 6851 conversations, 7333 windows, 845130 assistant targets.
Verify parquet SHA256; never replace requested data with raw upstream train.

Use original chat template, remove empty think blocks, 340-token windows with
stride 256. Mask prompts, padding and already-covered overlap; retain final
short windows and each assistant target once. Continuation windows can omit
part of the question. No test examples are used for optimization or selection.

Full student optimization, frozen/eval teacher, BF16 autocast with FP32 weights,
AdamW weight decay .01, three epochs, seed 42, microbatch 16, accumulation 4,
cosine LR 5e-6 to 5e-7, clip 1. Loss is 0.5 CE + 0.5 T² KL(teacher || student),
T=1.5, computed in FP32 on shifted assistant targets. Normalize accumulation by
actual target-token count, including the partial final batch. Total 345 updates.
Checkpoint every epoch. Final epoch is fixed in advance, without test selection.

Evaluate all 1319 released `validation_gsm8k` rows (official GSM8K main/test),
zero-shot greedy, original suffix, max 256 new tokens, total context 340, batches
with exactly equal prompt lengths. Report strict #### numeric accuracy and
secondary last-number accuracy; invalid output/overflow remain in denominator.
Report assistant CE/PPL and teacher KL before/after. Final exported FP16 weights
are reloaded and tensor-verified before final evaluation. Baseline is the original
Dense student. The test split name does not make it training validation data.

`python -m unittest discover -s scripts/runpod_gsm8k_distill -p test_distill.py -v`
checks loss shifting/masking, KL identity and answer parsing. `preflight.json`
records actual CUDA backward, nonzero KL and frozen teacher verification.
`completion.json` is created only after training, reload, and both full evaluations.

Remote root: `/workspace/moe-dense-gsm8k-distill-20261025` on existing volume
`aiz2tt6eps`. Task Pod: `cqko3psy4ps2w8`, A40, quoted GPU $0.59/hour.
Local results: `outputs/moe-dense-gsm8k-distill-20261025`.
Export and verify hashes, then remove only the task Pod; preserve existing volume.
HF upload runs locally using existing credentials, after results are verified.
