# Three-source Dense/MoE full SFT

Requested2026-10-24. Two independent Secure Cloud RTX5090 Pods at the current
catalog rate$1.19/hour each. Continue from the **full-document pretraining** final
weights under `/shared-volume/minimind-full-pretrain-20261024`, checking their
completion-record SHA256, config and original6400-token tokenizer. Do not use the
older UltraChat/GSM8K models. Output:`/shared-volume/minimind-mixed-sft-20261024`.

## Frozen data and loss

- HuggingFaceH4/ultrachat_200k revision8049631c405ae6576f93f445c6b8166f76f5505a:
  train_sft only trains; test_sft is held out; gen shards excluded.
- HuggingFaceTB/smoltalk revision5feaf2fd3ffca7c237fc38d1861bc30365d48ffa:
  all/train (9shards) only, all/test held out. Component configs are not included
  a second time. The user's requested full SmolTalk is used, not smol-smoltalk.
- openai/gsm8k revision740312add88f781978c0658806c59bc2815b9866:
  main/train only, all1319 main/test questions held out; socratic excluded.

Official test first-user prompts are canonicalized and excluded from training.
Normalized full GSM8K test-question substrings in **any** training message are
also excluded with an Aho-Corasick matcher. This is exact/substring protection,
not a guarantee against paraphrases or contamination in upstream pretraining.
Remove duplicate complete normalized conversations across sources, processing
GSM8K first to retain requested original math examples. No source oversampling:
shuffle the retained natural window mixture; report actual counts and SmolTalk
component counts. Unsupported message schemas and overlaps are counted.

Render original chat template with deterministic empty-think-block removal in
both training and inference. Keep original nonempty reasoning/tool-call content.
GSM8K includes original rationale plus ####number, with the same step-by-step
prompt suffix in training/evaluation. Fast role/EOS span masking is checked
against native SFTDataset.generate_labels for128examples per selected split.

Tokenize complete conversations, mask user/system/tool/padding, then use length340
windows, stride256, overlap84. Mask already-used overlap targets. Keep and pad
**final short windows**. Assert each shifted assistant target and each labeled EOS
appears exactly once across windows. Drop only windows without active targets.
Long continuation windows can lack the complete original question. This limitation
is recorded, not described as full question/answer conditioning in every window.
Freeze int16 arrays shaped[N,2,340] containing input_ids and labels. Values fit
int16 (vocabulary6400, labels use-100). Verify file sizes and SHA256 of all copies.

## Training and eval

Full-parameter SFT,2epochs, BF16, batch64 x accumulation2, effective batch128,
AdamW LR5e-5,3%warmup capped1000updates then cosine to5e-6, weight decay0.01,
clip1,seed42. Same arrays/shuffle policy for Dense and MoE. Empty final batches
are not created; partial accumulation groups use their actual number of steps.
Do not overwrite original pretraining weights or user trainer/tokenizer edits.

Before and after SFT, generate all1319heldout GSM8K questions zero-shot greedy,
max256new tokens with total context340. Group equal-length prompts without
padding. Overlong prompts, invalid final formatting and generation limits stay
in the denominator. Report strict ####number and secondary last-number scores.
Also evaluate6arithmetic and8English prompts (greedy/sampling), and assistant-only
PPL on fixed512windows per source (selected from512official test conversations;
GSM8K includes all1319test conversations before window selection). No heldout
metric selects checkpoints. After tests reload the final published FP16 weights.

Stage FP32 model/optimizer resumes and FP16 inference weights locally; publish
fresh update-numbered files and verify SHA256. First update, every2000updates and
epoch end are checkpointed; retain2latest optimizer checkpoints plus epoch models.
Final checks require exact2epochs/update counts, finite tensors, strict reload,
and inference weights equal resume.model.half().

## Lifecycle and evidence

Shared Global Volume cmuxpmy6p000007l12uzr990b is mounted at/shared-volume.
Local receipts:`outputs/minimind/mixed-sft-20261024`.
W&B:`mind4u/MiniMind-Mixed-SFT`. A separate local user systemd supervisor streams
metrics to W&B using only local.env credentials. Pod offline streams are backups.
No user API keys are transferred to either Pod. Source bundle excludes.env.

The Dense Pod prepares data; both jobs wait for verified prepared.json, then train
independently. Supervisor exports/hash-checks results, verifies W&B finished on
the server, terminates only these task Pods and confirms fresh404reads. Preserve
Global Volume and all original models. On diagnosed failure, preserve diagnostics
and stop the affected task GPU; do not claim completion. A stopped disk still bills.

Run helper tests from repository root:
`python3 -m unittest discover -s minimind/scripts/runpod_mixed_sft -p test_prepare.py -v`.
Actual CUDA/backward, native-mask, first-checkpoint and online-log evidence must
come from execution. Script presence is not training success.

## MoE startup recovery (2026-10-25)

The initial MoE process encountered a transient JSONDecodeError while reading
prepare_status.json immediately before preparation completed. It had not loaded
data or taken any optimizer update. The supervisor preserved diagnostics and
stopped that task GPU. The same Pod was restarted after adding bounded retry
for partial/missing JSON publication reads. The verified frozen arrays and
original pretrain checkpoint are reused; Dense training continues unchanged.
IO-retry tests cover transient and persistent malformed JSON. Recovery details
and initial receipts are retained under the local MoE diagnostics directory.
