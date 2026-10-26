# Full-document FineWeb-Edu pretraining

Requested2026-10-24. Train fresh Dense and MoE (768hidden,8layers; MoE4experts,
top1) for2epochs on independent RTX5090 Secure Cloud Pods. Catalog read quoted
$1.19/hour each. Original6400-token tokenizer is preserved. Existing pretrained,
UltraChat and GSM8K artifacts are not overwritten.

## Data policy

Source: `/shared-volume/fineweb-edu-10BT-score-3.5/pretrain-score-3.5.jsonl`.
The existing dataset has `score >= 3.5`, revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`,1288228documents,6525738157bytes.
Verify its complete conversion SHA256 and tokenizer hashes before processing.

Tokenize each **complete document**, without truncation. Form BOS+text+EOS and
emit every complete340-token window with stride339 (one context token overlaps).
The shifted next-token targets are unique across adjacent windows. Drop the
remaining short window **per document**, including all of any document shorter
than340tokens; no padding and no cross-document packing. EOS is only the real
document end, never an artificial chunk boundary. Counts distinguish all source
tokens, retained unique tokens, dropped tokens and training targets.

Split by SHA256(text), first8bytes modulo4096==0 for validation and otherwise
training, before chunking. Identical document texts cannot cross the split.
Uniformly shuffle all training chunks every epoch. This gives equal weight per
retained token/transition, not equal weight per original document. Each epoch
uses every complete training chunk exactly once. Record document/window offsets
in `data/document-index.jsonl`. Publish uint16 arrays only after source/copy hash
checks; manifest `prepared.json` is the readiness gate for both jobs.

## Training and verification

Batch64 x gradient accumulation4, nominal effective batch256; BF16; AdamW
LR3e-4,2%warmup capped250updates then cosine to3e-5; weight decay0.01; clip1;
seed42. Native model forward and next-token loss are retained. Basic
PretrainDataset and the user's tokenizer code are unchanged. The cloud job uses
its own prepared full-document dataset instead of the prefix-only dataset.

`smoke.py` tests actual full-size Dense/MoE BF16 forward/backward/AdamW with
batch64,length340 and MoE router gradients, then discards smoke weights.
`test_chunks.py` verifies short-tail dropping and exact transition coverage.
Run from repository root:
`python3 -m unittest discover -s minimind/scripts/runpod_full_pretrain -p test_chunks.py -v`.

Save first update, every2000updates, and each epoch end. Stage checkpoints on
local Pod disk; publish under fresh update-numbered names; verify published
SHA256; retain the latest2optimizer checkpoints plus both epoch inference models.
Resumable files contain FP32 model/optimizer state and exact epoch/step/update.
Automatic continuation after failure is not claimed: the supervisor preserves
published checkpoints and stops the task GPU for diagnosis.

Final checks require expected2epochs/update counts, finite weights and optimizer,
strict model reload, inference weight equality to resume.model converted toFP16,
and shared-volume publication hashes. Evaluate1024fixed seeded heldout chunks
(or all if fewer) before and after training; after evaluation uses reloaded saved
inference weights. Also preserve4English raw continuation probes. These are
pretrain probes, not chat/GSM8K benchmark results. PPL is not directly comparable
with the earlier prefix-only validation set.

## Durable outputs, W&B and lifecycle

Global Volume `cmuxpmy6p000007l12uzr990b`, name`shared-volume`, mount`/shared-volume`.
Root:`/shared-volume/minimind-full-pretrain-20261024`.
Final models:`{dense,moe}/out/pretrain_768[_moe].pth`; tokenizer:`tokenizer/`.
Local receipts:`outputs/minimind/full-pretrain-20261024`.

Start Dense for preparation and training. The local supervisor creates MoE only
after preparation completes, rereading stock/official image and refusing a rate
above the quoted$1.19/hour. CUDA preflight precedes remote launch.

Supervisor user service:`minimind-full-pretrain-20261024.service`.
It runs independently of the chat, restarts on process failure, reads credentials
from the local `.env`, streams metrics into W&B online, and keeps the original
Pod offline W&B records as backup. No user API credentials are copied to Pods.
W&B project:`mind4u/MiniMind-FineWeb-Full-Pretrain`.

After each job, export/hash-check results, verify W&B server finished/complete,
terminate only that task Pod, and verify a fresh404read. The Global Volume stays.
On detected training/preparation failure, preserve diagnostics and stop that task
Pod to release GPU; a stopped Pod disk still bills. The service does not silently
restart from scratch. Completion/cleanup JSON and live observations establish
actual status; the existence of these scripts does not establish success.
