# RTX 5090 Agent RL run

Requested 2026-10-26. Continue the native `trainer/train_agent.py` algorithm from
`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` revision
`7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`, using all 30,196 rows of
`caspar/agent-rl-en` revision `57e3a151f11b0cdd2e150dc6fb2e508842b96ab0`.

## Resources and lifecycle

- Active Pod: `w4f4hf9r54sk2m`, one Secure Cloud RTX 5090 32GB in EUR-NO-1. Previous task Pod `8ko9ec7m6irjpv` is stopped.
- Catalog GPU rate at provisioning: $1.19/hour, plus storage.
- Active network volume: `o1j9cjv7va`, 50GB; mounted at `/workspace`. Previous task volume `mbuojaqjm8` is retained.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
- Remote root: `/workspace/minimind-agent`; only SSH is exposed externally.
- Local receipts: `outputs/minimind/agent-rl-20261026` from repository root.
- Supervisor: local user service `minimind-agent-rl-20261026.service`.
- Public output: `caspar/MiniMind-Dense-64M-Agent-RL-v1.0`.

`watch.py` streams training metrics to W&B from the local machine, exports and
hash-checks final artifacts, publishes to HF, verifies uploaded weights by
read-back, and deletes only this task's Pods; network volumes are preserved. It keeps checkpoints and
stops the task GPU after a persistent diagnosed failure; stopped storage still
bills. No Runpod, HF or W&B user API keys are copied to the Pod. The remote W&B
run is an offline backup; the local supervisor supplies the online run.

## Training

Full-parameter CISPO, one epoch, batch 4, four generations per prompt, 7,549
optimizer updates, AdamW LR 3e-7 with cosine decay to 3e-8, beta 0.1, BF16,
gradient clipping 1.0, seed 42. Backpropagation uses one response per microbatch,
normalized by the same global count of valid responses; full-batch versus
microbatch losses and gradients are tested for both CISPO and GRPO. Up to three tool turns, 256 generated tokens per
turn, final training trajectory cap 2,500 tokens. Long trajectories are left
truncated by the existing trainer. Prompt token lengths: median 525, p99 1,669,
maximum 2,403. All records train; no validation partition was invented.

SGLang 0.5.5 / Transformers 4.57.1 / PyTorch 2.8, Triton attention, with the
vLLM 0.11.0 optional rotary kernel for head dimension 96. Concurrent
HTTP rollouts preserve prompt/group order. Native MiniMind and Qwen3 FP32 logits
were identical in the local parity check. The SGLang server uses an equivalent
Qwen3 export; its native token log probabilities and weight reload are checked
on the GPU before training. Weights synchronize every 10 steps and checkpoints
save every 200 steps and at the end. The 16-row smoke run is separate and its
weights/optimizer are not carried into the full run.

Reward model: `Skywork/Skywork-Reward-V2-Qwen3-1.7B` (revision `e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc`). Raw classification scores are divided by 10 before the existing [-3, 3] clipping; this is a scale adjustment, not probability calibration. System messages are excluded from the reward-model conversation following its model card. Existing tool/format/ground-truth
reward heuristics are retained. The unit converter now handles temperature
offsets. Empty thinking blocks are removed to match the base SFT template;
thinking generation is disabled. Other source data/reward limitations remain.

## Verification

Run local integration checks from the repository root:

```bash
minimind/.venv/bin/python -m unittest discover \
  -s minimind/scripts/runpod_agent -p 'test_*.py' -v
```

`prepare.py` pins and hashes the source, strict-loads native weights, checks
Qwen3 parity and stages a smoke set. `run.py` checks real CUDA/SGLang/reward
execution, evaluates before training, executes the actual trainer, verifies the
complete epoch and finite changed weights, evaluates afterwards and reloads the
FP16 export strictly. `evaluate.py` uses the same 16 prompts before/after (13
tool/ground-truth cases and three qualitative prompts). This is a sample check,
not a general benchmark; numeric/string matches use the existing permissive
validator. See generated evaluation JSON for exact prompts, outputs and overlap.

Do not infer success from these scripts. Completion requires `finished.json`
locally, the HF publication receipt and fresh not-found reads for all task Pods. `live_status.json` and `watch.log` record ongoing work and failures.

## Reward correction after interrupted run

The original run was stopped at user request. For this dataset, all 10,208
tool rows have nonempty ground truth and all 19,988 general rows have empty
ground truth. The trainer uses tools plus nonempty ground truth to select the
execution-based reward branch; it does not require tool calls merely because
tools are exposed. Required-tool refusals receive -1, malformed or failed calls
-1.5, and successful execution plus matching final answer up to +3. Generic
Skywork/length rewards cannot offset failure on those tasks. Numeric ground
truth matching requires a complete number rather than a substring.

Final publication now requires sample ground-truth accuracy above the baseline
and at least one expected tool call. This is a minimum gate, not a claim of broad
model quality. `verify_reward_fix.py` is a separate local same-template math
SFT experiment on held-out operands; it neither resumes the cloud job nor
validates full Agent RL.
