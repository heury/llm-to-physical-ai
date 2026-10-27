# Hermes multi-turn SFT followed by Agent RL

User-authorized RTX 5090 run, 2026-10-26. Uses only `func-calling.json` from
NousResearch/hermes-function-calling-v1 revision dae3e1d28cfbcf4b915c04ea1e072030529b4bda
for SFT. Original base is caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0 at
7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d. No previous failed RL or toy math warmup
weights are reused.

Conversion validates JSON calls and normalizes tool responses, renders tool schemas
once, supervises each assistant call/answer including EOS, and masks the preceding
context. Conversations are grouped by user text for deterministic ~10% validation.
Invalid rows and >4096-token targets are excluded with counts; targets are not truncated.
Local whole-file check: 1893 input rows, 793 invalid rows, 1786 train targets,
203 validation targets, 38 overlength targets skipped, 899 retained post-tool targets.
These counts must match the GPU manifest before accepting the run.

SFT: 3 epochs, response batch 1, accumulation 8, 2e-5 cosine to 2e-6, BF16 compute,
FP32 trainable weights, AdamW weight decay .01, gradient clip 1, seed42. Choose the
best of three epochs by token-weighted validation CE. Generate up to 16 call and
16 post-tool validation examples before and after; completion is not factual correctness.

The local supervisor publishes SFT first, verifies its public uploaded weight bytes,
and sends a non-secret acknowledgement. Only then does pipeline.py invoke the actual
trainer/train_agent.py on all 30196 caspar/agent-rl-en rows, pinned at
57e3a151f11b0cdd2e150dc6fb2e508842b96ab0. RL settings: 1 epoch, 7549 updates,
batch4 x4 generations, LR3e-7 cosine, beta.1, microbatch1, three turns of max256 tokens,
max retained context2500, policy sync10, save200. SGLang0.5.5 Triton rollout; reward
Skywork/Skywork-Reward-V2-Qwen3-1.7B e51ea3e08fb81326c3b812a7ff0cb9cee83e59cc.
General reward score scale10; labeled tool task execution rewards bypass generic score.
Unfinished valid-call trajectories now receive -2 and identical successful repeats
receive -0.25 each, bounded below by -3. These penalties are an experimental choice.

Both models are published as experimental with actual sample results; lack of improved
accuracy is not mislabeled as success. Technical integrity failures prevent publication.
Model IDs: caspar/MiniMind-Dense-64M-Hermes-FunctionCalling-SFT-v1.0 and
caspar/MiniMind-Dense-64M-Hermes-Agent-RL-v1.0. Credentials remain local.

Remote root: /workspace/minimind-hermes-agent. Local receipts:
outputs/minimind/hermes-agent-20261026 (repository root).
Supervisor: minimind-hermes-agent-20261026.service. After final upload and byte
verification, remove only task Pod IDs recorded in this run's ledger. Preserve volumes.
On persistent failure stop the task Pod and retain checkpoints. Actual completion requires
finished.json plus publication and deletion receipts; source files do not prove completion.

Checks (run in minimind/):

```bash
.venv/bin/python -m unittest trainer.test_agent_rewards scripts.runpod_agent.test_integration scripts.runpod_hermes_agent.test_data -q
```
