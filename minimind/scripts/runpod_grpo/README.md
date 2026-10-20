# UltraFeedback GRPO on the existing A40

The policy and frozen reference start from the completed UltraFeedback DPO
checkpoint. All 61,135 `train_prefs` rows are scanned at the pinned dataset
revision. The 56,910 prompts fitting the 768-token prompt budget are retained;
4,225 longer prompts are excluded. Duplicate prompt-ID rows within training
are preserved (five occurrences). Test IDs cannot overlap training IDs. There
is no random training subset. A fixed hash-selected 128-prompt heldout set is
used for baseline and final evaluation.

The separately frozen `Skywork/Skywork-Reward-V2-Qwen3-0.6B` scorer uses its own
tokenizer and chat template. Its revision is pinned and files are persisted on
the volume. Following its model card, system messages are omitted from reward
scoring. Chosen/rejected scores in the dataset are not reused for new responses.
128 heldout preference pairs check the scorer's label agreement before training.

Training uses MiniMind's GRPO trainer and Torch rollout engine, with run-specific
copies here. The run explicitly selects `loss_type=grpo` rather than the native
CISPO default. Each question produces four new completions; group-normalized
advantages drive the clipped surrogate plus a frozen-reference KL penalty.
The reward is raw scorer output clipped to [-20,20], divided by five, minus a
repeated-trigram penalty capped at 0.5. Empty responses receive -5. There are no
length or think-tag bonuses. Thinking prompts are disabled.

Rollout changes: float32 selected-token log probabilities, EOS-inclusive masks
that exclude subsequent tokens, identical old/current attention masks, and
unwarped sampling (temperature=1, top_p=1, top_k=0). Only completion logits are
materialized during training. Reference and policy use the same BF16 autocast.
The four rollout contract tests were demonstrated failing before fixes and
passing afterward. The real-batch smoke test also requires the initial policy
ratio to be near one, reference KL near zero, nonzero finite gradients, reward
variation, and an actual optimizer update on a disposable copy of the policy.

The smoke batch uses the eight longest retained prompts, four completions each,
and a 256-token completion budget to measure memory conservatively. Production
training starts fresh from DPO; smoke weights never overwrite the source or
production checkpoint. The final settings and measured estimate are written to
`training_settings.json` and `smoke.json`.

The detached launcher runs baseline evaluation, one full training epoch, final
strict/finite weight and optimizer checks, heldout generation/scoring, and
packaging. W&B logs reward, raw reward, repetition, KL, gradients, group variance,
response length, policy ratios, progress and ETA. Checkpoints are saved every
100 steps, at the first step, and at the final step. The final evaluation uses the training scorer,
so its scores are not an independent accuracy benchmark; manual generation
samples are included for review.

Remote root: `/shared-volume/minimind-grpo-ultrafeedback-20261020`.
Local root: `out/ultrafeedback-grpo-20261020`.
Final weight: `out/grpo_768.pth` remotely, `grpo_768.pth` locally.
The local watcher must run with authorized network access; it fails visibly on
SSH errors and verifies every final bundle file's SHA256 after download. The
existing user-owned Pod is preserved. Optimizer state, converted data and scorer
weights remain on the volume; the local final bundle contains policy, tokenizer,
configuration, manifests, environment versions and evaluation reports.
