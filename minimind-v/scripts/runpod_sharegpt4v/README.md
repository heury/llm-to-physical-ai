# ShareGPT4V on a dedicated Runpod A40

The job uses all 102,025 original `Lin-Chen/ShareGPT4V` 100k-caption records,
pinned at `55d02b0bc53a2754095a14110dda6daedd95671d`. Images are obtained
from pinned `wooj1nBot/sharegpt4v_flat_dataset` plus four pinned web-image
archives in `DavidNguyen/ShareGPT4V-Sam`. Exact original image filenames map
image bytes to the source records. Captions from mirrors are never used. Missing images abort the job instead of silently
selecting a subset. Images are resized to SigLIP's 256×256 input geometry and
stored losslessly in the job's ephemeral image cache. The manifest reports truncation and excluded short targets.

Base: the user's completed UltraChat MiniMind 768/8 SFT model and its matching
6400-token tokenizer. Vision: frozen `google/siglip2-base-patch32-256` at
`94dffa8cb1179de3e03f091dbc3917e5d5a9ae84`. Phase 1 trains the randomly
initialized vision projector for one epoch at 4e-4. Phase 2 trains projector
and first/last language layers for one epoch at 5e-6. Both use BF16, sequence
768, batch 16, accumulation 2, cosine decay to 10%, seed 42. 256 fixed unique image
groups are held out in both stages. Only assistant answer tokens are targets.

The native model and upstream trainers are unchanged. This isolated runner uses
the real `wandb` SDK, not the upstream `swanlab as wandb` alias. Smoke tests
check finite backward gradients, memory and frozen parameters without updating
weights. Each stage saves its first step, every 250 steps and last step. Resume
state is recorded for investigation; automated mid-epoch resume is not provided.
A launch after partial training must be investigated before restarting.

Artifacts: `/shared-volume/minimind-v-sharegpt4v-20261020`. Final weights,
vision encoder, tokenizer, code, resolved dependencies, validation reports,
four greedy generations and SHA256 manifest are packaged as `artifacts.tar.gz`.
The local watcher downloads only after exit code 0 and verifies every hash.
Local destination: `outputs/minimind-v/sharegpt4v-20261020`.

The dedicated task Pod is `5yhriom3b1z0rs`. Existing Pod `ncpk3lqf5m7421`
and its jobs must never be stopped or changed by this task.

User-authorized cleanup: after both training stages and final validation succeed,
the local watcher downloads the complete bundle and verifies every SHA256.
Only then it terminates task-created Pod `5yhriom3b1z0rs` and confirms its
absence through a fresh inventory read. Shared-volume artifacts and existing
Pod `ncpk3lqf5m7421` are preserved. A durable local receipt permits cleanup
retries even after the task Pod is gone. The local watcher must remain running.
