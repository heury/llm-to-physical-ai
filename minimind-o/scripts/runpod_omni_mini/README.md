# RTX 5090 Omni mini + LLaVA fine-tuning

This run starts from the published native dense Omni checkpoint, not a random
Talker or a text-only backbone. Exact revisions are in `prepare.py`.

- Dedicated Secure Cloud Pod: `zuq0hsnyvgsmjz`, one RTX 5090 32GB, GPU $1.19/hour.
- Remote directory: `/workspace/omni-mini-20261025` (80GB Pod volume).
- Local artifacts: `outputs/omni-mini-20261025`, excluded from Git.
- Destination: private `caspar/MiniMind-O-Mini-LLaVA-5090-20261025`.
- Stages: one epoch T2A full voice model (2e-5), A2A audio projector (1e-4),
  A2A full voice model (2e-5), visual captions vision projector (5e-5),
  visual QA vision projector (5e-5). Voice effective batch 32, visual 32.
- Audio: 515,415 T2A rows and 76,797 A2A rows, each minus 128 validation rows.
  Voice sequence 1024; audio-only user prompts for A2A. Native codec delays,
  reference/speaker conditions and 10x audio stop-token loss are retained.
- Vision: all 49,744 caption training rows and 149,227 visual QA training rows;
  original 256/690 validation and 690 test rows are preserved. Images are read
  from the original ZIPs and EXIF-corrected. Sequence length 384.
- Frozen SenseVoice, SigLIP2 and Mimi. Vision stages update only `vision_proj`.
- No augmentation or scheduled sampling. Fixed per-row native multi-turn and
  reference selection, seed 42. Tail positions with no labels are cropped.
- Measure validation before/after stages; save stage checkpoints, periodic
  latest weights, loss logs, exact dependencies, and source. Native checkpoint
  loading must be strict. Generation produces real WAV and image/audio inputs.
- `watch.py` runs on the host with existing credentials, never transfers those
  credentials to the Pod. It verifies downloaded SHA256s, uploads to the Hub,
  checks remote sizes and LFS hashes, then deletes only this run's Pod.
- If the 12-hour deadline is reached before success, stop the Pod and preserve
  its disk for investigation. This is not reported as completed training.

`setup.sh` installs dependencies into the template's Python and downloads inputs.
`run.sh` runs training and sample generation. Commands execute from the remote
project root; `watch.py` executes from the local minimind-o directory.

A stage with both its checkpoint and completed summary can be reloaded on a
restart. A partial stage restarts from the previous completed stage; optimizer
mid-stage resume is not implemented. Never start two training processes together.

Completion requires `SUCCESS.json`, verified local and Hub artifacts, and a
confirmed absent Pod. Source or script existence is not evidence of success.

## Completed result

Completed on 2026-10-25. All five stages processed their full training split with
zero rows skipped for missing labels; pure training time was 74.08 minutes.
Final checkpoint SHA256:
`60309d7d328bd70cf66db855f7c2559c875c7634eb25386a0af5b4f3ca9b4e4d`.

Private model: https://huggingface.co/caspar/MiniMind-O-Mini-LLaVA-5090-20261025

Verified Hub commit: `b7b050060232c91df87ceff4aa8a507cc8021191`.
All 68 uploaded artifact files (including the checksum manifest) were checked
against their remote Git/LFS object hashes. Pod deletion was confirmed by the
completion watcher after successful upload verification.

[Full Korean report](../../outputs/omni-mini-20261025/artifacts/REPORT_ko.md)
and [all seven generated samples](../../outputs/omni-mini-20261025/artifacts/samples/results.json)
are retained locally. All seven samples produced non-silent WAVs and finished
before the token limit. They include semantic failures: the model used
“expensive” as a synonym of “exuberant,” answered “dogs and cats” for animal-derived
products, and misclassified street parking as a parking lot. Lower validation
loss does not establish reliable answer accuracy. The report preserves failures.
