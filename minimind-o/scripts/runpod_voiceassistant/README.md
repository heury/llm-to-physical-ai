# VoiceAssistant-400K → MiniMind-O

Dedicated Secure Cloud Pod `jwdo0nr8pb9ikf`, A40 48GB ($0.59/hour).
Existing Pods and all other jobs are preserved.

- Dataset: `gpt-omni/VoiceAssistant-400K`, revision
  `65dab707fcdd3d43dcfb834398aa9fed4116be3a`, 470,054 original records / 325 source shards. Use only **10,000 selected records**: seeded1,000 records per shard from IDs0,36,72,108,144,180,216,252,288,324. Hold out128, train up to9,872 valid examples. This is a distributed subset, not a uniform sample of the entire dataset. The selected source indices are saved in `selection/`.
- Base: published native dense `jingyaogong/minimind-3o-pytorch/sft_omni_768.pth`,
  pinned `9e22fb0e51852359ff51bd48d91ec3a345dbd75b`; this is fine-tuning,
  not training the 115M backbone from random initialization.
- Question input: original audio through frozen SenseVoice; original question
  text is retained in converted records. Answer text and audio are both targets.
- Answer codec conversion: original SNAC24k (1:2:4 temporal hierarchy) → waveform
  → Mimi24k eight2048-entry codebooks. SNAC integers are never used as Mimi labels.
- One epoch, BF16, sequence1024, batch4, accumulation4, lr2e-5 cosine→2e-6,
  gradient clip1. Frozen vision projector/encoders. Native model unchanged.
- Shards are downloaded one at a time, converted in256-source-row chunks, then
  trained. Source corruptions and examples losing answer labels after sequence
  truncation are explicitly recorded. Answer audio is capped at80seconds before
  encoding; model context can truncate earlier. Fixed first128 usable examples
  are held out; this initial-shard validation is not a representative benchmark.
- Real backward smoke without update, first optimizer-update checkpoint audit,
  periodic250-update and per-chunk checkpoint, final strict reload/finite checks.
- W&B SDK uses **offline** mode, plus JSON progress on shared-volume. No API key
  is transferred. Offline W&B logs remain on shared-volume; summaries and JSON
  progress are included in the final bundle.

Remote root: `/shared-volume/minimind-o-voiceassistant-20261020`.
Local final destination: `outputs/minimind-o/voiceassistant-20261020`.

The independent local bootstrap `/tmp/omni_bootstrap.py` waits for the container,
uploads a credential-free source archive, installs a venv and starts the remote
job. `/tmp/omni-bootstrap-status.json` records installation/launch status.
After launch, `/tmp/omni_watch.py` polls status and downloads `artifacts.tar.gz`
only after exit0, then verifies each file against `SHA256.json`. Only after local verification, it stops this task’s dedicated Pod to end GPU billing. Existing Pods are preserved. Watcher status:
`/tmp/minimind-o-voiceassistant-watch-status.json`.

Final bundle contains `sft_omni_768.pth`, tokenizer, frozen SenseVoice/Mimi/vision
models, source, resolved dependencies, validation, generated WAVs and hashes.
The shared volume additionally retains converted data and optimizer resume state.
Partial-run resume is intentionally not automatic: investigate progress before
restarting, because automatic restart could repeat updates or overwrite evidence.

Do not call the job complete until all10 selected source shards and10,000 selected source records are processed, final
validation/package succeeds, and the downloaded local hashes all verify.
