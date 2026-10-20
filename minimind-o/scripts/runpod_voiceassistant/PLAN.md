# VoiceAssistant-400K training plan

Goal: fine-tune native MiniMind-O on10,000 selected source records from10 distributed shards on a new dedicated Pod;
keep final artifacts on the user's existing shared-volume and verify local copies.

Design: fine-tune published dense sft_omni_768 weights, matching native tokenizer,
frozen SenseVoice and vision encoders. Convert SNAC (1:2:4 temporal hierarchy)
to 24kHz audio, then eight Mimi codebooks. Keep source question waveforms.
Process one pinned parquet shard at a time, deterministic single-epoch training;
first128 selected usable records held out. Persist converted shards, per-shard exclusions,
optimizer state, validation, code/dependency versions, final bundle and hashes.

1. Verify codec mapping tests (red/green) against original mini-omni utility.
2. Provision new Secure A40 with existing global shared volume; verify CUDA/mount.
3. Install isolated dependencies and pin dataset/codec/checkpoint revisions.
4. Convert first shard, check actual decode/encode, audio input and label masks;
   finite real-batch backward without optimizer update, baseline validation.
5. Full fine-tune: batch 4, accumulation 4, BF16, max sequence 1024,
   lr 2e-5 cosine to 2e-6, clip 1, deterministic seed42, log W&B offline and JSON,
   checkpoint first optimizer update, every250 updates, each shard and final.
6. Validate fixed128 after full epoch, finite checkpoint strict reload, decode
   generated audio samples, package weights/tokenizer/encoders/code/hash report.
7. Detached local watcher downloads finished bundle and verifies all SHA256.

Full dataset training remains incomplete until every shard is consumed, final
validation and packaging exit0, and the complete local bundle verifies.
