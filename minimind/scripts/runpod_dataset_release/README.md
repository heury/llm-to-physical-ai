# Exact completed training dataset release

Exports only the completed 2026-10-24 full-document pretraining and mixed SFT,
and 2026-10-25 short LLaVA vision training experiments. Original volume files are
read-only inputs. It does not launch training or modify model repositories.

- `export.py`: run on the temporary CPU Pod with the existing global volume;
  exports source records, original train/evaluation arrays, indexes and selected
  original image bytes; validates existing source/array hashes and row counts.
- `cards.py`: copies pinned source cards, tokenizer and original preparation code;
  describes upstream licenses and exact split/window semantics.
- `publish.py`: run locally with the existing Hugging Face login; transfers and
  hashes all files, uploads three public datasets, checks anonymous Hub LFS hashes
  and downloads non-LFS files plus a representative Parquet shard; tags `v1.0`.
  The Hugging Face credential is never transferred to the Pod.

The release uses explicit experiment paths and the one task-owned Pod connection.
Do not run it against an unrelated experiment without updating and reviewing those
paths. Source cards must first be downloaded at `cards.SOURCES` revisions into
`outputs/dataset-release-v1.0/source-cards/`.

Run local publication from the repository root:

```bash
/tmp/minimind-v-exercise-venv/bin/python minimind/scripts/runpod_dataset_release/publish.py
```

Publication receipts are in `outputs/dataset-release-v1.0/*-published.json`.
`status.json` describes current progress; a dataset is complete only when its
published receipt exists. Delete only the temporary dataset-export Pod after all
three public releases verify; preserve the global volume and pre-existing Pods.

Validation: Python compile check, byte hashes against original manifests, exact
exported conversation counts, remote/local file hash comparison and anonymous
public-repository checks. The standalone recipes preserve historical paths and
are reference material, not a turnkey training launcher.

Attribution supplements used in this release:

- Run `vision_attribution.py` before the vision export finishes so the image URL
  and individual COCO license records enter its final hash manifest.
- Run `fineweb_attribution.py` after the pretrain export; it verifies each source
  Parquet hash and matches every text row to the historical JSONL, then appends
  attribution shards to the export manifest.
- `prefetch.py` may overlap later downloads with uploads. `transfer.py` uses
  dataset-level locks and verifies all file hashes after parallel SSH range reads.

These scripts contain no credentials. The original Runpod and Hugging Face
credentials remain in the local environment; publication sends only artifact data.

`validate.py` loads every public dataset configuration used for training, checks
sample token windows against the bundled tokenizer/template, and checks original
image bytes plus resized training pixel hashes across vision splits.
`finalize.py` waits for all three verified publication receipts, runs that check,
and deletes only the recorded temporary CPU Pod. Successful cleanup is recorded
in `cleanup.json`; no cleanup occurs when validation fails.

Vision v1.0.1 is a metadata-only follow-up from `patch_decoder.py`: original JPEG
bytes match, but a sampled image decodes to different pixels with Pillow 13.0.0 /
libjpeg-turbo 3.2.0. Pillow 11.3.0 / libjpeg-turbo 3.1.1 reproduces all five sampled
training pixel hashes. The original training package version was not recorded;
this is a validated reproduction environment, not an inferred training version.
Install Pillow 11.3.0 into `/tmp/minimind-pillow-11` for the finalizer's isolated
image validation. Original v1.0 remains tagged; no dataset rows or images change.
