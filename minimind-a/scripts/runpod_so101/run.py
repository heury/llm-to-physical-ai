"""Pinned SO101 LeRobot-v3 data -> frozen native VLM -> train action expert.

Run on a dedicated CUDA Pod. All durable outputs stay on shared-volume.
"""

import hashlib
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import av
import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from model.model_vla import MiniMindVLA, VLAConfig
from dataset.vla_dataset import VLAProcessor, collate_batch, pad_tokens
from scripts.runpod_so101.data_utils import (
    make_chunks,
    normalize,
    normalization,
    split_episodes,
)

ROOT = Path(
    os.environ.get("SO101_RUN_ROOT", "/shared-volume/minimind-a-so101-20261020")
)
BASE = Path("/shared-volume/minimind-v-sharegpt4v-20261020")
DATASET = os.environ.get("SO101_DATASET_ID", "caspar/so101_pickplace_sim")
REVISION = os.environ.get(
    "SO101_DATASET_REVISION", "9b1c5189e3858e9254fc5e48259a035d22da497a"
)
CAMERA = "observation.images.top"
EPOCHS, BATCH, HORIZON = 100, 128, 16


def save(name, obj):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def phase(name):
    save("phase.json", dict(phase=name, time=time.time()))
    print("PHASE", name, flush=True)


def frozen_digest(model):
    digest = hashlib.sha256()
    for key, value in model.vlm.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.cuda.get_device_properties(0)
    assert os.environ.get("SO101_EXPECT_GPU", "B300") in device.name
    assert device.total_memory > 30 * 1024**3
    torch.cuda.set_per_process_memory_fraction(0.94)
    phase("read_original_dataset")
    data_dir = ROOT / "dataset"
    info = json.loads((data_dir / "meta/info.json").read_text())
    table = pq.read_table(data_dir / "data/chunk-000/file-000.parquet").to_pydict()
    episodes = pq.read_table(
        data_dir / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    task_rows = pq.read_table(data_dir / "meta/tasks.parquet").to_pylist()
    tasks = {}
    for row in task_rows:
        text = row.get("task", row.get("__index_level_0__"))
        assert isinstance(text, str) and text.strip(), row
        tasks[int(row["task_index"])] = text
    state = np.asarray(table["observation.state"], dtype=np.float32)
    actions = np.asarray(table["action"], dtype=np.float32)
    episode_ids = np.asarray(table["episode_index"], dtype=np.int64)
    indices = np.asarray(table["index"], dtype=np.int64)
    assert np.array_equal(indices, np.arange(len(indices)))
    total_frames = int(info["total_frames"])
    total_episodes = int(info["total_episodes"])
    assert info["codebase_version"] == "v3.0" and info["fps"] == 20
    assert len(indices) == total_frames
    assert len(episodes) == total_episodes
    assert state.shape == actions.shape == (total_frames, 6)
    assert np.isfinite(state).all() and np.isfinite(actions).all()
    train_episodes, val_episodes = split_episodes(episode_ids, 5, 42)
    train_ids = np.flatnonzero(np.isin(episode_ids, train_episodes))
    val_ids = np.flatnonzero(np.isin(episode_ids, val_episodes))
    stats = dict(
        state=normalization(state, train_ids),
        action=normalization(actions, train_ids),
        method="train-frame population mean/std; floor std=1e-6",
        action_names=info["features"]["action"]["names"],
        action_semantics="Original absolute joint target values; no delta conversion",
        fps=info["fps"],
    )
    save("normalization.json", stats)
    normalized_state = normalize(state, stats["state"])
    target, action_mask = make_chunks(
        normalize(actions, stats["action"]), episode_ids, HORIZON
    )
    # Metadata timestamps align concatenated video frames to independent episode rows.
    video_indices = np.full(len(indices), -1, dtype=np.int64)
    for episode in episodes:
        ep = int(episode["episode_index"])
        rows = np.flatnonzero(episode_ids == ep)
        assert len(rows) == int(episode["length"])
        assert rows[0] == int(episode["dataset_from_index"])
        assert rows[-1] + 1 == int(episode["dataset_to_index"])
        start = float(episode[f"videos/{CAMERA}/from_timestamp"])
        times = start + np.asarray(table["timestamp"], dtype=np.float64)[rows]
        video_indices[rows] = np.rint(times * info["fps"]).astype(np.int64)
    assert len(np.unique(video_indices)) == len(indices) and video_indices.min() == 0
    manifest = dict(
        dataset=DATASET,
        revision=REVISION,
        total_episodes=total_episodes,
        total_frames=total_frames,
        train_episodes=train_episodes,
        validation_episodes=val_episodes,
        train_frames=len(train_ids),
        validation_frames=len(val_ids),
        split_seed=42,
        train_validation_episode_disjoint=True,
        camera=CAMERA,
        omitted_camera="observation.images.wrist (single-camera teaching model)",
        horizon=HORIZON,
        state_dim=6,
        action_dim=6,
        tasks=tasks,
        action_padding="Masked after each episode ends; never cross episodes",
        state_min=state.min(0).tolist(),
        state_max=state.max(0).tolist(),
        action_min=actions.min(0).tolist(),
        action_max=actions.max(0).tolist(),
        source_files={
            str(p.relative_to(data_dir)): sha(p)
            for p in data_dir.rglob("*")
            if p.is_file() and ".cache" not in p.parts
        },
    )
    save("dataset_manifest.json", manifest)
    phase("load_native_pretrained_vlm")
    cfg = VLAConfig(
        vlm_config=dict(
            hidden_size=768,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=6400,
            image_hidden_size=768,
            image_token_len=64,
            image_ids=[12],
            max_position_embeddings=512,
        ),
        vision_model_path=str(BASE / "vision_encoder"),
        state_dim=6,
        action_dim=6,
        chunk_size=HORIZON,
        expert_dim=256,
        expert_heads=4,
        expert_layers=2,
        freeze_vlm=True,
    )
    model = MiniMindVLA(cfg)
    checkpoint = BASE / "out/sft_vlm_768.pth"
    assert (
        sha(checkpoint)
        == "4e40a7e96bd1e846f9f2c0a5ed29eecf6e7701480f0eb4a33f24aa603590df58"
    )
    model.load_vlm_weights(checkpoint)
    frozen_before = frozen_digest(model)
    processor = VLAProcessor(cfg, model.vlm.processor)
    task_tokens = {
        task_id: processor.encode_instruction(text) for task_id, text in tasks.items()
    }
    ids = [task_tokens[int(task_id)] for task_id in table["task_index"]]
    input_ids, language_mask = pad_tokens(ids)
    length = input_ids.size(1)
    model.cuda().eval()
    phase("cache_frozen_vlm_contexts")
    cache_file = ROOT / f"context_cache_layers{len(model.vlm.model.layers)}.pt"
    if cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu", weights_only=True)
        assert (
            cached["revision"] == REVISION and cached["frozen_sha256"] == frozen_before
        )
        hidden = cached["hidden"]
        assert hidden.shape == (total_frames, length, 768)
    else:
        hidden = torch.empty(len(ids), length, 768, dtype=torch.bfloat16)
        wanted = {int(v): int(i) for i, v in enumerate(video_indices)}
        frames, frame_rows = [], []
        seen = set()
        started = time.time()

        def encode():
            pixels = model.vlm.processor(images=frames, return_tensors="pt")[
                "pixel_values"
            ].cuda()
            rows = torch.tensor(frame_rows, dtype=torch.long)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.vlm(
                    input_ids[rows].cuda(), language_mask[rows].cuda(), pixels
                )
            assert torch.isfinite(features).all()
            hidden[rows] = features.cpu().to(torch.bfloat16)
            seen.update(frame_rows)
            if len(seen) % 512 == 0 or len(seen) == total_frames:
                save(
                    "progress.json",
                    dict(
                        phase="context_cache",
                        frames=len(seen),
                        total_frames=total_frames,
                        elapsed_seconds=time.time() - started,
                        time=time.time(),
                    ),
                )
                print("CACHED", len(seen), "/", total_frames, flush=True)
            frames.clear()
            frame_rows.clear()

        video = data_dir / f"videos/{CAMERA}/chunk-000/file-000.mp4"
        with av.open(str(video)) as container:
            for frame in container.decode(video=0):
                frame_index = int(round(float(frame.time) * info["fps"]))
                row = wanted.get(frame_index)
                if row is None:
                    continue
                assert row not in seen and row not in frame_rows
                if row in [int(train_ids[0]), int(val_ids[0])]:
                    (ROOT / "examples").mkdir(exist_ok=True)
                    frame.to_image().save(ROOT / f"examples/frame_{row}.jpg")
                frames.append(frame.to_image())
                frame_rows.append(row)
                if len(frames) == 32:
                    encode()
            if frames:
                encode()
        assert len(seen) == total_frames, (
            len(seen),
            sorted(set(range(total_frames)) - seen)[:20],
        )
        torch.save(
            dict(revision=REVISION, frozen_sha256=frozen_before, hidden=hidden),
            cache_file,
        )
    # Cached computation is valid because the complete VLM is frozen and stays in eval.
    hidden = hidden.cuda()
    valid = language_mask.cuda()
    states = torch.from_numpy(normalized_state).cuda()
    targets = torch.from_numpy(target).cuda()
    masks = torch.from_numpy(action_mask).cuda()
    train_index = torch.from_numpy(train_ids).cuda()
    val_index = torch.from_numpy(val_ids).cuda()
    phase("training_setup")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_ids) / BATCH)
    settings = dict(
        epochs=EPOCHS,
        batch_size=BATCH,
        steps_per_epoch=steps_per_epoch,
        optimizer_updates_target=EPOCHS * steps_per_epoch,
        learning_rate=1e-4,
        schedule="100 step linear warmup; cosine to 1e-5",
        dtype="bfloat16 autocast; FP32 weights",
        freeze_vlm=True,
        frozen_sha256=frozen_before,
        vlm_checkpoint=str(checkpoint),
        vlm_checkpoint_sha256=sha(checkpoint),
        context_cache=True,
        seed=42,
        trainable_parameters=sum(p.numel() for p in trainable),
        parameters=sum(p.numel() for p in model.parameters()),
        gpu=device.name,
        gpu_memory_bytes=device.total_memory,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        camera=CAMERA,
        horizon=HORIZON,
    )
    save("training_settings.json", settings)

    def context(rows):
        return model.project_context(hidden[rows], valid[rows], states[rows])

    def flow_loss(rows, noise=None, time_value=None):
        memory, visibility = context(rows)
        return model.flow_loss(
            memory, visibility, targets[rows], masks[rows], noise, time_value
        )

    def evaluate(action_metrics=False, prediction_file="validation_predictions.npz"):
        model.eval()
        total = 0.0
        count = 0
        gen = torch.Generator(device="cuda").manual_seed(4321)
        pred_chunks = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for rows in val_index.split(128):
                noise = torch.randn(
                    (len(rows), HORIZON, 6), device="cuda", generator=gen
                )
                t = torch.rand(len(rows), device="cuda", generator=gen)
                weight = int(masks[rows].sum()) * 6
                total += float(flow_loss(rows, noise, t)) * weight
                count += weight
                if action_metrics:
                    memory, visibility = context(rows)
                    x = model.integrate_actions(memory, visibility, noise, steps=10)
                    pred_chunks.append(x.float().cpu())
        report = dict(validation_flow_mse=total / count, validation_frames=len(val_ids))
        if action_metrics:
            prediction = torch.cat(pred_chunks).numpy()
            truth = target[val_ids]
            mask = action_mask[val_ids]

            def metric(p, t):
                err = (p - t) ** 2
                return float(err[mask].mean())

            raw_pred = prediction * np.asarray(stats["action"]["std"]) + np.asarray(
                stats["action"]["mean"]
            )
            raw_truth = truth * np.asarray(stats["action"]["std"]) + np.asarray(
                stats["action"]["mean"]
            )
            hold = np.broadcast_to(state[val_ids, None, :], raw_truth.shape)
            mean = np.broadcast_to(np.asarray(stats["action"]["mean"]), raw_truth.shape)
            report.update(
                normalized_action_mse=metric(prediction, truth),
                raw_action_mse=metric(raw_pred, raw_truth),
                hold_current_state_raw_mse=metric(hold, raw_truth),
                train_mean_raw_mse=metric(mean, raw_truth),
                first_step_raw_mse=float(
                    ((raw_pred[:, 0] - raw_truth[:, 0]) ** 2).mean()
                ),
                per_joint_raw_mse=((raw_pred - raw_truth) ** 2)[mask].mean(0).tolist(),
                euler_steps=10,
                noise_seed=4321,
                scope="Offline held-out episode error; no simulation rollout or success-rate estimate",
            )
            np.savez(
                ROOT / prediction_file,
                row_indices=val_ids,
                predicted_actions=raw_pred,
                target_actions=raw_truth,
                valid_action_mask=mask,
            )
        model.train()
        return report

    phase("baseline_evaluation")
    save("evaluation_before.json", evaluate(True, "validation_predictions_before.npz"))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Real BF16 backward smoke, but do not consume an optimizer update.
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        smoke_loss = flow_loss(train_index[:BATCH])
    smoke_loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
    assert all(p.grad is None for p in model.vlm.parameters())
    assert (
        model.action_out.weight.grad is not None
        and torch.isfinite(model.action_out.weight.grad).all()
    )
    save(
        "smoke.json",
        dict(
            loss=float(smoke_loss),
            grad_norm=float(norm),
            backward=True,
            frozen_vlm_has_no_grad=True,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    phase("train_action_expert")
    history = []
    updates = 0
    best = float("inf")
    best_epoch = None
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        order = train_index[torch.randperm(len(train_index), device="cuda")]
        total = count = 0
        for rows in order.split(BATCH):
            updates += 1
            ratio = min(1.0, updates / 100)
            if updates > 100:
                ratio = 0.1 + 0.45 * (
                    1
                    + math.cos(
                        math.pi
                        * (updates - 100)
                        / max(settings["optimizer_updates_target"] - 100, 1)
                    )
                )
            for group in optimizer.param_groups:
                group["lr"] = 1e-4 * ratio
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = flow_loss(rows)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            weight = int(masks[rows].sum()) * 6
            total += float(loss.detach()) * weight
            count += weight
        row = dict(
            epoch=epoch,
            optimizer_updates=updates,
            train_loss=total / count,
            elapsed_seconds=time.time() - start,
            lr=optimizer.param_groups[0]["lr"],
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            row.update(evaluate(False))
            if row["validation_flow_mse"] < best:
                best = row["validation_flow_mse"]
                best_epoch = epoch
                model.save_checkpoint(ROOT / "model/best_vla.pt")
        history.append(row)
        save("history.json", history)
        save(
            "progress.json",
            dict(
                phase="train",
                total_epochs=EPOCHS,
                total_optimizer_updates=settings["optimizer_updates_target"],
                best_epoch=best_epoch,
                best_val_loss=best,
                time=time.time(),
                **row,
            ),
        )
        print("EPOCH", json.dumps(row), flush=True)
        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            model.save_checkpoint(ROOT / "model/final_vla.pt")
            save(
                "checkpoint.json",
                dict(
                    epoch=epoch,
                    optimizer_updates=updates,
                    sha256=sha(ROOT / "model/final_vla.pt"),
                ),
            )
    assert updates == settings["optimizer_updates_target"]
    phase("final_evaluation_and_verification")
    final = evaluate(True, "validation_predictions_final.npz")
    final.update(
        epochs=EPOCHS,
        optimizer_updates=updates,
        best_epoch=best_epoch,
        best_validation_flow_mse=best,
        training_seconds=time.time() - start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    assert frozen_digest(model) == frozen_before
    final["frozen_vlm_bitwise_unchanged"] = True
    assert all(torch.isfinite(p).all() for p in model.parameters())
    final["finite_weights"] = True
    # Check the serialized native VLA including tensor shapes and a real image path.
    restored = MiniMindVLA.load_checkpoint(ROOT / "model/final_vla.pt")
    assert all(torch.isfinite(p).all() for p in restored.parameters())
    restored.cuda().eval()
    first = int(val_ids[0])
    image = ROOT / f"examples/frame_{first}.jpg"
    processor = VLAProcessor(cfg, restored.vlm.processor)
    b = collate_batch(
        [
            processor(
                image,
                tasks[int(table["task_index"][first])],
                normalized_state[first].tolist(),
            )
        ]
    )
    b = {k: v.cuda() for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sample = restored.sample_actions(**b, steps=10)
    assert sample.shape == (1, HORIZON, 6) and torch.isfinite(sample).all()
    final.update(
        strict_load=True,
        image_inference_finite=True,
        checkpoint_sha256=sha(ROOT / "model/final_vla.pt"),
    )
    save("evaluation_after.json", final)
    raw_sample = sample[0].float().cpu().numpy() * np.asarray(
        stats["action"]["std"]
    ) + np.asarray(stats["action"]["mean"])
    save(
        "example_prediction.json",
        dict(
            row_index=first,
            episode=int(episode_ids[first]),
            instruction=tasks[int(table["task_index"][first])],
            actions=raw_sample.tolist(),
            state=state[first].tolist(),
        ),
    )
    # Evaluate the validation-selected best checkpoint as a separate selectable policy.
    model.load_state_dict(
        torch.load(ROOT / "model/best_vla.pt", map_location="cuda", weights_only=True)[
            "model"
        ],
        strict=False,
    )
    save(
        "evaluation_best.json",
        dict(epoch=best_epoch, **evaluate(True, "validation_predictions_best.npz")),
    )
    save("model/config.json", asdict(cfg))
    assert all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    torch.save(
        dict(optimizer=optimizer.state_dict(), epoch=EPOCHS, optimizer_updates=updates),
        ROOT / "model/training_state.pt",
    )
    (ROOT / "requirements-resolved.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    phase("verify_durable_artifacts")
    files = [
        p
        for sub in ["model", "examples", "source"]
        for p in (ROOT / sub).rglob("*")
        if p.is_file()
    ]
    files += [
        p
        for p in ROOT.glob("*.json")
        if p.name
        not in [
            "SHA256.json",
            "progress.json",
            "phase.json",
            "pipeline_phase.json",
            "pipeline_error.json",
        ]
    ]
    files += list(ROOT.glob("*.npz"))
    files += [ROOT / "requirements-resolved.txt"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    save("SHA256.json", hashes)
    assert all(sha(ROOT / name) == expected for name, expected in hashes.items())
    save(
        "SUCCESS",
        dict(
            status="complete",
            epochs=EPOCHS,
            optimizer_updates=updates,
            model_sha256=hashes["model/final_vla.pt"],
            verified_files=len(hashes),
            shared_volume_verified=True,
        ),
    )
    phase("complete")
    print("TRAINING_COMPLETE", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
