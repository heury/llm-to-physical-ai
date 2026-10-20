"""Minimal supervised flow-matching trainer (single process, CPU or CUDA)."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from dataset.vla_dataset import VLADataset, collate_batch
from model.model_vla import MiniMindVLA, VLAConfig


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    # Fixed validation corruption permits comparison across epochs.
    generator = torch.Generator(device=device).manual_seed(123)
    with torch.no_grad():
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            noise = torch.randn(b["actions"].shape, generator=generator, device=device)
            time = torch.rand(b["actions"].size(0), generator=generator, device=device)
            loss = model(**b, noise=noise, time=time)
            weight = int(b["action_mask"].sum()) * model.config.action_dim
            total += loss.item() * weight
            count += weight
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/toy/train.jsonl")
    p.add_argument("--val-data", default=None)
    p.add_argument(
        "--config",
        help="VLAConfig JSON; omitted = small language model with native SigLIP",
    )
    p.add_argument(
        "--vision-model-path", help="local MiniMind-V SigLIP model/processor directory"
    )
    p.add_argument("--vlm-checkpoint", help="native MiniMind-V state_dict .pth")
    p.add_argument("--output", default="out/toy")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.threads) < 1 or a.lr <= 0:
        p.error("positive training settings required")
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    config = (
        VLAConfig(**json.loads(Path(a.config).read_text())) if a.config else VLAConfig()
    )
    if config.freeze_vlm and not a.vlm_checkpoint:
        p.error(
            "freeze_vlm=true requires --vlm-checkpoint; "
            "set freeze_vlm=false to train a randomly initialized VLM"
        )
    if a.vision_model_path:
        config.vision_model_path = str(Path(a.vision_model_path).resolve())
    model = MiniMindVLA(config)
    print(
        f"VLM language layers: {len(model.vlm.model.layers)} used / "
        f"{config.vlm_config.get('num_hidden_layers', 8)} source (first half)"
    )
    if a.vlm_checkpoint:
        model.load_vlm_weights(a.vlm_checkpoint)
    model.to(a.device)
    data = VLADataset(a.data, config, model.vlm.processor)
    loader = DataLoader(
        data, batch_size=a.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val = None
    if a.val_data:
        val = DataLoader(
            VLADataset(a.val_data, config, model.vlm.processor),
            batch_size=a.batch_size,
            collate_fn=collate_batch,
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr
    )
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    if val:
        history.append(dict(epoch=0, val_loss=evaluate(model, val, a.device)))
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = {k: v.to(a.device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weight = int(batch["action_mask"].sum()) * config.action_dim
            total += loss.item() * weight
            count += weight
        row = dict(epoch=epoch, train_loss=total / count)
        if val:
            row["val_loss"] = evaluate(model, val, a.device)
        history.append(row)
        print(json.dumps(row), flush=True)
    model.save_checkpoint(output / "vla.pt")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {output}/vla.pt")


if __name__ == "__main__":
    main()
