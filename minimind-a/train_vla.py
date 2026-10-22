"""정답 행동 청크로 속도장을 학습하는 단일 프로세스 trainer (CPU/CUDA)."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from dataset.vla_dataset import VLADataset, collate_batch
from model.model_vla import MiniMindVLA, VLAConfig


def evaluate(model, loader, device):
    """동일한 noise/time으로 각 epoch의 flow loss를 비교한다."""
    model.eval()
    total = 0.0
    count = 0
    # Fixed validation corruption permits comparison across epochs.
    generator = torch.Generator(device=device).manual_seed(123)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            noise = torch.randn(
                batch["actions"].shape, generator=generator, device=device
            )
            time = torch.rand(
                batch["actions"].size(0), generator=generator, device=device
            )
            loss = model(**batch, noise=noise, time=time)
            # 청크 길이가 달라도 유효한 행동 원소 수로 가중 평균한다.
            weight = int(batch["action_mask"].sum()) * model.config.action_dim
            total += loss.item() * weight
            count += weight
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/toy/train.jsonl")
    parser.add_argument("--val-data", default=None)
    parser.add_argument(
        "--config",
        help="VLAConfig JSON; omitted = small language model with native SigLIP",
    )
    parser.add_argument(
        "--vision-model-path", help="local MiniMind-V SigLIP model/processor directory"
    )
    parser.add_argument("--vlm-checkpoint", help="native MiniMind-V state_dict .pth")
    parser.add_argument("--output", default="out/toy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("positive training settings required")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = (
        VLAConfig(**json.loads(Path(args.config).read_text()))
        if args.config
        else VLAConfig()
    )
    if config.freeze_vlm and not args.vlm_checkpoint:
        parser.error(
            "freeze_vlm=true requires --vlm-checkpoint; "
            "set freeze_vlm=false to train a randomly initialized VLM"
        )
    if args.vision_model_path:
        config.vision_model_path = str(Path(args.vision_model_path).resolve())
    model = MiniMindVLA(config)
    print(
        f"VLM language layers: {len(model.vlm.model.layers)} used / "
        f"{config.vlm_config.get('num_hidden_layers', 8)} source (first half)"
    )
    if args.vlm_checkpoint:
        model.load_vlm_weights(args.vlm_checkpoint)
    model.to(args.device)
    data = VLADataset(args.data, config, model.vlm.processor)
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch
    )
    val = None
    if args.val_data:
        val = DataLoader(
            VLADataset(args.val_data, config, model.vlm.processor),
            batch_size=args.batch_size,
            collate_fn=collate_batch,
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    if val:
        history.append(dict(epoch=0, val_loss=evaluate(model, val, args.device)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
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
            row["val_loss"] = evaluate(model, val, args.device)
        history.append(row)
        print(json.dumps(row), flush=True)
    model.save_checkpoint(output / "vla.pt")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (output / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {output}/vla.pt")


if __name__ == "__main__":
    main()
