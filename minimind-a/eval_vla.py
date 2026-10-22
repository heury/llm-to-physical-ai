"""Predict a normalized action chunk for a JSONL sample, optionally score targets."""

import argparse
import json
from pathlib import Path

import torch
from dataset.vla_dataset import VLAProcessor, collate_batch
from model.model_vla import MiniMindVLA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/toy/vla.pt")
    parser.add_argument("--data", default="data/toy/val.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--vision-model-path", help="override relocated local SigLIP directory"
    )
    parser.add_argument("--output", help="optional JSON output file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = (
        MiniMindVLA.load_checkpoint(args.checkpoint, args.vision_model_path)
        .to(args.device)
        .eval()
    )
    path = Path(args.data)
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    row = rows[args.index]
    processor = VLAProcessor(model.config, model.vlm.processor)
    item = processor(path.parent / row["image"], row["instruction"], row["state"])
    batch = {k: v.to(args.device) for k, v in collate_batch([item]).items()}
    actions = model.sample_actions(**batch, steps=args.steps)[0].cpu()
    result = dict(
        instruction=row["instruction"],
        actions=actions.tolist(),
        units="normalized; same units as training",
    )
    if "actions" in row:
        target = torch.tensor(row["actions"], dtype=torch.float32)
        if (
            target.ndim != 2
            or target.size(1) != model.config.action_dim
            or not 1 <= len(target) <= len(actions)
            or not torch.isfinite(target).all()
        ):
            raise ValueError("invalid target actions")
        result["mse"] = ((actions[: len(target)] - target) ** 2).mean().item()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
