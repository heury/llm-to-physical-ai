"""Predict a normalized action chunk for a JSONL sample, optionally score targets."""

import argparse
import json
from pathlib import Path

import torch
from dataset.vla_dataset import VLAProcessor, collate_batch
from model.model_vla import MiniMindVLA


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="out/toy/vla.pt")
    p.add_argument("--data", default="data/toy/val.jsonl")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--vision-model-path", help="override relocated local SigLIP directory"
    )
    p.add_argument("--output", help="optional JSON output file")
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    model = (
        MiniMindVLA.load_checkpoint(a.checkpoint, a.vision_model_path)
        .to(a.device)
        .eval()
    )
    path = Path(a.data)
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    r = rows[a.index]
    processor = VLAProcessor(model.config, model.vlm.processor)
    item = processor(path.parent / r["image"], r["instruction"], r["state"])
    b = {k: v.to(a.device) for k, v in collate_batch([item]).items()}
    actions = model.sample_actions(**b, steps=a.steps)[0].cpu()
    result = dict(
        instruction=r["instruction"],
        actions=actions.tolist(),
        units="normalized; same units as training",
    )
    if "actions" in r:
        target = torch.tensor(r["actions"], dtype=torch.float32)
        if (
            target.ndim != 2
            or target.size(1) != model.config.action_dim
            or not 1 <= len(target) <= len(actions)
        ):
            raise ValueError("invalid target actions")
        result["mse"] = ((actions[: len(target)] - target) ** 2).mean().item()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(text + "\n")


if __name__ == "__main__":
    main()
