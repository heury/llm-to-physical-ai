"""CPU-first supervised decision training; optionally initialize from a native MiniMind LM."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ByteTokenizer, DecisionCollator, DecisionDataset
from model.decision import DecisionModel, save_checkpoint
from model.model_minimind import MiniMindConfig


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            valid = batch["labels"] != -100
            size = int(valid.sum())
            if size == 0:
                continue
            total_loss += output["loss"].item() * size
            correct += ((output["logits"].argmax(-1) == batch["labels"]) & valid).sum().item()
            count += size
    if count == 0:
        raise ValueError("Evaluation requires at least one labelled question")
    return {"loss": total_loss / count, "accuracy": correct / count, "questions": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(Path(__file__).parent / "data/train.jsonl"))
    parser.add_argument("--validation", help="Optional held-out labelled JSONL")
    parser.add_argument("--output", default="out/decision")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--max-length", type=int, default=512, help="Total packed tokens per record")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--lm-checkpoint", help="Native dense MiniMind LM state_dict .pth")
    parser.add_argument("--lm-config", help="Exact source MiniMind config JSON, required with --lm-checkpoint")
    parser.add_argument("--tokenizer", help="Local MiniMind tokenizer directory, required with --lm-checkpoint")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.threads < 1:
        parser.error("epochs, batch-size, lr and threads must be positive")
    if bool(args.lm_checkpoint) != bool(args.lm_config) or (args.lm_checkpoint and not args.tokenizer):
        parser.error("LM initialization needs --lm-checkpoint, --lm-config and --tokenizer together")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True) if args.tokenizer else ByteTokenizer()
    if args.lm_config:
        config = MiniMindConfig(**json.loads(Path(args.lm_config).read_text(encoding="utf-8")))
    else:
        config = MiniMindConfig(hidden_size=64, num_hidden_layers=2, vocab_size=len(tokenizer),
                                num_attention_heads=4, num_key_value_heads=2,
                                intermediate_size=192, max_position_embeddings=args.max_length)
    if len(tokenizer) != config.vocab_size:
        parser.error("tokenizer vocabulary does not match LM config")
    if args.max_length > config.max_position_embeddings:
        parser.error("max-length exceeds config.max_position_embeddings")
    model = DecisionModel(config)
    if args.lm_checkpoint:
        model.load_lm_state_dict(torch.load(args.lm_checkpoint, map_location="cpu", weights_only=True))
    if args.freeze_backbone:
        model.freeze_backbone()
    model.to(args.device)
    collate = DecisionCollator(tokenizer, args.max_length)
    dataset = DecisionDataset(args.data)
    # Validate whole packed records before optimization (no silent truncation).
    for row in dataset:
        collate([row])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    evaluation = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    validation = DataLoader(DecisionDataset(args.validation), batch_size=args.batch_size, collate_fn=collate) if args.validation else None
    if validation is not None:
        for row in validation.dataset:
            collate([row])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    history = [{"epoch": 0, "train": evaluate(model, evaluation, args.device)}]
    print(json.dumps(history[-1]), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_backbone:
            model.backbone.eval()
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        metrics = {"epoch": epoch, "train": evaluate(model, evaluation, args.device)}
        if validation is not None:
            metrics["validation"] = evaluate(model, validation, args.device)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
    save_checkpoint(args.output, model, tokenizer, args.max_length)
    report = {"arguments": vars(args), "parameters": sum(p.numel() for p in model.parameters()),
              "history": history}
    (Path(args.output) / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
