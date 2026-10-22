"""Answer choice/score/noul questions in one forward; no autoregressive generation."""
import argparse
import json
from pathlib import Path

import torch

from dataset import DecisionCollator, validate_row
from model.decision import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="out/decision")
    parser.add_argument("--request", help="JSON file with input and typed questions")
    parser.add_argument("--input", help="Single-choice convenience input")
    parser.add_argument("--choices", nargs="+", help="Single-choice convenience candidates")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.request:
        if args.input is not None or args.choices is not None:
            parser.error("Use either --request or --input with --choices")
        row = json.loads(Path(args.request).read_text(encoding="utf-8"))
    else:
        if args.input is None or not args.choices:
            parser.error("Provide --request, or both --input and --choices")
        row = {"input": args.input, "choices": args.choices}
    row = validate_row(row, require_label=False)
    torch.set_num_threads(1)
    model, tokenizer, length = load_checkpoint(args.checkpoint, args.device)
    batch = DecisionCollator(tokenizer, length)([row])
    batch = {key: value.to(args.device) for key, value in batch.items()}
    with torch.inference_mode():
        probabilities = model(**batch)["probabilities"]
    print(json.dumps(model.decode([row], probabilities)[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
