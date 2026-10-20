"""Offline RGB reaching task: pick red/blue target; predict 2-D positions."""

import argparse
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw


def generate(output, count=128, seed=42, chunk_size=8):
    root = Path(output)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        red = [rng.uniform(-0.8, 0.8) for _ in range(2)]
        blue = [rng.uniform(-0.8, 0.8) for _ in range(2)]
        state = [rng.uniform(-0.8, 0.8) for _ in range(2)]
        color = rng.choice(["red", "blue"])
        target = red if color == "red" else blue
        image = Image.new("RGB", (64, 64), (30, 30, 30))
        draw = ImageDraw.Draw(image)
        for point, fill in [(red, "red"), (blue, "blue"), (state, "white")]:
            x, y = [(v + 1) * 31.5 for v in point]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=fill)
        name = f"images/{i:05d}.png"
        image.save(root / name)
        # Absolute normalized XY setpoints, NOT deltas or motor commands.
        actions = [
            [s + (t - s) * (step + 1) / chunk_size for s, t in zip(state, target)]
            for step in range(chunk_size)
        ]
        rows.append(
            dict(
                image=name,
                instruction=f"move to the {color} target",
                state=state,
                actions=actions,
            )
        )
    split = max(1, int(count * 0.8))
    for name, part in [("train", rows[:split]), ("val", rows[split:])]:
        (root / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in part) + "\n"
        )
    print(
        f"generated {len(rows[:split])} train / {len(rows[split:])} val records in {root}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/toy")
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=8)
    a = p.parse_args()
    if a.count < 2 or a.chunk_size < 1:
        p.error("count >= 2 and chunk-size >= 1 required")
    generate(a.output, a.count, a.seed, a.chunk_size)
