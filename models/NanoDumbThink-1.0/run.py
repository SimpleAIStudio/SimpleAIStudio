import json
import re
from pathlib import Path

import torch

from data import SCALE, Sample, make_batch
from model import NanoDumbThink


ROOT = Path(__file__).parent


with open(
    ROOT / "config.json",
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)


model = NanoDumbThink(
    hidden=config["hidden"],
    middle=config["middle"],
    think_steps=config["think_steps"],
)

model.load_state_dict(
    torch.load(
        ROOT / "weights.pt",
        map_location="cpu",
    )
)

model.eval()


def parse_problem(text):
    patterns = [
        (
            r"(\d+)\s*\+\s*(\d+)",
            "add",
            "number",
        ),
        (
            r"(\d+)\s*-\s*(\d+)",
            "sub",
            "number",
        ),
        (
            r"(\d+)\s*>\s*(\d+)",
            "gt",
            "logic",
        ),
        (
            r"(\d+)\s*<\s*(\d+)",
            "lt",
            "logic",
        ),
    ]

    for pattern, task, kind in patterns:
        match = re.fullmatch(
            pattern,
            text,
        )

        if match:
            return Sample(
                task,
                tuple(
                    map(
                        int,
                        match.groups(),
                    )
                ),
                0,
                kind,
            )

    match = re.fullmatch(
        r"(\d+)\s+(\d+)\s+(\d+)\s+\?",
        text,
    )

    if match:
        return Sample(
            "next",
            tuple(
                map(
                    int,
                    match.groups(),
                )
            ),
            0,
            "number",
        )

    return None


def solve(sample):
    batch = make_batch([sample])

    with torch.no_grad():
        value = model(
            batch["task"],
            batch["numbers"],
            batch["mask"],
        )[0].item()

    if sample.kind == "logic":
        return (
            "YES"
            if value > 0
            else "NO"
        )

    return str(
        round(value * SCALE)
    )


print()
print("Nano DumbThink 1.0")
print(f"{config['parameters']:,} parameters")
print(f"{config['think_steps']} thinking steps")
print()
print("Examples:")
print("  7 + 12")
print("  14 - 3")
print("  9 > 4")
print("  3 < 1")
print("  2 5 8 ?")
print()
print("Input range: 0-20")
print("/exit to quit")
print()


while True:
    try:
        text = input(
            "Problem > "
        ).strip()

    except (
        KeyboardInterrupt,
        EOFError,
    ):
        print()
        break

    if not text:
        continue

    if text.lower() in {
        "/exit",
        "/quit",
    }:
        break

    sample = parse_problem(text)

    if sample is None:
        print(
            "DumbThink > I don't know "
            "that kind of problem yet."
        )
        print()
        continue

    if not all(
        0 <= value <= 20
        for value in sample.numbers
    ):
        print(
            "DumbThink > use numbers "
            "from 0 to 20."
        )
        print()
        continue

    print(
        f"DumbThink > {solve(sample)}"
    )
    print()