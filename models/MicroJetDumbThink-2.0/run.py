import json
import re
from pathlib import Path

import torch

from data import (
    Sample,
    decode,
    make_batch,
)
from model import MicroJet


root = Path(__file__).parent


with open(
    root / "config.json",
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)


model = MicroJet(
    hidden=config["hidden"],
    middle=config["middle"],
    think_steps=config["think_steps"],
)

model.load_state_dict(
    torch.load(
        root / "weights.pt",
        map_location="cpu",
    )
)

model.eval()


def parse(text):
    text = text.strip()

    checks = [
        (
            r"(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)",
            "sum3",
            "number",
        ),
        (
            r"(\d+)\s*>\s*(\d+)\s*>\s*(\d+)",
            "chain_gt",
            "logic",
        ),
        (
            r"max\s+(\d+)\s+(\d+)\s+(\d+)",
            "max3",
            "number",
        ),
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
            r"(\d+)\s*[\*x]\s*(\d+)",
            "mul",
            "number",
        ),
        (
            r"(\d+)\s*/\s*(\d+)",
            "div",
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
        (
            r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\?",
            "next",
            "number",
        ),
    ]

    for pattern, task, kind in checks:
        match = re.fullmatch(
            pattern,
            text,
            flags=re.IGNORECASE,
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

    return None


def solve(sample):
    batch = make_batch([sample])

    with torch.no_grad():
        raw = model(
            batch["task"],
            batch["numbers"],
            batch["mask"],
        )[0].item()

    answer = decode(
        sample,
        raw,
    )

    if sample.kind == "logic":
        return (
            "YES"
            if answer
            else "NO"
        )

    return str(answer)


print()
print("Micro Jet DumbThink 2.0")
print("-----------------------")
print()
print(
    f"{config['parameters']:,} parameters"
)
print(
    f"{config['think_steps']} thinking steps"
)

print()
print("Try:")
print("  7 + 12")
print("  14 - 3")
print("  6 * 7")
print("  18 / 3")
print("  9 > 4")
print("  3 < 1")
print("  2 5 8 11 ?")
print("  3 + 4 + 5")
print("  max 3 8 5")
print("  9 > 6 > 2")

print()
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

    sample = parse(text)

    if sample is None:
        print(
            "Micro Jet > I don't know "
            "that problem type yet."
        )
        print()
        continue

    if any(
        value < 0
        or value > 20
        for value in sample.numbers
    ):
        print(
            "Micro Jet > use numbers "
            "from 0 to 20."
        )
        print()
        continue

    if (
        sample.task == "div"
        and sample.numbers[1] == 0
    ):
        print(
            "Micro Jet > nope."
        )
        print()
        continue

    print(
        f"Micro Jet > {solve(sample)}"
    )
    print()