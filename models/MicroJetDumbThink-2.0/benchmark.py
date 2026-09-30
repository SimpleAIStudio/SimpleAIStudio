import json
from collections import defaultdict
from pathlib import Path

import torch

from data import (
    build_samples,
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


_, tests = build_samples()

scores = defaultdict(
    lambda: [0, 0]
)

failures = []


with torch.no_grad():
    for start in range(
        0,
        len(tests),
        512,
    ):
        chunk = tests[
            start:start + 512
        ]

        batch = make_batch(chunk)

        output = model(
            batch["task"],
            batch["numbers"],
            batch["mask"],
        )

        for sample, raw in zip(
            chunk,
            output.tolist(),
        ):
            predicted = decode(
                sample,
                raw,
            )

            if sample.kind == "logic":
                expected = bool(
                    sample.answer
                )
            else:
                expected = int(
                    sample.answer
                )

            scores[
                sample.task
            ][1] += 1

            if predicted == expected:
                scores[
                    sample.task
                ][0] += 1

            elif len(failures) < 12:
                failures.append(
                    (
                        sample.task,
                        sample.numbers,
                        expected,
                        predicted,
                    )
                )


correct_total = 0
test_total = 0
results = {}


print()
print("Micro Jet DumbThink benchmark")
print("-----------------------------")
print("held-out problems only")
print()


for task in config["tasks"]:
    correct, total = scores[task]

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    correct_total += correct
    test_total += total

    results[task] = {
        "correct": correct,
        "total": total,
        "accuracy": round(
            accuracy,
            2,
        ),
    }

    print(
        f"{task:<10} "
        f"{correct:>4}/{total:<4} "
        f"{accuracy:>6.2f}%"
    )


overall = (
    correct_total
    / test_total
    * 100
)


print()
print(
    f"Overall    "
    f"{correct_total}/{test_total} "
    f"{overall:.2f}%"
)


if failures:
    print()
    print("A few misses:")
    print()

    for (
        task,
        numbers,
        expected,
        predicted,
    ) in failures:

        text = " ".join(
            str(value)
            for value in numbers
        )

        print(
            f"  {task:<10} "
            f"{text:<14} "
            f"wanted {expected}, "
            f"got {predicted}"
        )


result = {
    "model":
        "Micro Jet DumbThink",

    "version":
        "2.0",

    "parameters":
        config["parameters"],

    "test_samples":
        test_total,

    "overall_accuracy":
        round(
            overall,
            2,
        ),

    "tasks":
        results,
}


with open(
    root / "benchmark.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result,
        file,
        indent=2,
    )


print()
print("Saved benchmark.json")
print()