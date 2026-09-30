import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from data import Sample, decode, make_batch
from model import MicroJet


root = Path(__file__).parent

seed = 20261001
tests_per_task = 30

rng = random.Random(seed)


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
        root / "weights_release_candidate.pt",
        map_location="cpu",
    )
)

model.eval()


def choose(samples):
    rng.shuffle(samples)
    return samples[:tests_per_task]


tests = []


# Addition
candidates = []

for a in range(25):
    for b in range(25):
        if a <= 20 and b <= 20:
            continue

        candidates.append(
            Sample(
                "add",
                (a, b),
                a + b,
                "number",
            )
        )

tests.extend(choose(candidates))


# Subtraction
candidates = []

for a in range(25):
    for b in range(25):
        if a <= 20 and b <= 20:
            continue

        candidates.append(
            Sample(
                "sub",
                (a, b),
                a - b,
                "number",
            )
        )

tests.extend(choose(candidates))


# Greater than
candidates = []

for a in range(25):
    for b in range(25):
        if a <= 20 and b <= 20:
            continue

        candidates.append(
            Sample(
                "gt",
                (a, b),
                float(a > b),
                "logic",
            )
        )

tests.extend(choose(candidates))


# Less than
candidates = []

for a in range(25):
    for b in range(25):
        if a <= 20 and b <= 20:
            continue

        candidates.append(
            Sample(
                "lt",
                (a, b),
                float(a < b),
                "logic",
            )
        )

tests.extend(choose(candidates))


# Multiplication
# Training only saw 0-12.
candidates = []

for a in range(21):
    for b in range(21):
        if a <= 12 and b <= 12:
            continue

        if a > 15 or b > 15:
            continue

        candidates.append(
            Sample(
                "mul",
                (a, b),
                a * b,
                "number",
            )
        )

tests.extend(choose(candidates))


# Exact division
# Training dividends were <= 20.
candidates = []

for divisor in range(1, 11):
    for answer in range(1, 21):
        value = divisor * answer

        if not 21 <= value <= 40:
            continue

        candidates.append(
            Sample(
                "div",
                (value, divisor),
                answer,
                "number",
            )
        )

tests.extend(choose(candidates))


# Arithmetic sequences
# Training sequence values stayed <= 20.
candidates = []

for start in range(25):
    for step in range(-4, 5):
        if step == 0:
            continue

        values = tuple(
            start + step * i
            for i in range(4)
        )

        answer = start + step * 4

        if not all(
            0 <= value <= 24
            for value in values
        ):
            continue

        if not 0 <= answer <= 24:
            continue

        if max(values + (answer,)) <= 20:
            continue

        candidates.append(
            Sample(
                "next",
                values,
                answer,
                "number",
            )
        )

tests.extend(choose(candidates))


# Three-number sum
# Training only saw each input from 0-8.
candidates = []

for a in range(11):
    for b in range(11):
        for c in range(11):
            if max(a, b, c) <= 8:
                continue

            candidates.append(
                Sample(
                    "sum3",
                    (a, b, c),
                    a + b + c,
                    "number",
                )
            )

tests.extend(choose(candidates))


# Max of three
candidates = []

for a in range(11):
    for b in range(11):
        for c in range(11):
            if max(a, b, c) <= 8:
                continue

            candidates.append(
                Sample(
                    "max3",
                    (a, b, c),
                    max(a, b, c),
                    "number",
                )
            )

tests.extend(choose(candidates))


# Chained comparison
# Training only saw each input from 0-10.
candidates = []

for a in range(13):
    for b in range(13):
        for c in range(13):
            if max(a, b, c) <= 10:
                continue

            candidates.append(
                Sample(
                    "chain_gt",
                    (a, b, c),
                    float(
                        a > b
                        and b > c
                    ),
                    "logic",
                )
            )

tests.extend(choose(candidates))


scores = defaultdict(
    lambda: [0, 0]
)

failures = []


with torch.no_grad():
    for start in range(
        0,
        len(tests),
        256,
    ):
        chunk = tests[
            start:start + 256
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
            prediction = decode(
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

            scores[sample.task][1] += 1

            if prediction == expected:
                scores[sample.task][0] += 1

            elif len(failures) < 20:
                failures.append(
                    (
                        sample.task,
                        sample.numbers,
                        expected,
                        prediction,
                    )
                )


task_order = [
    "add",
    "sub",
    "mul",
    "div",
    "gt",
    "lt",
    "next",
    "sum3",
    "max3",
    "chain_gt",
]


correct_total = 0
test_total = 0
results = {}


print()
print("Micro Jet DumbThink 2.0")
print("Generalization Challenge")
print("------------------------")
print()
print(
    "These problems are outside the ranges "
    "used during training."
)
print()


for task in task_order:
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
        f"{correct:>2}/{total:<2} "
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
            f"{text:<16} "
            f"wanted {expected}, "
            f"got {predicted}"
        )


result = {
    "model":
        "Micro Jet DumbThink",

    "version":
        "2.0",

    "benchmark":
        "generalization",

    "seed":
        seed,

    "tests_per_task":
        tests_per_task,

    "total_tests":
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
    root / "generalization_benchmark.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        result,
        file,
        indent=2,
    )


print()
print(
    "Saved generalization_benchmark.json"
)
print()