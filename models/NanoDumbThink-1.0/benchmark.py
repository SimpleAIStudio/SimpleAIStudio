import json
from collections import defaultdict
from pathlib import Path

import torch

from data import SCALE, build_samples, make_batch
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


_, test_samples = build_samples()

scores = defaultdict(
    lambda: [0, 0]
)

errors = defaultdict(list)
failures = []


with torch.no_grad():
    for sample in test_samples:
        batch = make_batch([sample])

        value = model(
            batch["task"],
            batch["numbers"],
            batch["mask"],
        )[0].item()

        if sample.kind == "number":
            raw = value * SCALE

            predicted = round(raw)
            expected = int(sample.answer)

            errors[sample.task].append(
                abs(raw - expected)
            )

        else:
            predicted = value > 0
            expected = bool(sample.answer)

        correct = predicted == expected

        scores[sample.task][1] += 1

        if correct:
            scores[sample.task][0] += 1

        elif len(failures) < 8:
            failures.append(
                (
                    sample.task,
                    sample.numbers,
                    expected,
                    predicted,
                    value,
                )
            )


total_correct = 0
total_tests = 0
results = {}


print()
print("Nano DumbThink benchmark")
print("------------------------")
print("held-out problems inside the 0-20 range")
print()


for task in [
    "add",
    "sub",
    "gt",
    "lt",
    "next",
]:
    correct, total = scores[task]

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    total_correct += correct
    total_tests += total

    results[task] = {
        "correct": correct,
        "total": total,
        "accuracy": round(accuracy, 2),
    }

    extra = ""

    if errors[task]:
        mae = (
            sum(errors[task])
            / len(errors[task])
        )

        extra = f"  mae {mae:.3f}"

        results[task]["mae"] = round(
            mae,
            4,
        )

    print(
        f"{task:<8} "
        f"{correct:>3}/{total:<3} "
        f"{accuracy:>6.2f}%"
        f"{extra}"
    )


overall = (
    total_correct
    / total_tests
    * 100
)


print()
print(
    f"Overall  "
    f"{total_correct}/{total_tests} "
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
        raw,
    ) in failures:
        text = " ".join(
            str(x)
            for x in numbers
        )

        print(
            f"  {task:<5} "
            f"{text:<10} "
            f"wanted {expected}, "
            f"got {predicted} "
            f"(raw {raw:.4f})"
        )


benchmark = {
    "model": config["name"],
    "version": config["version"],
    "parameters": config["parameters"],
    "test_samples": total_tests,
    "overall_accuracy": round(overall, 2),
    "tasks": results,
}


with open(
    ROOT / "benchmark.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        benchmark,
        file,
        indent=2,
    )


print()
print("Saved benchmark.json")
print()