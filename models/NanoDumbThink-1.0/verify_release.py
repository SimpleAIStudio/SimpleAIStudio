from pathlib import Path

import torch

from data import build_samples, make_batch
from model import NanoDumbThink

import sys


ROOT = Path(__file__).parent

STUDIO = ROOT.parents[1]

RELEASE = (
    STUDIO
    / "releases"
    / "NanoDumbThink-1.0"
)

sys.path.insert(
    0,
    str(RELEASE),
)

from runtime import NanoRuntime


import json


with open(
    ROOT / "config.json",
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)


torch_model = NanoDumbThink(
    hidden=config["hidden"],
    middle=config["middle"],
    think_steps=config["think_steps"],
)

torch_model.load_state_dict(
    torch.load(
        ROOT / "weights.pt",
        map_location="cpu",
    )
)

torch_model.eval()


numpy_model = NanoRuntime(
    RELEASE
)


_, tests = build_samples()


max_difference = 0.0
answer_mismatches = []


with torch.no_grad():
    for sample in tests:
        batch = make_batch([sample])

        torch_value = torch_model(
            batch["task"],
            batch["numbers"],
            batch["mask"],
        )[0].item()

        numpy_value = numpy_model.forward(
            sample.task,
            sample.numbers,
        )

        difference = abs(
            torch_value - numpy_value
        )

        max_difference = max(
            max_difference,
            difference,
        )

        if sample.kind == "number":
            torch_answer = round(
                torch_value
                * config["scale"]
            )

            numpy_answer = round(
                numpy_value
                * config["scale"]
            )

        else:
            torch_answer = (
                torch_value > 0
            )

            numpy_answer = (
                numpy_value > 0
            )

        if torch_answer != numpy_answer:
            answer_mismatches.append(
                (
                    sample.task,
                    sample.numbers,
                    torch_answer,
                    numpy_answer,
                )
            )


print()
print("Nano DumbThink export check")
print("---------------------------")
print()
print(f"Problems checked : {len(tests)}")
print(f"Answer mismatches: {len(answer_mismatches)}")
print(f"Max raw difference: {max_difference:.10f}")
print()


if answer_mismatches:
    print("FAILED")
    print()

    for item in answer_mismatches[:10]:
        print(item)

else:
    print("PASS")
    print()
    print(
        "NumPy runtime matches "
        "the frozen PyTorch model."
    )

print()