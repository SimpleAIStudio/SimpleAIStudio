import json
from pathlib import Path

import torch

from data import (
    build_samples,
    decode,
    make_batch,
)
from model import MicroJet
from runtime import MicroJetRuntime


root = Path(__file__).parent

release = (
    root.parent.parent
    / "releases"
    / "MicroJetDumbThink-2.0"
)


with open(
    root / "config.json",
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)


torch_model = MicroJet(
    hidden=config["hidden"],
    middle=config["middle"],
    think_steps=config["think_steps"],
)

torch_model.load_state_dict(
    torch.load(
        root / "weights_release_candidate.pt",
        map_location="cpu",
    )
)

torch_model.eval()


numpy_model = MicroJetRuntime(
    release / "weights.npz"
)


_, tests = build_samples()


answer_mismatches = 0
max_raw_difference = 0.0


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

        torch_output = torch_model(
            batch["task"],
            batch["numbers"],
            batch["mask"],
        ).tolist()

        for sample, torch_raw in zip(
            chunk,
            torch_output,
        ):
            numpy_raw = (
                numpy_model.predict_raw(
                    sample.task,
                    sample.numbers,
                )
            )

            difference = abs(
                torch_raw
                - numpy_raw
            )

            max_raw_difference = max(
                max_raw_difference,
                difference,
            )

            torch_answer = decode(
                sample,
                torch_raw,
            )

            numpy_answer = decode(
                sample,
                numpy_raw,
            )

            if (
                torch_answer
                != numpy_answer
            ):
                answer_mismatches += 1


print()
print(
    "Micro Jet DumbThink "
    "release verification"
)

print(
    "--------------------"
    "------------------"
)

print()

print(
    f"Problems checked : "
    f"{len(tests):,}"
)

print(
    f"Answer mismatches: "
    f"{answer_mismatches:,}"
)

print(
    f"Max raw difference: "
    f"{max_raw_difference:.10f}"
)

print()


if answer_mismatches == 0:
    print("PASS")
    print(
        "NumPy runtime matches the "
        "frozen PyTorch model."
    )

else:
    print("FAIL")
    print(
        "Do not package this release."
    )


print()