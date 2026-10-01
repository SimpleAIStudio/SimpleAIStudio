import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from model import Vex
from tokenizer import ByteTokenizer


root = Path(__file__).parent


benchmark_path = (
    root
    / "data"
    / "v3c"
    / "benchmark.json"
)


checkpoint_path = (
    root
    / "checkpoints"
    / "vex-v3c-best.pt"
)


results_path = (
    root
    / "v3c_benchmark_results.json"
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


tokenizer = ByteTokenizer()


model = Vex(
    vocab_size=256,
    context=256,
    hidden=96,
    heads=4,
    middle=384,
    layers=4,
    dropout=0.05,
).to(device)


checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)


model.load_state_dict(
    checkpoint["model"]
)

model.eval()


with open(
    benchmark_path,
    "r",
    encoding="utf-8",
) as file:
    benchmark = json.load(
        file
    )


def normalize(text):
    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = text.strip(
        " \t\r\n"
        ".,!?;:\"'()[]{}"
    )

    return text


@torch.no_grad()
def generate(prompt):
    text = (
        f"User: {prompt}\n"
        f"Vex:"
    )


    encoded = tokenizer.encode(
        text
    )


    encoded = encoded[
        -256:
    ]


    tokens = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device,
    )


    for _ in range(100):
        current = tokens[
            :,
            -256:
        ]


        logits, _ = model(
            current
        )


        next_token = torch.argmax(
            logits[
                :,
                -1,
                :
            ],
            dim=-1,
            keepdim=True,
        )


        tokens = torch.cat(
            (
                tokens,
                next_token,
            ),
            dim=1,
        )


        generated = tokenizer.decode(
            tokens[
                0,
                len(encoded):
            ].tolist()
        )


        if (
            "\nUser:"
            in generated
        ):
            break


        if (
            "\n\n"
            in generated
        ):
            break


    generated = tokenizer.decode(
        tokens[
            0,
            len(encoded):
        ].tolist()
    )


    for marker in [
        "\nUser:",
        "\nVex:",
        "\n\n",
    ]:
        if marker in generated:
            generated = generated.split(
                marker,
                1,
            )[0]


    return generated.strip()


def score_item(
    response,
    item,
):
    response_norm = normalize(
        response
    )


    expected = [
        normalize(value)
        for value in item[
            "expected"
        ]
    ]


    match = item[
        "match"
    ]


    if match == "exact":
        return any(
            response_norm == value
            for value in expected
        )


    if match == "contains_any":
        return any(
            value in response_norm
            for value in expected
        )


    if match == "contains_all":
        return all(
            value in response_norm
            for value in expected
        )


    raise ValueError(
        f"Unknown match rule: "
        f"{match}"
    )


results = []

stats = defaultdict(
    lambda: {
        "correct": 0,
        "total": 0,
    }
)


print()
print(
    "Vex Turbo DumbThink 3.0"
)

print(
    "V3-C benchmark"
)

print(
    "-----------------------"
)

print()

print(
    f"Checkpoint : "
    f"step {checkpoint['best_step']}"
)

print(
    f"Tests      : "
    f"{len(benchmark)}"
)

print(
    f"Device     : "
    f"{device}"
)

print()


for index, item in enumerate(
    benchmark,
    start=1,
):
    response = generate(
        item["prompt"]
    )


    correct = score_item(
        response,
        item,
    )


    category = item[
        "category"
    ]


    stats[
        category
    ][
        "total"
    ] += 1


    if correct:
        stats[
            category
        ][
            "correct"
        ] += 1


    results.append(
        {
            **item,
            "response":
                response,

            "correct":
                correct,
        }
    )


    if (
        index % 25 == 0
        or index == len(
            benchmark
        )
    ):
        print(
            f"tested "
            f"{index:>3}"
            f"/{len(benchmark)}"
        )


print()
print(
    "Results"
)

print(
    "-------"
)

print()


overall_correct = 0
overall_total = 0


for category in sorted(
    stats
):
    correct = stats[
        category
    ][
        "correct"
    ]

    total = stats[
        category
    ][
        "total"
    ]


    overall_correct += (
        correct
    )

    overall_total += (
        total
    )


    accuracy = (
        correct
        / total
        * 100
    )


    print(
        f"{category:<15}"
        f"{correct:>3}"
        f"/{total:<3}  "
        f"{accuracy:>6.2f}%"
    )


overall_accuracy = (
    overall_correct
    / overall_total
    * 100
)


print(
    "-" * 32
)


print(
    f"{'OVERALL':<15}"
    f"{overall_correct:>3}"
    f"/{overall_total:<3}  "
    f"{overall_accuracy:>6.2f}%"
)


output = {
    "model":
        "Vex Turbo DumbThink 3.0",

    "stage":
        "V3-C",

    "checkpoint_step":
        checkpoint[
            "best_step"
        ],

    "tests":
        overall_total,

    "correct":
        overall_correct,

    "accuracy":
        overall_accuracy,

    "categories":
        dict(stats),

    "results":
        results,
}


with open(
    results_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        output,
        file,
        indent=2,
    )


print()
print(
    f"Saved: "
    f"{results_path}"
)

print()