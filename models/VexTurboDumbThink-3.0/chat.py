import json
from pathlib import Path

import torch

from model import Vex
from tokenizer import ByteTokenizer


root = Path(__file__).parent

checkpoint_path = (
    root
    / "checkpoints"
    / "vex-v3c-best.pt"
)

config_path = (
    root
    / "config.json"
)


with open(
    config_path,
    "r",
    encoding="utf-8-sig",
) as file:
    config = json.load(file)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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


tokenizer = ByteTokenizer()

history = ""
history_enabled = False
sampling_enabled = False


def generate_response(message):
    global history

    if history_enabled:
        prompt = (
            history
            + f"User: {message}\n"
            + "Vex:"
        )

    else:
        prompt = (
            f"User: {message}\n"
            f"Vex:"
        )


    encoded = tokenizer.encode(
        prompt
    )

    encoded = encoded[-256:]


    tokens = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=device,
    )


    for _ in range(120):
        current = tokens[
            :,
            -256:
        ]

        logits, _ = model(
            current
        )

        logits = logits[
            :,
            -1,
            :
        ]


        if sampling_enabled:
            logits = (
                logits / 0.65
            )

            values, _ = torch.topk(
                logits,
                20,
            )

            cutoff = values[
                :,
                -1:
            ]

            logits = logits.masked_fill(
                logits < cutoff,
                float("-inf"),
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=-1,
                )
            )

            next_token = (
                torch.multinomial(
                    probabilities,
                    1,
                )
            )

        else:
            next_token = (
                torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )
            )


        tokens = torch.cat(
            (
                tokens,
                next_token,
            ),
            dim=1,
        )


        generated = (
            tokenizer.decode(
                tokens[
                    0,
                    len(encoded):
                ].tolist()
            )
        )


        if (
            "\nUser:"
            in generated
            or "\n\n"
            in generated
        ):
            break


    response = tokenizer.decode(
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
        if marker in response:
            response = (
                response.split(
                    marker,
                    1,
                )[0]
            )


    response = response.strip()


    if history_enabled:
        history += (
            f"User: {message}\n"
            f"Vex: {response}\n"
        )

        history = history[-450:]


    return response


print()
print("Vex Turbo DumbThink 3.0")
print("V3-C")
print("-----------------------")
print()

print(
    f"Device     : {device}"
)

if device == "cuda":
    print(
        f"GPU        : "
        f"{torch.cuda.get_device_name(0)}"
    )

print(
    f"Parameters : "
    f"{model.parameter_count():,}"
)

print(
    f"Context    : 256 bytes"
)

print(
    f"Best step  : "
    f"{checkpoint['best_step']}"
)

print(
    f"Best macro : "
    f"{checkpoint['best_loss']:.4f}"
)

print()
print("Decode     : GREEDY")
print("History    : OFF")
print()

print("/sample   toggle sampling")
print("/history  toggle history")
print("/clear    clear history")
print("/exit     quit")
print()


while True:
    try:
        message = input(
            "You > "
        ).strip()

    except (
        KeyboardInterrupt,
        EOFError,
    ):
        print()
        break


    if not message:
        continue


    command = message.lower()


    if command in {
        "/exit",
        "/quit",
    }:
        break


    if command == "/clear":
        history = ""

        print(
            "Vex > history cleared."
        )

        print()
        continue


    if command == "/history":
        history_enabled = (
            not history_enabled
        )

        history = ""

        state = (
            "ON"
            if history_enabled
            else "OFF"
        )

        print(
            f"Vex > history {state}."
        )

        print()
        continue


    if command == "/sample":
        sampling_enabled = (
            not sampling_enabled
        )

        state = (
            "SAMPLING"
            if sampling_enabled
            else "GREEDY"
        )

        print(
            f"Vex > decode mode: {state}"
        )

        print()
        continue


    response = generate_response(
        message
    )

    print(
        f"Vex > {response}"
    )

    print()