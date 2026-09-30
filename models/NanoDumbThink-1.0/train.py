import json
import random
from pathlib import Path

import torch
from torch import nn

from data import SCALE, build_samples, make_batch
from model import NanoDumbThink


ROOT = Path(__file__).parent

WEIGHTS = ROOT / "weights.pt"
CONFIG = ROOT / "config.json"

SEED = 7
EPOCHS = 600
BATCH_SIZE = 128

random.seed(SEED)
torch.manual_seed(SEED)


model = NanoDumbThink(
    hidden=32,
    middle=48,
    think_steps=8,
)

train_samples, test_samples = build_samples()


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.003,
    weight_decay=0.0002,
)

loss_fn = nn.SmoothL1Loss()


def exact_accuracy(samples):
    model.eval()

    correct = 0

    with torch.no_grad():
        for sample in samples:
            batch = make_batch([sample])

            value = model(
                batch["task"],
                batch["numbers"],
                batch["mask"],
            )[0].item()

            if sample.kind == "number":
                prediction = round(value * SCALE)
                expected = int(sample.answer)

            else:
                prediction = value > 0
                expected = bool(sample.answer)

            if prediction == expected:
                correct += 1

    return correct / len(samples) * 100


print()
print("Nano DumbThink 1.0")
print("-------------------")
print()
print(f"Parameters : {model.parameter_count():,}")
print(f"Train set  : {len(train_samples):,}")
print(f"Test set   : {len(test_samples):,}")
print(f"Think steps: {model.think_steps}")
print()
print("Training...")
print()


for epoch in range(1, EPOCHS + 1):
    model.train()

    random.shuffle(train_samples)

    total_loss = 0.0
    batches = 0

    for start in range(
        0,
        len(train_samples),
        BATCH_SIZE,
    ):
        samples = train_samples[
            start:start + BATCH_SIZE
        ]

        batch = make_batch(samples)

        output = model(
            batch["task"],
            batch["numbers"],
            batch["mask"],
        )

        loss = loss_fn(
            output,
            batch["target"],
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    if epoch == 350:
        for group in optimizer.param_groups:
            group["lr"] = 0.001

    if epoch == 500:
        for group in optimizer.param_groups:
            group["lr"] = 0.0004

    if epoch == 1 or epoch % 50 == 0:
        exact = exact_accuracy(
            train_samples
        )

        print(
            f"epoch {epoch:>3}  "
            f"loss {total_loss / batches:.6f}  "
            f"exact {exact:>6.2f}%"
        )


torch.save(
    model.state_dict(),
    WEIGHTS,
)


config = {
    "name": "Nano DumbThink",
    "version": "1.0",
    "hidden": model.hidden,
    "middle": model.middle,
    "think_steps": model.think_steps,
    "parameters": model.parameter_count(),
    "scale": SCALE,
    "input_min": 0,
    "input_max": 20,
}


with open(
    CONFIG,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        indent=2,
    )


raw_size = model.parameter_count() * 4


print()
print("Training complete.")
print()
print(f"Parameters    : {model.parameter_count():,}")
print(f"Raw brain size: {raw_size / 1024:.2f} KB")
print(f"Saved         : {WEIGHTS}")
print()