import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn

from data import build_samples, decode, make_batch
from model import MicroJet


root = Path(__file__).parent

weights_path = root / "weights.pt"
config_path = root / "config.json"

seed = 22
epochs = 700
batch_size = 256
samples_per_task = 500

random.seed(seed)
torch.manual_seed(seed)


model = MicroJet(
    hidden=64,
    middle=96,
    think_steps=12,
)

train_samples, test_samples = build_samples()


groups = defaultdict(list)

for sample in train_samples:
    groups[sample.task].append(sample)


counts = Counter(
    sample.task
    for sample in train_samples
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0015,
    weight_decay=0.0001,
)

loss_fn = nn.SmoothL1Loss(
    reduction="none",
    beta=0.05,
)


def make_epoch():
    epoch_samples = []

    for task_samples in groups.values():
        if len(task_samples) >= samples_per_task:
            chosen = random.sample(
                task_samples,
                samples_per_task,
            )
        else:
            chosen = random.choices(
                task_samples,
                k=samples_per_task,
            )

        epoch_samples.extend(chosen)

    random.shuffle(epoch_samples)

    return epoch_samples


def exact_accuracy(samples):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(
            0,
            len(samples),
            512,
        ):
            chunk = samples[
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

                if prediction == expected:
                    correct += 1

                total += 1

    return correct / total * 100


print()
print("Micro Jet DumbThink 2.0")
print("-----------------------")
print()

print(
    f"Parameters : "
    f"{model.parameter_count():,}"
)

print(
    f"Train set  : "
    f"{len(train_samples):,}"
)

print(
    f"Test set   : "
    f"{len(test_samples):,}"
)

print(
    f"Think steps: "
    f"{model.think_steps}"
)

print()
print("Tasks:")

for task, count in counts.items():
    print(
        f"  {task:<10} "
        f"{count:>5}"
    )

print()
print(
    f"Balanced epoch: "
    f"{samples_per_task:,} samples "
    f"per task"
)

print()
print("Training...")
print()


best_accuracy = 0.0
best_epoch = 0
best_state = None


for epoch in range(
    1,
    epochs + 1,
):
    model.train()

    epoch_samples = make_epoch()

    total_loss = 0.0
    batches = 0

    for start in range(
        0,
        len(epoch_samples),
        batch_size,
    ):
        samples = epoch_samples[
            start:start + batch_size
        ]

        batch = make_batch(samples)

        output = model(
            batch["task"],
            batch["numbers"],
            batch["mask"],
        )

        losses = loss_fn(
            output,
            batch["target"],
        )

        loss = losses.mean()

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    if epoch == 300:
        for group in optimizer.param_groups:
            group["lr"] = 0.0006

    if epoch == 500:
        for group in optimizer.param_groups:
            group["lr"] = 0.0002

    if (
        epoch == 1
        or epoch % 25 == 0
    ):
        accuracy = exact_accuracy(
            train_samples
        )

        average_loss = (
            total_loss
            / batches
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"epoch {epoch:>3}  "
            f"loss {average_loss:.6f}  "
            f"exact {accuracy:>6.2f}%  "
            f"best {best_accuracy:>6.2f}%"
        )


if best_state is not None:
    model.load_state_dict(
        best_state
    )


torch.save(
    model.state_dict(),
    weights_path,
)


config = {
    "name":
        "Micro Jet DumbThink",

    "version":
        "2.0",

    "hidden":
        model.hidden,

    "middle":
        model.middle,

    "think_steps":
        model.think_steps,

    "parameters":
        model.parameter_count(),

    "tasks":
        list(counts.keys()),

    "training_samples":
        len(train_samples),

    "test_samples":
        len(test_samples),

    "best_epoch":
        best_epoch,

    "best_training_accuracy":
        round(
            best_accuracy,
            2,
        ),
}


with open(
    config_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        indent=2,
    )


raw_size = (
    model.parameter_count()
    * 4
)


print()
print("Training complete.")
print()

print(
    f"Best epoch    : "
    f"{best_epoch}"
)

print(
    f"Best train acc: "
    f"{best_accuracy:.2f}%"
)

print(
    f"Parameters    : "
    f"{model.parameter_count():,}"
)

print(
    f"Raw brain size: "
    f"{raw_size / 1024:.2f} KB"
)

print(
    f"Saved         : "
    f"{weights_path}"
)

print()