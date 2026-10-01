import json
import math
import random
import time
from pathlib import Path

import torch

from model import Vex


root = Path(__file__).parent

data_root = (
    root
    / "data"
    / "v3c"
)

train_dir = (
    data_root
    / "train"
)

valid_dir = (
    data_root
    / "valid"
)

checkpoint_dir = (
    root
    / "checkpoints"
)

checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True,
)


best_path = (
    checkpoint_dir
    / "vex-v3c-best.pt"
)

latest_path = (
    checkpoint_dir
    / "vex-v3c-latest.pt"
)

config_path = (
    root
    / "config.json"
)


categories = [
    "conversation",
    "identity",
    "knowledge",
    "reading",
    "language",
    "arithmetic",
    "logic",
    "sequences",
    "coding",
    "story_world",
]


seed = 303

context = 256

sequences_per_category = 12

batch_size = (
    sequences_per_category
    * len(categories)
)

max_steps = 3500

eval_every = 100
eval_batches_per_category = 4
eval_batch_size = 24

warmup_steps = 150

max_lr = 0.0006
min_lr = 0.00004

patience_evals = 10
min_stop_step = 1200


random.seed(seed)
torch.manual_seed(seed)


if torch.cuda.is_available():
    device = "cuda"

    torch.cuda.manual_seed_all(
        seed
    )

else:
    device = "cpu"


def load_bytes(path):
    data = path.read_bytes()

    return torch.tensor(
        list(data),
        dtype=torch.long,
        device=device,
    )


train_data = {}
valid_data = {}


for category in categories:
    train_data[
        category
    ] = load_bytes(
        train_dir
        / f"{category}.txt"
    )

    valid_data[
        category
    ] = load_bytes(
        valid_dir
        / f"{category}.txt"
    )


model = Vex(
    vocab_size=256,
    context=context,
    hidden=96,
    heads=4,
    middle=384,
    layers=4,
    dropout=0.05,
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)


positions = torch.arange(
    context,
    device=device,
)


def sample_from_data(
    data,
    count,
):
    max_start = (
        len(data)
        - context
        - 1
    )

    starts = torch.randint(
        0,
        max_start,
        (count,),
        device=device,
    )

    indexes = (
        starts[:, None]
        + positions[None, :]
    )

    x = data[
        indexes
    ]

    y = data[
        indexes + 1
    ]

    return x, y


def get_balanced_batch():
    x_parts = []
    y_parts = []


    for category in categories:
        x, y = sample_from_data(
            train_data[
                category
            ],
            sequences_per_category,
        )

        x_parts.append(x)
        y_parts.append(y)


    x = torch.cat(
        x_parts,
        dim=0,
    )

    y = torch.cat(
        y_parts,
        dim=0,
    )


    order = torch.randperm(
        x.size(0),
        device=device,
    )

    return (
        x[order],
        y[order],
    )


def learning_rate(step):
    if step <= warmup_steps:
        return (
            max_lr
            * step
            / warmup_steps
        )


    progress = (
        step
        - warmup_steps
    ) / (
        max_steps
        - warmup_steps
    )

    progress = min(
        max(
            progress,
            0.0,
        ),
        1.0,
    )


    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )


    return (
        min_lr
        + cosine
        * (
            max_lr
            - min_lr
        )
    )


@torch.no_grad()
def evaluate_category(
    category,
):
    data = valid_data[
        category
    ]

    losses = []


    for _ in range(
        eval_batches_per_category
    ):
        x, y = sample_from_data(
            data,
            eval_batch_size,
        )

        _, loss = model(
            x,
            y,
        )

        losses.append(
            loss.item()
        )


    return (
        sum(losses)
        / len(losses)
    )


@torch.no_grad()
def evaluate():
    model.eval()

    results = {}


    for category in categories:
        results[
            category
        ] = evaluate_category(
            category
        )


    macro_loss = (
        sum(
            results.values()
        )
        / len(results)
    )


    model.train()


    return (
        macro_loss,
        results,
    )


def save_checkpoint(
    path,
    step,
    best_loss,
    best_step,
    category_losses,
):
    torch.save(
        {
            "model":
                model.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "step":
                step,

            "best_loss":
                best_loss,

            "best_step":
                best_step,

            "category_losses":
                category_losses,
        },
        path,
    )


def save_config(
    step,
    best_loss,
    best_step,
    status,
):
    train_bytes = sum(
        len(data)
        for data in train_data.values()
    )

    valid_bytes = sum(
        len(data)
        for data in valid_data.values()
    )


    config = {
        "name":
            "Vex Turbo DumbThink",

        "version":
            "3.0",

        "training_stage":
            "V3-C balanced development",

        "status":
            status,

        "model_type":
            "Transformer Language Model",

        "tokenizer":
            "Byte-level",

        "vocab_size":
            256,

        "context":
            context,

        "hidden":
            96,

        "heads":
            4,

        "head_size":
            24,

        "middle":
            384,

        "layers":
            4,

        "dropout":
            0.05,

        "parameters":
            model.parameter_count(),

        "raw_brain_bytes":
            model.parameter_count()
            * 4,

        "categories":
            categories,

        "balanced_training":
            True,

        "sequences_per_category":
            sequences_per_category,

        "batch_size":
            batch_size,

        "train_bytes":
            train_bytes,

        "validation_bytes":
            valid_bytes,

        "current_step":
            step,

        "best_step":
            best_step,

        "best_validation_loss":
            (
                None
                if math.isinf(
                    best_loss
                )
                else best_loss
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


start_step = 0

best_loss = float(
    "inf"
)

best_step = 0

bad_evals = 0


if latest_path.exists():
    print()
    print(
        "Found V3-C checkpoint."
    )

    print(
        "Resuming training..."
    )


    checkpoint = torch.load(
        latest_path,
        map_location=device,
        weights_only=False,
    )


    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )


    optimizer.load_state_dict(
        checkpoint[
            "optimizer"
        ]
    )


    start_step = int(
        checkpoint[
            "step"
        ]
    )


    best_loss = float(
        checkpoint[
            "best_loss"
        ]
    )


    best_step = int(
        checkpoint[
            "best_step"
        ]
    )


print()
print(
    "Vex Turbo DumbThink 3.0"
)

print(
    "V3-C balanced training"
)

print(
    "-----------------------"
)

print()


print(
    f"Device       : "
    f"{device}"
)


if device == "cuda":
    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )


print(
    f"Parameters   : "
    f"{model.parameter_count():,}"
)


print(
    f"Raw brain    : "
    f"{model.parameter_count() * 4 / 1024 / 1024:.2f} MB"
)


print(
    f"Categories   : "
    f"{len(categories)}"
)


print(
    f"Per category : "
    f"{sequences_per_category}"
)


print(
    f"Batch        : "
    f"{batch_size}"
)


print(
    f"Context      : "
    f"{context}"
)


print(
    f"Max steps    : "
    f"{max_steps:,}"
)


if start_step:
    print(
        f"Resume step  : "
        f"{start_step:,}"
    )


print()
print(
    "Every training batch is "
    "category-balanced."
)

print()
print(
    "Training..."
)

print()


started = time.perf_counter()

recent_started = started

recent_step = start_step


step = start_step


try:
    for step in range(
        start_step + 1,
        max_steps + 1,
    ):
        model.train()


        lr = learning_rate(
            step
        )


        for group in (
            optimizer.param_groups
        ):
            group[
                "lr"
            ] = lr


        x, y = (
            get_balanced_batch()
        )


        _, loss = model(
            x,
            y,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )


        optimizer.step()


        if (
            step == 1
            or step % eval_every == 0
        ):
            if device == "cuda":
                torch.cuda.synchronize()


            valid_loss, category_losses = (
                evaluate()
            )


            perplexity = math.exp(
                min(
                    valid_loss,
                    20,
                )
            )


            now = time.perf_counter()


            elapsed = (
                now
                - recent_started
            )


            steps_done = (
                step
                - recent_step
            )


            speed = (
                steps_done
                / elapsed
                if elapsed > 0
                else 0
            )


            remaining = (
                max_steps
                - step
            )


            eta = (
                remaining
                / speed
                if speed > 0
                else 0
            )


            improved = (
                valid_loss
                < best_loss
            )


            if improved:
                best_loss = (
                    valid_loss
                )

                best_step = step

                bad_evals = 0


                save_checkpoint(
                    best_path,
                    step,
                    best_loss,
                    best_step,
                    category_losses,
                )

            else:
                bad_evals += 1


            save_checkpoint(
                latest_path,
                step,
                best_loss,
                best_step,
                category_losses,
            )


            save_config(
                step,
                best_loss,
                best_step,
                "training",
            )


            print(
                f"step {step:>4}  "
                f"train {loss.item():.4f}  "
                f"macro {valid_loss:.4f}  "
                f"ppl {perplexity:.2f}  "
                f"lr {lr:.6f}  "
                f"{speed:.2f} step/s  "
                f"eta {eta / 60:.1f}m"
            )


            print(
                "           "
                f"conv {category_losses['conversation']:.3f}  "
                f"id {category_losses['identity']:.3f}  "
                f"know {category_losses['knowledge']:.3f}  "
                f"read {category_losses['reading']:.3f}  "
                f"lang {category_losses['language']:.3f}"
            )


            print(
                "           "
                f"math {category_losses['arithmetic']:.3f}  "
                f"logic {category_losses['logic']:.3f}  "
                f"seq {category_losses['sequences']:.3f}  "
                f"code {category_losses['coding']:.3f}  "
                f"story {category_losses['story_world']:.3f}"
            )


            if improved:
                print(
                    "           "
                    "new best checkpoint"
                )


            recent_started = now

            recent_step = step


            if (
                step >= min_stop_step
                and bad_evals
                >= patience_evals
            ):
                print()
                print(
                    "Macro validation "
                    "stopped improving."
                )

                print(
                    "Early stopping."
                )

                break


except KeyboardInterrupt:
    print()
    print(
        "Training interrupted."
    )


    save_checkpoint(
        latest_path,
        step,
        best_loss,
        best_step,
        {},
    )


    save_config(
        step,
        best_loss,
        best_step,
        "interrupted",
    )


    print(
        "Progress saved."
    )

    print(
        "Run train.py again "
        "to resume."
    )

    raise SystemExit


save_config(
    step,
    best_loss,
    best_step,
    "complete",
)


total_time = (
    time.perf_counter()
    - started
)


print()
print(
    "Training finished."
)

print()


print(
    f"Best step   : "
    f"{best_step}"
)


print(
    f"Best macro  : "
    f"{best_loss:.4f}"
)


print(
    f"Best ppl    : "
    f"{math.exp(min(best_loss, 20)):.2f}"
)


print(
    f"Run time    : "
    f"{total_time / 60:.2f} minutes"
)


print(
    f"Best model  : "
    f"{best_path}"
)

print()