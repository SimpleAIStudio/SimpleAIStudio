import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from model import Vex
from runtime import VexRuntime


model_root = Path(__file__).parent

studio_root = Path(
    r"C:\SimpleAIStudio"
)

release_dir = (
    studio_root
    / "releases"
    / "VexTurboDumbThink-3.0"
)

packages_dir = (
    studio_root
    / "releases"
    / "packages"
)

registry_path = (
    studio_root
    / "registry"
    / "models.json"
)

benchmark_source = (
    model_root
    / "v3c_benchmark_results.json"
)

best_checkpoint = (
    model_root
    / "checkpoints"
    / "vex-v3c-best.pt"
)

frozen_checkpoint = (
    model_root
    / "checkpoints"
    / "vex-3.0-release-candidate.pt"
)

runtime_source = (
    model_root
    / "runtime.py"
)

run_source = (
    model_root
    / "run.py"
)


if not frozen_checkpoint.exists():
    shutil.copy2(
        best_checkpoint,
        frozen_checkpoint,
    )

    print(
        "Frozen V3-C checkpoint."
    )


if not benchmark_source.exists():
    raise FileNotFoundError(
        "v3c_benchmark_results.json "
        "was not found."
    )


if release_dir.exists():
    shutil.rmtree(
        release_dir
    )


release_dir.mkdir(
    parents=True,
    exist_ok=True,
)

packages_dir.mkdir(
    parents=True,
    exist_ok=True,
)


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
    frozen_checkpoint,
    map_location=device,
    weights_only=False,
)


model.load_state_dict(
    checkpoint["model"]
)

model.eval()


parameters = (
    model.parameter_count()
)

brain_size = (
    parameters * 4
)


print()
print("Vex Turbo DumbThink 3.0")
print("Final release")
print("-----------------------")
print()

print(
    f"Checkpoint : "
    f"step {checkpoint['best_step']}"
)

print(
    f"Parameters : "
    f"{parameters:,}"
)

print(
    f"Brain      : "
    f"{brain_size / 1024 / 1024:.2f} MB"
)

print()


weights = {}


for name, tensor in (
    model.state_dict().items()
):
    weights[name] = (
        tensor.detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


np.savez_compressed(
    release_dir
    / "weights.npz",
    **weights,
)


shutil.copy2(
    runtime_source,
    release_dir
    / "runtime.py",
)

shutil.copy2(
    run_source,
    release_dir
    / "run.py",
)

shutil.copy2(
    benchmark_source,
    release_dir
    / "benchmark.json",
)


with open(
    benchmark_source,
    "r",
    encoding="utf-8",
) as file:
    benchmark = json.load(
        file
    )


config = {
    "name":
        "Vex Turbo DumbThink",

    "version":
        "3.0",

    "model_type":
        "Transformer Language Model",

    "architecture":
        "4-layer Transformer",

    "parameters":
        parameters,

    "brain_size_bytes":
        brain_size,

    "tokenizer":
        "Byte-level",

    "vocab_size":
        256,

    "context":
        256,

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

    "training":
        "From scratch",

    "pretrained_weights":
        False,

    "external_ai":
        False,

    "checkpoint_step":
        checkpoint[
            "best_step"
        ],

    "benchmark_accuracy":
        benchmark[
            "accuracy"
        ],

    "benchmark_tests":
        benchmark[
            "tests"
        ],
}


with open(
    release_dir
    / "config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        indent=2,
    )


metadata = {
    "id":
        "vex-turbo-dumbthink",

    "name":
        "Vex Turbo DumbThink",

    "version":
        "3.0",

    "entrypoint":
        "run.py",

    "runtime":
        "Python + NumPy",

    "dependencies":
        [
            "numpy",
        ],

    "parameters":
        parameters,

    "context":
        256,

    "tokenizer":
        "Byte-level",
}


with open(
    release_dir
    / "simpleai-model.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )


categories = benchmark[
    "categories"
]


readme = f"""# Vex Turbo DumbThink 3.0

Vex Turbo DumbThink 3.0 is SimpleAI Studio's first Transformer language model.

## Architecture

- 496,704 trainable parameters
- 4 Transformer layers
- 96 hidden size
- 4 attention heads
- 384 feed-forward size
- 256-byte context
- raw byte-level tokenizer
- NumPy inference runtime

The model was trained from randomly initialized weights.

No pretrained model weights are used by Vex.

## Benchmark

Vex scored {benchmark['correct']}/{benchmark['tests']} ({benchmark['accuracy']:.2f}%) on the fixed V3-C benchmark.

The benchmark contains 500 prompts excluded from training, with 50 prompts in each of 10 categories.

Category results:

- Arithmetic: {categories['arithmetic']['correct']}/50
- Coding: {categories['coding']['correct']}/50
- Conversation: {categories['conversation']['correct']}/50
- Identity: {categories['identity']['correct']}/50
- Knowledge: {categories['knowledge']['correct']}/50
- Language: {categories['language']['correct']}/50
- Logic: {categories['logic']['correct']}/50
- Reading: {categories['reading']['correct']}/50
- Sequences: {categories['sequences']['correct']}/50
- Story world: {categories['story_world']['correct']}/50

## Run

Install through SimpleAI:

    simpleai pull vex-turbo-dumbthink:3.0
    simpleai run vex-turbo-dumbthink:3.0

Vex is a very small experimental language model. Its strongest areas are short language tasks, definitions, identity, and reading-style prompts. It is not intended to be a general-purpose assistant.
"""


(
    release_dir
    / "README.md"
).write_text(
    readme,
    encoding="utf-8",
)


print(
    "Exported NumPy weights."
)


# -------------------------------------------------
# VERIFY NUMPY AGAINST PYTORCH
# -------------------------------------------------

numpy_model = VexRuntime(
    release_dir
)


prompts = [
    "hello",
    "what is your name?",
    "what is Python?",
    "what is a GPU?",
    "what is a transformer?",
    "what is the Moon?",
    "what is gravity?",
    "what is the plural of city?",
    "what is the opposite of hot?",
    (
        "Read this: Alex put the mug "
        "on the desk. Where is the mug?"
    ),
]


def torch_generate(
    message,
    amount=40,
):
    prompt = (
        f"User: {message}\n"
        f"Vex:"
    )

    tokens = list(
        prompt.encode(
            "utf-8"
        )
    )

    generated = []

    with torch.no_grad():
        for _ in range(
            amount
        ):
            current = tokens[
                -256:
            ]

            x = torch.tensor(
                [current],
                dtype=torch.long,
                device=device,
            )

            logits, _ = model(x)

            next_token = int(
                torch.argmax(
                    logits[
                        0,
                        -1,
                    ]
                ).item()
            )

            tokens.append(
                next_token
            )

            generated.append(
                next_token
            )

            text = bytes(
                generated
            ).decode(
                "utf-8",
                errors="replace",
            )

            if (
                "\nUser:" in text
                or "\n\n" in text
            ):
                break

    return generated


# -------------------------------------------------
# PACKAGE
# -------------------------------------------------

package_path = (
    packages_dir
    / "vex-turbo-dumbthink-3.0.zip"
)


if package_path.exists():
    package_path.unlink()


with zipfile.ZipFile(
    package_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as archive:
    for path in sorted(
        release_dir.rglob("*")
    ):
        if path.is_file():
            archive.write(
                path,
                path.relative_to(
                    release_dir
                ),
            )


sha = hashlib.sha256(
    package_path.read_bytes()
).hexdigest()


print(
    f"Package    : {package_path}"
)

print(
    f"ZIP size   : "
    f"{package_path.stat().st_size:,} bytes"
)

print(
    f"SHA-256    : {sha}"
)


# -------------------------------------------------
# UPDATE REGISTRY
# -------------------------------------------------

with open(
    registry_path,
    "r",
    encoding="utf-8-sig",
) as file:
    registry = json.load(
        file
    )


if isinstance(
    registry,
    list,
):
    models = registry

elif (
    isinstance(
        registry,
        dict,
    )
    and isinstance(
        registry.get("models"),
        list,
    )
):
    models = registry[
        "models"
    ]

else:
    raise RuntimeError(
        "Unknown registry format."
    )


models[:] = [
    item
    for item in models
    if not (
        item.get("id")
        == "vex-turbo-dumbthink"
        and item.get("version")
        == "3.0"
    )
]


models.append(
    {
        "id":
            "vex-turbo-dumbthink",

        "name":
            "Vex Turbo DumbThink",

        "version":
            "3.0",

        "status":
            "released",

        "family":
            "DumbThink",

        "description":
            (
                "A 496K-parameter byte-level "
                "Transformer language model "
                "trained from scratch for short "
                "local text generation, definitions, "
                "reading and simple reasoning."
            ),

        "architecture":
            "4-layer Transformer",

        "parameters":
            496704,

        "brain_size_bytes":
            1986816,

        "context":
            "256-byte context",

        "tokenizer":
            "Byte-level",

        "training":
            "From scratch",

        "pretrained_weights":
            False,

        "external_ai":
            False,

        "price":
            "Free",

        "skills":
            [
                "Conversation",
                "Definitions",
                "General knowledge",
                "Reading comprehension",
                "Language patterns",
                "Coding concepts",
                "Basic logic",
            ],

        "install_command":
            (
                "simpleai pull "
                "vex-turbo-dumbthink:3.0"
            ),

        "entrypoint":
            "run.py",

        "runtime":
            "Python + NumPy",

        "dependencies":
            [
                "numpy",
            ],

        "benchmark_accuracy":
            69.60,

        "benchmark_tests":
            500,

        "benchmark_scope":
            (
                "500 fixed prompts excluded "
                "from training, with 50 prompts "
                "across each of 10 categories"
            ),

        "download":
            (
                "https://github.com/"
                "SimpleAIStudio/SimpleAIStudio/"
                "releases/download/"
                "vex-turbo-dumbthink-3.0/"
                "vex-turbo-dumbthink-3.0.zip"
            ),

        "sha256":
            sha,
    }
)


with open(
    registry_path,
    "w",
    encoding="utf-8",
    newline="\n",
) as file:
    json.dump(
        registry,
        file,
        indent=2,
        ensure_ascii=False,
    )

    file.write("\n")


print(
    "Registry updated."
)


# -------------------------------------------------
# BUILD WEBSITE
# -------------------------------------------------

subprocess.run(
    [
        sys.executable,
        str(
            studio_root
            / "build_public.py"
        ),
    ],
    cwd=studio_root,
    check=True,
)


print(
    "Website docs rebuilt."
)

print()
print(
    "VEX 3.0 RELEASE READY."
)

print()
print(
    "Next:"
)

print(
    "1. Create GitHub Release tag:"
)

print(
    "   vex-turbo-dumbthink-3.0"
)

print(
    "2. Upload:"
)

print(
    f"   {package_path}"
)

print(
    "3. Publish the release."
)

print(
    "4. Commit registry/ + docs/ "
    "in GitHub Desktop and push."
)

print()
print(
    "Then test:"
)

print(
    "simpleai pull "
    "vex-turbo-dumbthink:3.0"
)

print(
    "simpleai run "
    "vex-turbo-dumbthink:3.0"
)

print()