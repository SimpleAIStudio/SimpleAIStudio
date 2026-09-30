import hashlib
import json
import shutil
import zipfile
from pathlib import Path


root = Path(__file__).parent

studio = root.parent.parent

release = (
    studio
    / "releases"
    / "MicroJetDumbThink-2.0"
)

packages = (
    studio
    / "releases"
    / "packages"
)

packages.mkdir(
    parents=True,
    exist_ok=True,
)

zip_path = (
    packages
    / "micro-jet-dumbthink-2.0.zip"
)


readme = """# Micro Jet DumbThink 2.0

SimpleAI's Studio — Model #002

Micro Jet DumbThink 2.0 is a tiny recursive reasoning model trained
completely from random initialization.

## Model

- Parameters: 14,818
- Think steps: 12
- Raw learned weights: 57.88 KB
- Architecture: Recursive Neural Network
- Runtime: Python + NumPy
- Pretrained weights: None
- External AI APIs: None

## Skills

- Addition
- Subtraction
- Multiplication
- Exact integer division
- Greater-than comparison
- Less-than comparison
- Arithmetic sequences
- Three-number addition
- Maximum of three values
- Chained greater-than comparison

## Development benchmark

970 / 971 — 99.90%

This benchmark contains held-out combinations inside the model's
defined training domains.

## Generalization challenge

235 / 300 — 78.33%

These problems intentionally use values outside the ranges used
during training.

This result shows that some learned behaviours generalize strongly,
while others, especially multiplication, remain range-dependent.

## Run

Install using the SimpleAI CLI:

simpleai pull micro-jet-dumbthink:2.0

Then run:

simpleai run micro-jet-dumbthink:2.0
"""


with open(
    release / "README.md",
    "w",
    encoding="utf-8",
) as file:
    file.write(readme)


manifest_path = (
    release
    / "simpleai-model.json"
)

with open(
    manifest_path,
    "r",
    encoding="utf-8",
) as file:
    manifest = json.load(file)


manifest.update(
    {
        "architecture":
            "Recursive Neural Network",

        "brain_size_bytes":
            14818 * 4,

        "training":
            "From scratch",

        "pretrained_weights":
            False,

        "external_ai":
            False,

        "skills": [
            "Addition",
            "Subtraction",
            "Multiplication",
            "Exact integer division",
            "Greater-than comparison",
            "Less-than comparison",
            "Arithmetic sequences",
            "Three-number addition",
            "Maximum of three values",
            "Chained greater-than comparison",
        ],

        "benchmark": {
            "development": {
                "correct": 970,
                "total": 971,
                "accuracy": 99.90,
            },

            "generalization": {
                "correct": 235,
                "total": 300,
                "accuracy": 78.33,
            },
        },
    }
)


with open(
    manifest_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )


if zip_path.exists():
    zip_path.unlink()


with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as archive:

    for path in sorted(
        release.rglob("*")
    ):
        if not path.is_file():
            continue

        archive.write(
            path,
            path.relative_to(
                release
            ),
        )


sha256 = hashlib.sha256()

with open(
    zip_path,
    "rb",
) as file:
    while True:
        chunk = file.read(
            1024 * 1024
        )

        if not chunk:
            break

        sha256.update(chunk)


digest = sha256.hexdigest()

size = zip_path.stat().st_size


print()
print("Micro Jet DumbThink 2.0")
print("Release package")
print("-----------------------")
print()

print(
    f"Parameters : 14,818"
)

print(
    f"Raw brain  : 57.88 KB"
)

print(
    f"Dev score  : 970/971 "
    f"(99.90%)"
)

print(
    f"Gen score  : 235/300 "
    f"(78.33%)"
)

print()

print(
    f"ZIP size   : "
    f"{size:,} bytes"
)

print(
    f"Package    : "
    f"{zip_path}"
)

print()

print(
    f"SHA-256:"
)

print(
    digest
)

print()