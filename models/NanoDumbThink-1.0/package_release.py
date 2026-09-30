import hashlib
import json
import zipfile
from pathlib import Path


root = Path(__file__).parent
studio = root.parents[1]

release = studio / "releases" / "NanoDumbThink-1.0"
packages = studio / "releases" / "packages"

zip_path = packages / "nano-dumbthink-1.0.zip"
registry_path = studio / "registry" / "models.json"


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


if not release.exists():
    raise SystemExit(
        "Run export_release.py first."
    )


with open(
    release / "config.json",
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)


with open(
    release / "benchmark.json",
    "r",
    encoding="utf-8",
) as file:
    benchmark = json.load(file)


manifest = {
    "id": "nano-dumbthink",
    "name": "Nano DumbThink",
    "version": "1.0",
    "family": "DumbThink",

    "entrypoint": "run.py",

    "parameters": config["parameters"],
    "brain_size_bytes": config["parameters"] * 4,

    "architecture": "Recursive Neural Network",
    "runtime": "Python + NumPy",

    "training": "From scratch",
    "pretrained_weights": False,
    "external_ai": False,

    "dependencies": [
        "numpy"
    ],

    "input_min": 0,
    "input_max": 20,

    "skills": [
        "Addition",
        "Subtraction",
        "Greater-than comparison",
        "Less-than comparison",
        "Arithmetic sequences"
    ],

    "benchmark": {
        "tests": benchmark["test_samples"],
        "accuracy": benchmark["overall_accuracy"],
        "scope": "Held-out problems inside the 0-20 task domain"
    }
}


with open(
    release / "simpleai-model.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )


readme = f"""# Nano DumbThink 1.0

Nano DumbThink is a tiny recursive reasoning model made by SimpleAI's Studio.

It was trained from random weights without a pretrained AI model or external AI API.

## Model

Parameters: {config["parameters"]:,}
Thinking steps: {config["think_steps"]}
Raw learned weights: {config["parameters"] * 4:,} bytes
Runtime: Python + NumPy

## Skills

- Addition
- Subtraction
- Greater-than comparison
- Less-than comparison
- Simple arithmetic sequences

## Input domain

Numbers from 0 to 20.

## Benchmark

Held-out test problems: {benchmark["test_samples"]}
Accuracy: {benchmark["overall_accuracy"]:.2f}%

The benchmark measures held-out problems inside the model's defined task domain.
It is not a claim of general reasoning accuracy.

## Run

Using SimpleAI:

    simpleai run nano-dumbthink:1.0

Or directly:

    py run.py
"""


with open(
    release / "README.md",
    "w",
    encoding="utf-8",
) as file:
    file.write(readme)


packages.mkdir(
    parents=True,
    exist_ok=True,
)


if zip_path.exists():
    zip_path.unlink()


with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as archive:

    for path in sorted(release.rglob("*")):
        if path.is_file():
            archive.write(
                path,
                path.relative_to(release),
            )


package_hash = sha256(zip_path)


with open(
    registry_path,
    "r",
    encoding="utf-8-sig",
) as file:
    registry = json.load(file)


for model in registry["models"]:
    if (
        model.get("id") == "nano-dumbthink"
        and model.get("version") == "1.0"
    ):
        model.update(
            {
                "status": "released",

                "description":
                    "A tiny recursive reasoning model trained completely from scratch.",

                "parameters":
                    config["parameters"],

                "brain_size_bytes":
                    config["parameters"] * 4,

                "architecture":
                    "Recursive Neural Network",

                "runtime":
                    "Python + NumPy",

                "think_steps":
                    config["think_steps"],

                "training":
                    "From scratch",

                "pretrained_weights":
                    False,

                "external_ai":
                    False,

                "price":
                    "Free",

                "skills": manifest["skills"],

                "benchmark_accuracy":
                    benchmark["overall_accuracy"],

                "benchmark_tests":
                    benchmark["test_samples"],

                "benchmark_scope":
                    "Held-out problems inside the 0-20 task domain",

                "dependencies": [
                    "numpy"
                ],

                "entrypoint":
                    "run.py",

                "install_command":
                    "simpleai pull nano-dumbthink:1.0",

                "download":
                    zip_path.resolve().as_uri(),

                "sha256":
                    package_hash,
            }
        )

        break

else:
    raise SystemExit(
        "nano-dumbthink:1.0 is missing from models.json"
    )


with open(
    registry_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        registry,
        file,
        indent=4,
    )


print()
print("Nano DumbThink 1.0")
print("-------------------")
print()
print(f"Parameters : {config['parameters']:,}")
print(f"Package    : {zip_path}")
print(f"ZIP size   : {zip_path.stat().st_size:,} bytes")
print(f"Benchmark  : {benchmark['overall_accuracy']:.2f}%")
print()
print("SHA-256")
print(package_hash)
print()
print("Registry updated.")
print()