import json
import shutil
from pathlib import Path

import numpy as np
import torch

from model import NanoDumbThink


ROOT = Path(__file__).parent

STUDIO = ROOT.parents[1]

RELEASE = (
    STUDIO
    / "releases"
    / "NanoDumbThink-1.0"
)


with open(
    ROOT / "config.json",
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)


model = NanoDumbThink(
    hidden=config["hidden"],
    middle=config["middle"],
    think_steps=config["think_steps"],
)

model.load_state_dict(
    torch.load(
        ROOT / "weights.pt",
        map_location="cpu",
    )
)

model.eval()

state = model.state_dict()


if RELEASE.exists():
    shutil.rmtree(RELEASE)

RELEASE.mkdir(
    parents=True,
    exist_ok=True,
)


heads_weight = np.stack(
    [
        state[f"heads.{i}.weight"]
        .cpu()
        .numpy()
        .squeeze(0)
        for i in range(5)
    ]
)

heads_bias = np.stack(
    [
        state[f"heads.{i}.bias"]
        .cpu()
        .numpy()
        .squeeze(0)
        for i in range(5)
    ]
)


np.savez(
    RELEASE / "weights.npz",

    task_embed=(
        state["task_embed.weight"]
        .cpu()
        .numpy()
    ),

    input_weight=(
        state["input_layer.weight"]
        .cpu()
        .numpy()
    ),

    input_bias=(
        state["input_layer.bias"]
        .cpu()
        .numpy()
    ),

    brain1_weight=(
        state["brain.0.weight"]
        .cpu()
        .numpy()
    ),

    brain1_bias=(
        state["brain.0.bias"]
        .cpu()
        .numpy()
    ),

    brain2_weight=(
        state["brain.2.weight"]
        .cpu()
        .numpy()
    ),

    brain2_bias=(
        state["brain.2.bias"]
        .cpu()
        .numpy()
    ),

    norm_weight=(
        state["norm.weight"]
        .cpu()
        .numpy()
    ),

    norm_bias=(
        state["norm.bias"]
        .cpu()
        .numpy()
    ),

    head_weight=heads_weight,
    head_bias=heads_bias,
)


release_config = {
    "name": "Nano DumbThink",
    "version": "1.0",

    "parameters": config["parameters"],

    "architecture": "Recursive neural network",

    "hidden": config["hidden"],
    "middle": config["middle"],
    "think_steps": config["think_steps"],

    "scale": config["scale"],

    "input_min": 0,
    "input_max": 20,

    "runtime": "numpy",
}


with open(
    RELEASE / "config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        release_config,
        file,
        indent=2,
    )


manifest = {
    "id": "nano-dumbthink",

    "name": "Nano DumbThink",

    "version": "1.0",

    "family": "DumbThink",

    "entrypoint": "run.py",

    "parameters": config["parameters"],

    "architecture": "Recursive Neural Network",

    "runtime": "Python + NumPy",

    "training": "From scratch",

    "pretrained_weights": False,

    "external_ai": False,

    "input_range": "0-20",

    "skills": [
        "Addition",
        "Subtraction",
        "Greater-than comparison",
        "Less-than comparison",
        "Arithmetic sequences",
    ],
}


with open(
    RELEASE / "simpleai-model.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
    )


shutil.copy2(
    ROOT / "runtime.py",
    RELEASE / "runtime.py",
)

shutil.copy2(
    ROOT / "release_run.py",
    RELEASE / "run.py",
)

shutil.copy2(
    ROOT / "benchmark.json",
    RELEASE / "benchmark.json",
)


print()
print("Nano DumbThink 1.0 exported.")
print()
print(f"Release: {RELEASE}")
print()

for file in sorted(RELEASE.iterdir()):
    print(
        f"  {file.name:<22} "
        f"{file.stat().st_size:,} bytes"
    )

print()