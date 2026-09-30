import json
import shutil
from pathlib import Path

import numpy as np
import torch

from model import MicroJet


root = Path(__file__).parent

studio = root.parent.parent

release = (
    studio
    / "releases"
    / "MicroJetDumbThink-2.0"
)

release.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    root / "config.json",
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)


model = MicroJet(
    hidden=config["hidden"],
    middle=config["middle"],
    think_steps=config["think_steps"],
)

state = torch.load(
    root / "weights_release_candidate.pt",
    map_location="cpu",
)

model.load_state_dict(state)
model.eval()


def array(name):
    return (
        state[name]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


head_w = np.stack(
    [
        array(
            f"heads.{i}.weight"
        )[0]
        for i in range(10)
    ]
)

head_b = np.array(
    [
        array(
            f"heads.{i}.bias"
        )[0]
        for i in range(10)
    ],
    dtype=np.float32,
)


np.savez_compressed(
    release / "weights.npz",

    task_embed=array(
        "task_embed.weight"
    ),

    input_w=array(
        "input_layer.weight"
    ),

    input_b=array(
        "input_layer.bias"
    ),

    brain1_w=array(
        "brain.0.weight"
    ),

    brain1_b=array(
        "brain.0.bias"
    ),

    brain2_w=array(
        "brain.2.weight"
    ),

    brain2_b=array(
        "brain.2.bias"
    ),

    norm_w=array(
        "norm.weight"
    ),

    norm_b=array(
        "norm.bias"
    ),

    head_w=head_w,
    head_b=head_b,

    think_steps=np.array(
        config["think_steps"],
        dtype=np.int32,
    ),
)


shutil.copy2(
    root / "runtime.py",
    release / "runtime.py",
)

shutil.copy2(
    root / "release_run.py",
    release / "run.py",
)

shutil.copy2(
    root / "config.json",
    release / "config.json",
)

shutil.copy2(
    root / "benchmark.json",
    release / "benchmark.json",
)

shutil.copy2(
    root / "generalization_benchmark.json",
    release
    / "generalization_benchmark.json",
)


manifest = {
    "schema_version": 1,

    "id":
        "micro-jet-dumbthink",

    "name":
        "Micro Jet DumbThink",

    "version":
        "2.0",

    "entrypoint":
        "run.py",

    "runtime":
        "Python + NumPy",

    "dependencies": [
        "numpy"
    ],

    "parameters":
        14818,

    "think_steps":
        12,
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


print()
print("Micro Jet release exported.")
print()
print(
    f"Release folder: {release}"
)

print(
    f"NumPy brain:    "
    f"{release / 'weights.npz'}"
)

print()