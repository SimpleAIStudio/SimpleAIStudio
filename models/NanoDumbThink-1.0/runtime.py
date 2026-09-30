import json
from pathlib import Path

import numpy as np


TASKS = {
    "add": 0,
    "sub": 1,
    "gt": 2,
    "lt": 3,
    "next": 4,
}


class NanoRuntime:
    def __init__(self, folder):
        self.folder = Path(folder)

        with open(
            self.folder / "config.json",
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        weights = np.load(
            self.folder / "weights.npz"
        )

        self.task_embed = weights["task_embed"]

        self.input_weight = weights["input_weight"]
        self.input_bias = weights["input_bias"]

        self.brain1_weight = weights["brain1_weight"]
        self.brain1_bias = weights["brain1_bias"]

        self.brain2_weight = weights["brain2_weight"]
        self.brain2_bias = weights["brain2_bias"]

        self.norm_weight = weights["norm_weight"]
        self.norm_bias = weights["norm_bias"]

        self.head_weight = weights["head_weight"]
        self.head_bias = weights["head_bias"]

        self.think_steps = self.config["think_steps"]
        self.scale = self.config["scale"]

    def layer_norm(self, x):
        mean = x.mean()
        variance = ((x - mean) ** 2).mean()

        x = (
            x - mean
        ) / np.sqrt(
            variance + 1e-5
        )

        return (
            x * self.norm_weight
            + self.norm_bias
        )

    def forward(self, task, numbers):
        task_id = TASKS[task]

        values = np.zeros(
            3,
            dtype=np.float32,
        )

        mask = np.zeros(
            3,
            dtype=np.float32,
        )

        for i, value in enumerate(numbers):
            values[i] = float(value)
            mask[i] = 1.0

        x = np.concatenate(
            [
                self.task_embed[task_id],
                values / 20.0,
                mask,
            ]
        )

        state = np.tanh(
            self.input_weight @ x
            + self.input_bias
        )

        for _ in range(self.think_steps):
            update = (
                self.brain1_weight @ state
                + self.brain1_bias
            )

            update = np.tanh(update)

            update = (
                self.brain2_weight @ update
                + self.brain2_bias
            )

            state = self.layer_norm(
                state + update * 0.5
            )

        result = (
            self.head_weight[task_id] @ state
            + self.head_bias[task_id]
        )

        return float(result)

    def solve(self, task, numbers):
        value = self.forward(
            task,
            numbers,
        )

        if task in {"gt", "lt"}:
            return "YES" if value > 0 else "NO"

        return str(
            round(
                value * self.scale
            )
        )