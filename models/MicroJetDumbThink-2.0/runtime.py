from pathlib import Path

import numpy as np


TASKS = {
    "add": 0,
    "sub": 1,
    "mul": 2,
    "div": 3,
    "gt": 4,
    "lt": 5,
    "next": 6,
    "sum3": 7,
    "max3": 8,
    "chain_gt": 9,
}


SCALES = {
    "add": 40.0,
    "sub": 20.0,
    "mul": 24.0,
    "div": 20.0,
    "next": 20.0,
    "sum3": 24.0,
    "max3": 8.0,
}


class MicroJetRuntime:
    def __init__(self, weights_path):
        weights_path = Path(weights_path)

        data = np.load(weights_path)

        self.task_embed = data["task_embed"]

        self.input_w = data["input_w"]
        self.input_b = data["input_b"]

        self.brain1_w = data["brain1_w"]
        self.brain1_b = data["brain1_b"]

        self.brain2_w = data["brain2_w"]
        self.brain2_b = data["brain2_b"]

        self.norm_w = data["norm_w"]
        self.norm_b = data["norm_b"]

        self.head_w = data["head_w"]
        self.head_b = data["head_b"]

        self.think_steps = int(
            data["think_steps"]
        )

    @staticmethod
    def linear(x, weight, bias):
        return (
            x @ weight.T
            + bias
        )

    def layer_norm(self, x):
        mean = np.mean(x)
        variance = np.mean(
            (x - mean) ** 2
        )

        normalized = (
            (x - mean)
            / np.sqrt(
                variance
                + np.float32(1e-5)
            )
        )

        return (
            normalized
            * self.norm_w
            + self.norm_b
        )

    def predict_raw(
        self,
        task,
        numbers,
    ):
        if task not in TASKS:
            raise ValueError(
                f"Unknown task: {task}"
            )

        if len(numbers) > 5:
            raise ValueError(
                "Too many input numbers."
            )

        task_id = TASKS[task]

        values = np.zeros(
            5,
            dtype=np.float32,
        )

        mask = np.zeros(
            5,
            dtype=np.float32,
        )

        for i, value in enumerate(numbers):
            values[i] = np.float32(
                value
            )

            mask[i] = np.float32(
                1.0
            )

        task_vec = self.task_embed[
            task_id
        ]

        x = np.concatenate(
            (
                task_vec,
                values / np.float32(20.0),
                mask,
            )
        ).astype(
            np.float32,
            copy=False,
        )

        state = np.tanh(
            self.linear(
                x,
                self.input_w,
                self.input_b,
            )
        )

        for _ in range(
            self.think_steps
        ):
            hidden = np.tanh(
                self.linear(
                    state,
                    self.brain1_w,
                    self.brain1_b,
                )
            )

            update = self.linear(
                hidden,
                self.brain2_w,
                self.brain2_b,
            )

            state = self.layer_norm(
                state
                + update
                * np.float32(0.35)
            )

        raw = (
            state
            @ self.head_w[task_id]
            + self.head_b[task_id]
        )

        return float(raw)

    def solve(
        self,
        task,
        numbers,
    ):
        raw = self.predict_raw(
            task,
            numbers,
        )

        if task in {
            "gt",
            "lt",
            "chain_gt",
        }:
            return raw > 0

        return round(
            raw
            * SCALES[task]
        )