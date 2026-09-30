from dataclasses import dataclass

import torch

from model import TASKS


SCALE = 20.0


@dataclass
class Sample:
    task: str
    numbers: tuple
    answer: float
    kind: str


def is_test_sample(task, numbers):
    value = TASKS[task] * 97

    for i, number in enumerate(numbers):
        value += int(number) * (31 + i * 18)

    return value % 5 == 0


def build_samples():
    samples = []

    for a in range(21):
        for b in range(21):
            samples.append(
                Sample("add", (a, b), a + b, "number")
            )

            samples.append(
                Sample("sub", (a, b), a - b, "number")
            )

            samples.append(
                Sample("gt", (a, b), float(a > b), "logic")
            )

            samples.append(
                Sample("lt", (a, b), float(a < b), "logic")
            )

    for start in range(21):
        for step in range(-5, 6):
            a = start
            b = start + step
            c = start + step * 2
            answer = start + step * 3

            if not 0 <= b <= 20:
                continue

            if not 0 <= c <= 20:
                continue

            if not -20 <= answer <= 40:
                continue

            samples.append(
                Sample(
                    "next",
                    (a, b, c),
                    answer,
                    "number",
                )
            )

    train = []
    test = []

    for sample in samples:
        if is_test_sample(sample.task, sample.numbers):
            test.append(sample)
        else:
            train.append(sample)

    return train, test


def training_target(sample):
    if sample.task == "gt":
        a, b = sample.numbers

        # Half-step gives equality a negative margin.
        return ((a - b) - 0.5) / SCALE

    if sample.task == "lt":
        a, b = sample.numbers

        return ((b - a) - 0.5) / SCALE

    return sample.answer / SCALE


def make_batch(samples):
    tasks = []
    numbers = []
    masks = []
    targets = []

    for sample in samples:
        tasks.append(TASKS[sample.task])

        values = [0.0, 0.0, 0.0]
        mask = [0.0, 0.0, 0.0]

        for i, value in enumerate(sample.numbers):
            values[i] = float(value)
            mask[i] = 1.0

        numbers.append(values)
        masks.append(mask)
        targets.append(training_target(sample))

    return {
        "task": torch.tensor(
            tasks,
            dtype=torch.long,
        ),
        "numbers": torch.tensor(
            numbers,
            dtype=torch.float32,
        ),
        "mask": torch.tensor(
            masks,
            dtype=torch.float32,
        ),
        "target": torch.tensor(
            targets,
            dtype=torch.float32,
        ),
    }