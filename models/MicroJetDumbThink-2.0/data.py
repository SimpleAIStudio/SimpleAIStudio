from dataclasses import dataclass

import torch

from model import TASKS


SCALES = {
    "add": 40.0,
    "sub": 20.0,

    # Smaller scale = one-number errors matter more.
    # 144 / 24 = 6, which is still easy for the output head.
    "mul": 24.0,

    "div": 20.0,
    "next": 20.0,
    "sum3": 24.0,
    "max3": 8.0,
}


@dataclass
class Sample:
    task: str
    numbers: tuple
    answer: float
    kind: str


def is_test_sample(task, numbers):
    # Deterministic 80/20-ish split.
    #
    # These multipliers are deliberately not divisible
    # by 5, so we don't accidentally hold out entire rows.
    multipliers = (
        31,
        37,
        43,
        47,
        53,
    )

    value = (
        TASKS[task] * 113
        + 17
    )

    for i, number in enumerate(numbers):
        value += (
            int(number)
            * multipliers[i]
        )

    return value % 5 == 0


def logic_target(sample):
    if sample.task == "gt":
        a, b = sample.numbers

        return (
            (a - b) - 0.5
        ) / 20.0

    if sample.task == "lt":
        a, b = sample.numbers

        return (
            (b - a) - 0.5
        ) / 20.0

    if sample.task == "chain_gt":
        a, b, c = sample.numbers

        weakest = min(
            a - b,
            b - c,
        )

        return (
            weakest - 0.5
        ) / 10.0

    raise ValueError(
        f"Unknown logic task: {sample.task}"
    )


def training_target(sample):
    if sample.kind == "logic":
        return logic_target(sample)

    return (
        sample.answer
        / SCALES[sample.task]
    )


def build_samples():
    samples = []

    for a in range(21):
        for b in range(21):
            samples.append(
                Sample(
                    "add",
                    (a, b),
                    a + b,
                    "number",
                )
            )

            samples.append(
                Sample(
                    "sub",
                    (a, b),
                    a - b,
                    "number",
                )
            )

            samples.append(
                Sample(
                    "gt",
                    (a, b),
                    float(a > b),
                    "logic",
                )
            )

            samples.append(
                Sample(
                    "lt",
                    (a, b),
                    float(a < b),
                    "logic",
                )
            )

    for a in range(13):
        for b in range(13):
            samples.append(
                Sample(
                    "mul",
                    (a, b),
                    a * b,
                    "number",
                )
            )

    for divisor in range(1, 11):
        for answer in range(11):
            value = (
                divisor
                * answer
            )

            if value > 20:
                continue

            samples.append(
                Sample(
                    "div",
                    (value, divisor),
                    answer,
                    "number",
                )
            )

    for start in range(21):
        for step in range(-4, 5):
            if step == 0:
                continue

            values = tuple(
                start + step * i
                for i in range(4)
            )

            answer = (
                start
                + step * 4
            )

            if not all(
                0 <= value <= 20
                for value in values
            ):
                continue

            if not (
                0 <= answer <= 20
            ):
                continue

            samples.append(
                Sample(
                    "next",
                    values,
                    answer,
                    "number",
                )
            )

    for a in range(9):
        for b in range(9):
            for c in range(9):
                samples.append(
                    Sample(
                        "sum3",
                        (a, b, c),
                        a + b + c,
                        "number",
                    )
                )

                samples.append(
                    Sample(
                        "max3",
                        (a, b, c),
                        max(a, b, c),
                        "number",
                    )
                )

    for a in range(11):
        for b in range(11):
            for c in range(11):
                samples.append(
                    Sample(
                        "chain_gt",
                        (a, b, c),
                        float(
                            a > b
                            and b > c
                        ),
                        "logic",
                    )
                )

    train = []
    test = []

    for sample in samples:
        if is_test_sample(
            sample.task,
            sample.numbers,
        ):
            test.append(sample)
        else:
            train.append(sample)

    return train, test


def make_batch(samples):
    tasks = []
    numbers = []
    masks = []
    targets = []

    for sample in samples:
        tasks.append(
            TASKS[sample.task]
        )

        values = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        mask = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        for i, value in enumerate(
            sample.numbers
        ):
            values[i] = float(
                value
            )

            mask[i] = 1.0

        numbers.append(values)
        masks.append(mask)

        targets.append(
            training_target(
                sample
            )
        )

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


def decode(sample, raw):
    if sample.kind == "logic":
        return raw > 0

    return round(
        raw
        * SCALES[
            sample.task
        ]
    )