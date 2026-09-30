import re
from pathlib import Path

from runtime import NanoRuntime


ROOT = Path(__file__).parent

brain = NanoRuntime(ROOT)


def parse_problem(text):
    checks = [
        (
            r"(\d+)\s*\+\s*(\d+)",
            "add",
        ),
        (
            r"(\d+)\s*-\s*(\d+)",
            "sub",
        ),
        (
            r"(\d+)\s*>\s*(\d+)",
            "gt",
        ),
        (
            r"(\d+)\s*<\s*(\d+)",
            "lt",
        ),
    ]

    for pattern, task in checks:
        match = re.fullmatch(
            pattern,
            text,
        )

        if match:
            numbers = tuple(
                map(
                    int,
                    match.groups(),
                )
            )

            return task, numbers

    match = re.fullmatch(
        r"(\d+)\s+(\d+)\s+(\d+)\s+\?",
        text,
    )

    if match:
        numbers = tuple(
            map(
                int,
                match.groups(),
            )
        )

        return "next", numbers

    return None


print()
print("Nano DumbThink 1.0")
print("-------------------")
print()
print("3,901 parameters")
print("8 thinking steps")
print()
print("Try:")
print("  7 + 12")
print("  14 - 3")
print("  9 > 4")
print("  3 < 1")
print("  2 5 8 ?")
print()
print("/exit to quit")
print()


while True:
    try:
        text = input(
            "Problem > "
        ).strip()

    except (
        KeyboardInterrupt,
        EOFError,
    ):
        print()
        break

    if not text:
        continue

    if text.lower() in {
        "/exit",
        "/quit",
    }:
        break

    problem = parse_problem(text)

    if problem is None:
        print(
            "DumbThink > I don't know "
            "that kind of problem yet."
        )
        print()
        continue

    task, numbers = problem

    if not all(
        0 <= number <= 20
        for number in numbers
    ):
        print(
            "DumbThink > use numbers "
            "from 0 to 20."
        )
        print()
        continue

    answer = brain.solve(
        task,
        numbers,
    )

    print(
        f"DumbThink > {answer}"
    )

    print()