import re
from pathlib import Path

from runtime import MicroJetRuntime


root = Path(__file__).parent

brain = MicroJetRuntime(
    root / "weights.npz"
)


def parse(text):
    text = text.strip()

    patterns = [
        (
            r"(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)",
            "sum3",
        ),
        (
            r"(\d+)\s*>\s*(\d+)\s*>\s*(\d+)",
            "chain_gt",
        ),
        (
            r"max\s+(\d+)\s+(\d+)\s+(\d+)",
            "max3",
        ),
        (
            r"(\d+)\s*\+\s*(\d+)",
            "add",
        ),
        (
            r"(\d+)\s*-\s*(\d+)",
            "sub",
        ),
        (
            r"(\d+)\s*[\*x]\s*(\d+)",
            "mul",
        ),
        (
            r"(\d+)\s*/\s*(\d+)",
            "div",
        ),
        (
            r"(\d+)\s*>\s*(\d+)",
            "gt",
        ),
        (
            r"(\d+)\s*<\s*(\d+)",
            "lt",
        ),
        (
            r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\?",
            "next",
        ),
    ]

    for pattern, task in patterns:
        match = re.fullmatch(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            numbers = tuple(
                map(
                    int,
                    match.groups(),
                )
            )

            return task, numbers

    return None


def valid_domain(
    task,
    numbers,
):
    if task in {
        "add",
        "sub",
        "gt",
        "lt",
    }:
        return all(
            0 <= value <= 20
            for value in numbers
        )

    if task == "mul":
        return all(
            0 <= value <= 12
            for value in numbers
        )

    if task == "div":
        a, b = numbers

        if not (
            0 <= a <= 20
            and 1 <= b <= 10
        ):
            return False

        if a % b != 0:
            return False

        return (
            0 <= a // b <= 10
        )

    if task == "next":
        return all(
            0 <= value <= 20
            for value in numbers
        )

    if task in {
        "sum3",
        "max3",
    }:
        return all(
            0 <= value <= 8
            for value in numbers
        )

    if task == "chain_gt":
        return all(
            0 <= value <= 10
            for value in numbers
        )

    return False


print()
print("Micro Jet DumbThink 2.0")
print("-----------------------")
print()
print("14,818 parameters")
print("12 thinking steps")
print()
print("Try:")
print("  7 + 12")
print("  14 - 3")
print("  6 * 7")
print("  18 / 3")
print("  9 > 4")
print("  3 < 1")
print("  2 5 8 11 ?")
print("  3 + 4 + 5")
print("  max 3 8 5")
print("  9 > 6 > 2")
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

    parsed = parse(text)

    if parsed is None:
        print(
            "Micro Jet > I don't know "
            "that problem type yet."
        )
        print()
        continue

    task, numbers = parsed

    if not valid_domain(
        task,
        numbers,
    ):
        print(
            "Micro Jet > that problem "
            "is outside my 2.0 training domain."
        )
        print()
        continue

    answer = brain.solve(
        task,
        numbers,
    )

    if task in {
        "gt",
        "lt",
        "chain_gt",
    }:
        answer = (
            "YES"
            if answer
            else "NO"
        )

    print(
        f"Micro Jet > {answer}"
    )
    print()