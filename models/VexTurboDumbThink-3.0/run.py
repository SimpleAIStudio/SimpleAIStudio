import sys

from runtime import VexRuntime


vex = VexRuntime()


if len(sys.argv) > 1:
    prompt = " ".join(
        sys.argv[1:]
    )

    print(
        vex.respond(
            prompt
        )
    )

    raise SystemExit


print()
print("Vex Turbo DumbThink 3.0")
print("-----------------------")
print()
print("496,704 parameters")
print("256-byte context")
print()
print("/exit to quit")
print()


while True:
    try:
        message = input(
            "You > "
        ).strip()

    except (
        KeyboardInterrupt,
        EOFError,
    ):
        print()
        break

    if not message:
        continue

    if message.lower() in {
        "/exit",
        "/quit",
    }:
        break

    response = vex.respond(
        message
    )

    print(
        f"Vex > {response}"
    )

    print()