import random
from pathlib import Path


root = Path(__file__).parent
output = root / "data" / "training.txt"

rng = random.Random(3)


conversations = [
    (
        "hello",
        "hello! how are you?"
    ),
    (
        "hi",
        "hey! what are you working on?"
    ),
    (
        "how are you?",
        "i am doing well."
    ),
    (
        "what is your name?",
        "my name is Vex."
    ),
    (
        "who are you?",
        "i am Vex, a small language model."
    ),
    (
        "what can you do?",
        "i can read text and generate a response."
    ),
    (
        "are you an ai?",
        "yes. i am a small artificial intelligence model."
    ),
    (
        "where do you run?",
        "i run locally on your computer."
    ),
    (
        "are you online?",
        "no. i can run without an internet connection."
    ),
    (
        "what is SimpleAI?",
        "SimpleAI is a project for building small local models."
    ),
    (
        "tell me something",
        "small models can still learn useful patterns."
    ),
    (
        "do you like learning?",
        "learning patterns from text is what i was trained to do."
    ),
]


facts = [
    "The Earth moves around the Sun.",
    "The Moon moves around the Earth.",
    "Water can freeze into ice.",
    "Ice melts when it becomes warm enough.",
    "Rain falls from clouds.",
    "Plants use light to grow.",
    "Dogs are animals.",
    "Cats are animals.",
    "Birds have feathers.",
    "Fish live in water.",
    "Humans need water to live.",
    "The sky can appear blue during the day.",
    "Night happens when a place faces away from the Sun.",
    "A year contains twelve months.",
    "A week contains seven days.",
    "An hour contains sixty minutes.",
    "A minute contains sixty seconds.",
    "A triangle has three sides.",
    "A square has four equal sides.",
    "A computer follows instructions.",
    "Python is a programming language.",
    "A keyboard is used to type text.",
    "A screen displays visual information.",
    "Memory stores information.",
    "A model can learn patterns from examples.",
]


subjects = [
    "a dog",
    "a cat",
    "a bird",
    "a person",
    "a robot",
    "a computer",
    "a small model",
]

actions = [
    "can learn",
    "can move",
    "can observe",
    "can respond",
    "can remember patterns",
]

places = [
    "in a room",
    "outside",
    "near a house",
    "on a desk",
    "in a garden",
]


lines = []


for question, answer in conversations:
    for _ in range(120):
        lines.append(
            f"User: {question}\n"
            f"Vex: {answer}\n"
        )


for fact in facts:
    for _ in range(35):
        lines.append(
            fact + "\n"
        )


for _ in range(2500):
    subject = rng.choice(
        subjects
    )

    action = rng.choice(
        actions
    )

    place = rng.choice(
        places
    )

    lines.append(
        f"{subject.capitalize()} "
        f"{action} {place}.\n"
    )


for a in range(31):
    for b in range(31):
        lines.append(
            f"User: what is {a} + {b}?\n"
            f"Vex: {a + b}\n"
        )

        if a >= b:
            lines.append(
                f"User: what is {a} - {b}?\n"
                f"Vex: {a - b}\n"
            )


for a in range(13):
    for b in range(13):
        lines.append(
            f"User: what is {a} times {b}?\n"
            f"Vex: {a * b}\n"
        )


for number in range(101):
    kind = (
        "even"
        if number % 2 == 0
        else "odd"
    )

    lines.append(
        f"{number} is {kind}.\n"
    )


rng.shuffle(lines)


header = (
    "Vex is a small local language model.\n"
    "Vex was trained from random weights.\n"
    "Vex responds clearly and briefly.\n\n"
)


with open(
    output,
    "w",
    encoding="utf-8",
    newline="\n",
) as file:
    file.write(header)

    for line in lines:
        file.write(line)
        file.write("\n")


size = output.stat().st_size


print()
print("Vex starter corpus built.")
print()
print(
    f"Examples : {len(lines):,}"
)
print(
    f"Bytes    : {size:,}"
)
print(
    f"File     : {output}"
)
print()
print(
    "This is development data, "
    "not the final Vex corpus."
)
print()