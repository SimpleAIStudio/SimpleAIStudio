import random
from pathlib import Path


root = Path(__file__).parent
data_dir = root / "data"

data_dir.mkdir(
    parents=True,
    exist_ok=True,
)

train_path = data_dir / "v3b_train.txt"
valid_path = data_dir / "v3b_valid.txt"

rng = random.Random(300)


examples = set()


def add(text):
    text = text.strip()

    if text:
        examples.add(
            text + "\n"
        )


def qa(question, answer):
    add(
        f"User: {question}\n"
        f"Vex: {answer}"
    )


# -------------------------------------------------
# VEX / SIMPLEAI IDENTITY
# -------------------------------------------------

identity = [
    (
        [
            "what is your name?",
            "who are you?",
            "tell me your name.",
            "what should i call you?",
        ],
        [
            "my name is Vex.",
            "i am Vex.",
            "you can call me Vex.",
        ],
    ),

    (
        [
            "what are you?",
            "are you an ai?",
            "are you a language model?",
            "tell me about yourself.",
        ],
        [
            "i am Vex, a small local language model.",
            "yes. i am a small language model called Vex.",
            "i am a small artificial intelligence model.",
        ],
    ),

    (
        [
            "where do you run?",
            "do you run locally?",
            "where are you running?",
        ],
        [
            "i run locally on your computer.",
            "i am designed to run locally.",
        ],
    ),

    (
        [
            "are you online?",
            "do you need internet?",
            "can you work without internet?",
        ],
        [
            "i can run without an internet connection.",
            "i do not need the internet to generate text locally.",
        ],
    ),

    (
        [
            "what is SimpleAI?",
            "tell me about SimpleAI.",
            "what is SimpleAI's Studio?",
        ],
        [
            "SimpleAI's Studio is a project for building small local AI models.",
            "SimpleAI's Studio builds small models from scratch.",
        ],
    ),

    (
        [
            "were you pretrained?",
            "did you use pretrained weights?",
        ],
        [
            "no. my weights started randomly and were trained from scratch.",
            "no pretrained weights were used to create me.",
        ],
    ),
]


for questions, answers in identity:
    for question in questions:
        for answer in answers:
            qa(
                question,
                answer,
            )


# -------------------------------------------------
# CASUAL CONVERSATION
# -------------------------------------------------

casual = [
    (
        "hello",
        [
            "hello!",
            "hey!",
            "hello! how are you?",
        ],
    ),
    (
        "hi",
        [
            "hi!",
            "hey!",
            "hello!",
        ],
    ),
    (
        "hey vex",
        [
            "hey!",
            "hello!",
            "hey! what are you working on?",
        ],
    ),
    (
        "how are you?",
        [
            "i am doing well.",
            "i am ready to help.",
        ],
    ),
    (
        "what can you do?",
        [
            "i can read text and generate short responses.",
            "i can respond to text and recognize patterns i learned during training.",
        ],
    ),
    (
        "tell me something",
        [
            "small models can still learn useful patterns.",
            "computers can learn statistical patterns from examples.",
            "simple ideas can become interesting projects.",
        ],
    ),
    (
        "thanks",
        [
            "you're welcome!",
            "no problem!",
        ],
    ),
    (
        "thank you",
        [
            "you're welcome!",
            "happy to help.",
        ],
    ),
]


for question, answers in casual:
    for answer in answers:
        qa(
            question,
            answer,
        )


# -------------------------------------------------
# FACTS
# -------------------------------------------------

facts = {
    "dog":
        "a dog is an animal commonly kept by people as a pet.",

    "cat":
        "a cat is a small animal commonly kept as a pet.",

    "bird":
        "a bird is an animal with feathers and a beak.",

    "fish":
        "a fish is an animal that lives in water and uses gills.",

    "tree":
        "a tree is a plant with a woody trunk and branches.",

    "water":
        "water is a liquid that living things need.",

    "ice":
        "ice is frozen water.",

    "rain":
        "rain is liquid water that falls from clouds.",

    "earth":
        "Earth is the planet where humans live.",

    "moon":
        "the Moon is Earth's natural satellite.",

    "sun":
        "the Sun is the star at the center of the Solar System.",

    "planet":
        "a planet is a large object that orbits a star.",

    "star":
        "a star is a large glowing ball of hot gas.",

    "computer":
        "a computer is a machine that processes data and follows instructions.",

    "keyboard":
        "a keyboard is an input device used for typing.",

    "screen":
        "a screen displays visual information.",

    "memory":
        "computer memory stores information for programs to use.",

    "cpu":
        "a CPU is a processor that executes instructions.",

    "gpu":
        "a GPU is a processor designed for highly parallel calculations.",

    "python":
        "Python is a programming language.",

    "programming":
        "programming is the process of writing instructions for computers.",

    "variable":
        "a variable is a named place used to store a value in a program.",

    "function":
        "a function is a reusable block of instructions.",

    "loop":
        "a loop repeats instructions.",

    "model":
        "a machine learning model learns patterns from data.",

    "neural network":
        "a neural network is a model made from connected mathematical layers.",

    "transformer":
        "a transformer is a neural network architecture that uses attention.",

    "language model":
        "a language model predicts patterns in sequences of text.",

    "triangle":
        "a triangle is a shape with three sides.",

    "square":
        "a square is a shape with four equal sides.",

    "circle":
        "a circle is a round shape whose edge is equally distant from its center.",

    "week":
        "a week contains seven days.",

    "year":
        "a year contains twelve months.",

    "hour":
        "an hour contains sixty minutes.",

    "minute":
        "a minute contains sixty seconds.",
}


question_forms = [
    "what is {thing}?",
    "what is a {thing}?",
    "tell me about {thing}.",
    "explain {thing}.",
    "describe {thing}.",
]


for thing, answer in facts.items():
    for form in question_forms:
        question = form.format(
            thing=thing
        )

        qa(
            question,
            answer,
        )

    add(answer)


# -------------------------------------------------
# SIMPLE SCIENCE RELATIONSHIPS
# -------------------------------------------------

science = [
    (
        "what happens when water freezes?",
        "water becomes ice when it freezes.",
    ),
    (
        "what happens when ice melts?",
        "ice becomes liquid water when it melts.",
    ),
    (
        "why do plants need light?",
        "plants use light as an energy source for growth.",
    ),
    (
        "what does Earth orbit?",
        "Earth orbits the Sun.",
    ),
    (
        "what does the Moon orbit?",
        "the Moon orbits Earth.",
    ),
    (
        "what causes day and night?",
        "day and night happen because Earth rotates.",
    ),
    (
        "do humans need water?",
        "yes. humans need water to live.",
    ),
]


for question, answer in science:
    qa(
        question,
        answer,
    )


# -------------------------------------------------
# ARITHMETIC
# -------------------------------------------------

plus_forms = [
    "what is {a} + {b}?",
    "what is {a} plus {b}?",
    "add {a} and {b}.",
    "calculate {a} + {b}.",
]


minus_forms = [
    "what is {a} - {b}?",
    "what is {a} minus {b}?",
    "subtract {b} from {a}.",
    "calculate {a} - {b}.",
]


multiply_forms = [
    "what is {a} times {b}?",
    "what is {a} multiplied by {b}?",
    "calculate {a} * {b}.",
    "multiply {a} and {b}.",
]


for a in range(101):
    for b in range(101):
        if rng.random() < 0.22:
            answer = a + b

            for form in rng.sample(
                plus_forms,
                2,
            ):
                qa(
                    form.format(
                        a=a,
                        b=b,
                    ),
                    str(answer),
                )

        if rng.random() < 0.22:
            answer = a - b

            for form in rng.sample(
                minus_forms,
                2,
            ):
                qa(
                    form.format(
                        a=a,
                        b=b,
                    ),
                    str(answer),
                )


for a in range(26):
    for b in range(26):
        answer = a * b

        for form in multiply_forms:
            qa(
                form.format(
                    a=a,
                    b=b,
                ),
                str(answer),
            )


for divisor in range(1, 21):
    for result in range(21):
        value = (
            divisor
            * result
        )

        qa(
            f"what is {value} divided by {divisor}?",
            str(result),
        )

        qa(
            f"calculate {value} / {divisor}.",
            str(result),
        )


# -------------------------------------------------
# NUMBER CLASSIFICATION
# -------------------------------------------------

for number in range(201):
    parity = (
        "even"
        if number % 2 == 0
        else "odd"
    )

    other = (
        "odd"
        if parity == "even"
        else "even"
    )

    qa(
        f"is {number} {parity}?",
        "yes.",
    )

    qa(
        f"is {number} {other}?",
        "no.",
    )

    qa(
        f"is {number} even or odd?",
        f"{number} is {parity}.",
    )


# -------------------------------------------------
# COMPARISONS
# -------------------------------------------------

for _ in range(12000):
    a = rng.randint(
        0,
        500,
    )

    b = rng.randint(
        0,
        500,
    )

    if a > b:
        relation = "greater than"
    elif a < b:
        relation = "less than"
    else:
        relation = "equal to"

    qa(
        f"is {a} greater than {b}?",
        (
            "yes."
            if a > b
            else "no."
        ),
    )

    qa(
        f"is {a} less than {b}?",
        (
            "yes."
            if a < b
            else "no."
        ),
    )

    qa(
        f"compare {a} and {b}.",
        f"{a} is {relation} {b}.",
    )


# -------------------------------------------------
# MIN / MAX
# -------------------------------------------------

for _ in range(7000):
    numbers = [
        rng.randint(
            0,
            200,
        )
        for _ in range(3)
    ]

    text = " ".join(
        map(
            str,
            numbers,
        )
    )

    qa(
        f"what is the largest of {text}?",
        str(max(numbers)),
    )

    qa(
        f"what is the smallest of {text}?",
        str(min(numbers)),
    )


# -------------------------------------------------
# ARITHMETIC SEQUENCES
# -------------------------------------------------

for _ in range(10000):
    start = rng.randint(
        -50,
        100,
    )

    step = rng.choice(
        [
            -10,
            -8,
            -6,
            -5,
            -4,
            -3,
            -2,
            -1,
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
        ]
    )

    values = [
        start + step * i
        for i in range(5)
    ]

    answer = (
        start
        + step * 5
    )

    sequence = " ".join(
        map(
            str,
            values,
        )
    )

    qa(
        f"what comes next: {sequence} ?",
        str(answer),
    )


# -------------------------------------------------
# BOOLEAN LOGIC
# -------------------------------------------------

values = [
    True,
    False,
]


for a in values:
    for b in values:
        a_text = str(a).lower()
        b_text = str(b).lower()

        qa(
            f"{a_text} and {b_text}?",
            str(
                a and b
            ).lower(),
        )

        qa(
            f"{a_text} or {b_text}?",
            str(
                a or b
            ).lower(),
        )

        qa(
            f"{a_text} xor {b_text}?",
            str(
                a != b
            ).lower(),
        )


qa(
    "not true?",
    "false",
)

qa(
    "not false?",
    "true",
)


# -------------------------------------------------
# GENERATED SENTENCES
# -------------------------------------------------

subjects = [
    "the dog",
    "the cat",
    "the bird",
    "the robot",
    "the student",
    "the programmer",
    "the computer",
    "the small model",
    "the scientist",
]

verbs = [
    "walked",
    "waited",
    "looked",
    "worked",
    "learned",
    "moved",
    "rested",
    "watched",
]

places = [
    "in the garden",
    "inside the room",
    "near the window",
    "beside the house",
    "at the desk",
    "outside",
]

endings = [
    "during the morning.",
    "before lunch.",
    "for a short time.",
    "while everything was quiet.",
    "and then stopped.",
]


for _ in range(30000):
    add(
        f"{rng.choice(subjects).capitalize()} "
        f"{rng.choice(verbs)} "
        f"{rng.choice(places)} "
        f"{rng.choice(endings)}"
    )


# -------------------------------------------------
# TINY GENERATED STORIES
# -------------------------------------------------

names = [
    "Alex",
    "Sam",
    "Jamie",
    "Taylor",
    "Morgan",
]

objects = [
    "book",
    "ball",
    "box",
    "computer",
    "lamp",
]

locations = [
    "desk",
    "table",
    "shelf",
    "floor",
    "chair",
]


for _ in range(12000):
    name = rng.choice(
        names
    )

    item = rng.choice(
        objects
    )

    location = rng.choice(
        locations
    )

    add(
        f"{name} had a {item}. "
        f"{name} placed the {item} on the {location}. "
        f"The {item} was now on the {location}."
    )


# -------------------------------------------------
# SIMPLE READING QUESTIONS
# -------------------------------------------------

for _ in range(8000):
    name = rng.choice(
        names
    )

    item = rng.choice(
        objects
    )

    location = rng.choice(
        locations
    )

    add(
        f"Text: {name} put the {item} on the {location}.\n"
        f"Question: where is the {item}?\n"
        f"Answer: the {item} is on the {location}."
    )


# -------------------------------------------------
# SPLIT
# -------------------------------------------------

examples = list(
    examples
)

rng.shuffle(
    examples
)


valid_count = max(
    1,
    int(
        len(examples)
        * 0.05
    )
)


valid_examples = examples[
    :valid_count
]

train_examples = examples[
    valid_count:
]


header = (
    "Vex is a small local language model "
    "built by SimpleAI's Studio.\n\n"
)


with open(
    train_path,
    "w",
    encoding="utf-8",
    newline="\n",
) as file:
    file.write(header)

    for example in train_examples:
        file.write(example)
        file.write("\n")


with open(
    valid_path,
    "w",
    encoding="utf-8",
    newline="\n",
) as file:
    file.write(header)

    for example in valid_examples:
        file.write(example)
        file.write("\n")


train_size = (
    train_path.stat().st_size
)

valid_size = (
    valid_path.stat().st_size
)


print()
print("Vex V3-B corpus")
print("----------------")
print()

print(
    f"Examples    : "
    f"{len(examples):,}"
)

print(
    f"Train       : "
    f"{len(train_examples):,} examples"
)

print(
    f"Validation  : "
    f"{len(valid_examples):,} examples"
)

print()

print(
    f"Train size  : "
    f"{train_size / 1024 / 1024:.2f} MB"
)

print(
    f"Valid size  : "
    f"{valid_size / 1024 / 1024:.2f} MB"
)

print()

print(
    f"Train file  : "
    f"{train_path}"
)

print(
    f"Valid file  : "
    f"{valid_path}"
)

print()