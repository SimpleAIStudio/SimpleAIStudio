import json
import random
import shutil
from pathlib import Path


root = Path(__file__).parent

output_dir = (
    root
    / "data"
    / "v3c"
)

train_dir = (
    output_dir
    / "train"
)

valid_dir = (
    output_dir
    / "valid"
)

benchmark_path = (
    output_dir
    / "benchmark.json"
)

manifest_path = (
    output_dir
    / "manifest.json"
)


seed = 303

rng = random.Random(seed)

benchmark_rng = random.Random(
    3303
)


if output_dir.exists():
    shutil.rmtree(
        output_dir
    )


train_dir.mkdir(
    parents=True,
    exist_ok=True,
)

valid_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# -------------------------------------------------
# CORE CONTENT
# -------------------------------------------------

knowledge = [
    (
        "Earth",
        "Earth is the planet where humans live.",
        ["planet"],
    ),
    (
        "the Moon",
        "the Moon is Earth's natural satellite.",
        ["satellite"],
    ),
    (
        "the Sun",
        "the Sun is the star at the center of the Solar System.",
        ["star"],
    ),
    (
        "water",
        "water is a liquid that living things need.",
        ["liquid"],
    ),
    (
        "ice",
        "ice is frozen water.",
        ["frozen water"],
    ),
    (
        "rain",
        "rain is liquid water that falls from clouds.",
        ["cloud"],
    ),
    (
        "a dog",
        "a dog is an animal commonly kept as a pet.",
        ["animal"],
    ),
    (
        "a cat",
        "a cat is a small animal commonly kept as a pet.",
        ["animal"],
    ),
    (
        "a bird",
        "a bird is an animal with feathers and a beak.",
        ["feather"],
    ),
    (
        "a fish",
        "a fish is an animal that lives in water and uses gills.",
        ["gill"],
    ),
    (
        "a tree",
        "a tree is a plant with a woody trunk and branches.",
        ["plant"],
    ),
    (
        "a planet",
        "a planet is a large object that orbits a star.",
        ["orbit"],
    ),
    (
        "a star",
        "a star is a large glowing ball of hot gas.",
        ["gas"],
    ),
    (
        "gravity",
        "gravity is a force that attracts masses toward each other.",
        ["force"],
    ),
    (
        "oxygen",
        "oxygen is a chemical element and a gas used by humans for respiration.",
        ["gas"],
    ),
    (
        "a week",
        "a week contains seven days.",
        ["seven"],
    ),
    (
        "a year",
        "a year contains twelve months.",
        ["twelve"],
    ),
    (
        "an hour",
        "an hour contains sixty minutes.",
        ["sixty"],
    ),
    (
        "a minute",
        "a minute contains sixty seconds.",
        ["sixty"],
    ),
    (
        "a triangle",
        "a triangle is a shape with three sides.",
        ["three"],
    ),
    (
        "a square",
        "a square is a shape with four equal sides.",
        ["four"],
    ),
    (
        "a circle",
        "a circle is a round shape whose edge is equally distant from its center.",
        ["round"],
    ),
    (
        "photosynthesis",
        "photosynthesis is the process plants use to turn light energy into chemical energy.",
        ["plants", "light"],
    ),
    (
        "evaporation",
        "evaporation is the change of a liquid into a gas.",
        ["liquid", "gas"],
    ),
    (
        "the heart",
        "the heart is an organ that pumps blood through the body.",
        ["blood"],
    ),
]


coding = [
    (
        "Python",
        "Python is a programming language.",
        ["programming language"],
    ),
    (
        "a variable",
        "a variable is a named place used to store a value.",
        ["store", "value"],
    ),
    (
        "a function",
        "a function is a reusable block of instructions.",
        ["reusable"],
    ),
    (
        "a loop",
        "a loop repeats instructions.",
        ["repeat"],
    ),
    (
        "a list",
        "a list stores an ordered collection of values.",
        ["collection"],
    ),
    (
        "a dictionary",
        "a dictionary stores key-value pairs.",
        ["key", "value"],
    ),
    (
        "a string",
        "a string is a sequence of text characters.",
        ["text"],
    ),
    (
        "an integer",
        "an integer is a whole number.",
        ["whole number"],
    ),
    (
        "a boolean",
        "a boolean is a value that is either true or false.",
        ["true", "false"],
    ),
    (
        "an if statement",
        "an if statement runs code when a condition is met.",
        ["condition"],
    ),
    (
        "a class",
        "a class is a blueprint for creating objects.",
        ["blueprint"],
    ),
    (
        "an object",
        "an object is an instance of a class.",
        ["instance"],
    ),
    (
        "JSON",
        "JSON is a text format for structured data.",
        ["structured data"],
    ),
    (
        "a CPU",
        "a CPU executes general-purpose instructions.",
        ["instructions"],
    ),
    (
        "a GPU",
        "a GPU performs many calculations in parallel.",
        ["parallel"],
    ),
    (
        "RAM",
        "RAM is short-term working memory used by programs.",
        ["memory"],
    ),
    (
        "a neural network",
        "a neural network is a model made from connected mathematical layers.",
        ["layers"],
    ),
    (
        "a transformer",
        "a transformer is a neural network architecture that uses attention.",
        ["attention"],
    ),
    (
        "attention",
        "attention lets a model weigh information from different positions in a sequence.",
        ["sequence"],
    ),
    (
        "a token",
        "a token is a unit of input processed by a language model.",
        ["input"],
    ),
    (
        "a byte",
        "a byte is a unit of digital information containing eight bits.",
        ["eight", "bits"],
    ),
    (
        "an algorithm",
        "an algorithm is a sequence of steps for solving a problem.",
        ["steps"],
    ),
    (
        "a parameter",
        "a model parameter is a learned numerical value.",
        ["learned", "value"],
    ),
    (
        "training",
        "training adjusts model parameters using data.",
        ["parameters", "data"],
    ),
    (
        "inference",
        "inference is the process of using a trained model to produce an output.",
        ["trained model"],
    ),
]


plurals = {
    "cat": "cats",
    "dog": "dogs",
    "book": "books",
    "car": "cars",
    "tree": "trees",
    "box": "boxes",
    "dish": "dishes",
    "class": "classes",
    "baby": "babies",
    "city": "cities",
    "toy": "toys",
    "key": "keys",
    "bus": "buses",
    "fox": "foxes",
    "watch": "watches",
    "apple": "apples",
    "chair": "chairs",
    "table": "tables",
    "bird": "birds",
    "house": "houses",
    "day": "days",
    "boy": "boys",
    "girl": "girls",
    "computer": "computers",
    "robot": "robots",
}


opposites = {
    "hot": "cold",
    "big": "small",
    "up": "down",
    "fast": "slow",
    "light": "dark",
    "old": "young",
    "happy": "sad",
    "open": "closed",
    "early": "late",
    "inside": "outside",
    "full": "empty",
    "hard": "soft",
    "high": "low",
    "near": "far",
    "wet": "dry",
    "strong": "weak",
    "day": "night",
    "left": "right",
    "start": "stop",
    "true": "false",
    "on": "off",
    "more": "less",
    "above": "below",
    "before": "after",
    "same": "different",
}


names = [
    "Alex",
    "Sam",
    "Jamie",
    "Taylor",
    "Morgan",
    "Robin",
    "Charlie",
    "Emery",
    "Frankie",
    "Sky",
]


objects = [
    "lamp",
    "book",
    "mug",
    "ball",
    "key",
    "box",
    "phone",
    "hat",
    "coin",
    "notebook",
]


places = [
    "table",
    "desk",
    "shelf",
    "chair",
    "floor",
    "bench",
    "drawer",
    "counter",
    "bed",
    "windowsill",
]


# -------------------------------------------------
# BENCHMARK
# -------------------------------------------------

benchmark = []


def benchmark_add(
    category,
    prompt,
    expected,
    match="exact",
):
    if isinstance(
        expected,
        str,
    ):
        expected = [expected]

    number = (
        sum(
            item["category"] == category
            for item in benchmark
        )
        + 1
    )

    benchmark.append(
        {
            "id":
                f"{category}-{number:03d}",

            "category":
                category,

            "prompt":
                prompt,

            "match":
                match,

            "expected":
                expected,
        }
    )


# -------------------------------------------------
# 50 CONVERSATION
# -------------------------------------------------

greetings = [
    "hello",
    "hi",
    "hey",
    "hello vex",
    "hey vex",
    "good morning",
    "good afternoon",
    "good evening",
    "hi there",
    "hello there",
]


thanks = [
    "thanks",
    "thank you",
    "thanks vex",
    "thank you vex",
    "thanks for helping",
]


benchmark_prefixes = [
    "",
    "quickly, ",
    "just saying, ",
    "okay, ",
    "hey, ",
]


for i in range(25):
    prompt = (
        benchmark_prefixes[
            i // 5
        ]
        + greetings[
            i % len(greetings)
        ]
    )

    benchmark_add(
        "conversation",
        prompt,
        [
            "hello",
            "hi",
            "hey",
        ],
        "contains_any",
    )


for i in range(25):
    prompt = (
        benchmark_prefixes[
            i // 5
        ]
        + thanks[
            i % len(thanks)
        ]
    )

    benchmark_add(
        "conversation",
        prompt,
        [
            "welcome",
            "no problem",
            "happy to help",
        ],
        "contains_any",
    )


# -------------------------------------------------
# 50 IDENTITY
# -------------------------------------------------

identity_benchmark = [
    (
        "what is your name?",
        ["vex"],
        "contains_all",
    ),
    (
        "who are you?",
        ["vex"],
        "contains_all",
    ),
    (
        "what should i call you?",
        ["vex"],
        "contains_all",
    ),
    (
        "introduce yourself.",
        ["vex"],
        "contains_all",
    ),
    (
        "are you a language model?",
        ["yes"],
        "contains_any",
    ),
    (
        "are you an ai?",
        ["yes"],
        "contains_any",
    ),
    (
        "do you run locally?",
        ["local"],
        "contains_any",
    ),
    (
        "do you need the internet?",
        ["no", "without"],
        "contains_any",
    ),
    (
        "were you trained from scratch?",
        ["yes", "scratch"],
        "contains_any",
    ),
    (
        "did you use pretrained weights?",
        ["no", "pretrained"],
        "contains_any",
    ),
]


identity_prefixes = [
    "",
    "quick question: ",
    "tell me this: ",
    "i want to know: ",
    "can you answer this: ",
]


for question, expected, match in identity_benchmark:
    for prefix in identity_prefixes:
        benchmark_add(
            "identity",
            prefix + question,
            expected,
            match,
        )


# -------------------------------------------------
# 50 KNOWLEDGE
# -------------------------------------------------

for thing, answer, keywords in knowledge:
    benchmark_add(
        "knowledge",
        f"briefly explain {thing}.",
        keywords,
        "contains_all",
    )

    benchmark_add(
        "knowledge",
        f"give me one fact about {thing}.",
        keywords,
        "contains_all",
    )


# -------------------------------------------------
# 50 READING
# -------------------------------------------------

benchmark_names = [
    "Avery",
    "Blake",
    "Casey",
    "Drew",
    "Elliot",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
]


for i in range(50):
    name = benchmark_names[
        i % len(benchmark_names)
    ]

    item = objects[
        (i * 3)
        % len(objects)
    ]

    place = places[
        (i * 7)
        % len(places)
    ]

    benchmark_add(
        "reading",
        (
            f"Read this: {name} put the "
            f"{item} on the {place}. "
            f"Where is the {item}?"
        ),
        [place],
        "contains_all",
    )


# -------------------------------------------------
# 50 LANGUAGE
# -------------------------------------------------

for word, plural in plurals.items():
    benchmark_add(
        "language",
        f"what is the plural of {word}?",
        [plural],
        "exact",
    )


for word, opposite in opposites.items():
    benchmark_add(
        "language",
        f"what is the opposite of {word}?",
        [opposite],
        "exact",
    )


# -------------------------------------------------
# 50 ARITHMETIC
# -------------------------------------------------

for i in range(50):
    operation = i % 4

    if operation == 0:
        a = benchmark_rng.randint(
            3,
            95,
        )

        b = benchmark_rng.randint(
            2,
            95,
        )

        prompt = (
            f"please add {a} and {b}."
        )

        answer = a + b

    elif operation == 1:
        a = benchmark_rng.randint(
            20,
            100,
        )

        b = benchmark_rng.randint(
            0,
            a,
        )

        prompt = (
            f"please subtract {b} from {a}."
        )

        answer = a - b

    elif operation == 2:
        a = benchmark_rng.randint(
            2,
            20,
        )

        b = benchmark_rng.randint(
            2,
            20,
        )

        prompt = (
            f"please multiply {a} by {b}."
        )

        answer = a * b

    else:
        divisor = benchmark_rng.randint(
            2,
            15,
        )

        answer = benchmark_rng.randint(
            2,
            15,
        )

        value = (
            divisor
            * answer
        )

        prompt = (
            f"please divide "
            f"{value} by {divisor}."
        )

    benchmark_add(
        "arithmetic",
        prompt,
        [str(answer)],
        "exact",
    )


# -------------------------------------------------
# 50 LOGIC
# -------------------------------------------------

for _ in range(25):
    number = benchmark_rng.randint(
        1,
        250,
    )

    kind = (
        "even"
        if number % 2 == 0
        else "odd"
    )

    benchmark_add(
        "logic",
        f"is {number} {kind}?",
        ["yes"],
        "exact",
    )


for _ in range(25):
    a = benchmark_rng.randint(
        0,
        300,
    )

    b = benchmark_rng.randint(
        0,
        300,
    )

    answer = (
        "yes"
        if a > b
        else "no"
    )

    benchmark_add(
        "logic",
        f"is {a} larger than {b}?",
        [answer],
        "exact",
    )


# -------------------------------------------------
# 50 SEQUENCES
# -------------------------------------------------

steps = [
    -9,
    -7,
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
    7,
    9,
]


for _ in range(50):
    start = benchmark_rng.randint(
        -30,
        60,
    )

    step = benchmark_rng.choice(
        steps
    )

    values = [
        start + step * index
        for index in range(5)
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

    benchmark_add(
        "sequences",
        (
            f"continue this sequence: "
            f"{sequence} ?"
        ),
        [str(answer)],
        "exact",
    )


# -------------------------------------------------
# 50 CODING
# -------------------------------------------------

for thing, answer, keywords in coding:
    benchmark_add(
        "coding",
        (
            f"in one sentence, "
            f"explain {thing}."
        ),
        keywords,
        "contains_all",
    )

    benchmark_add(
        "coding",
        (
            f"what does {thing} "
            f"mean in computing?"
        ),
        keywords,
        "contains_all",
    )


# -------------------------------------------------
# 50 STORY WORLD
# -------------------------------------------------

story_names = [
    "Kai",
    "Lena",
    "Mia",
    "Noah",
    "Owen",
    "Pia",
    "Quinn",
    "Rory",
    "Sage",
    "Tess",
]


for i in range(50):
    name = story_names[
        i % len(story_names)
    ]

    first_item = objects[
        (i + 2)
        % len(objects)
    ]

    second_item = objects[
        (i + 5)
        % len(objects)
    ]

    first_place = places[
        (i + 1)
        % len(places)
    ]

    second_place = places[
        (i + 6)
        % len(places)
    ]

    benchmark_add(
        "story_world",
        (
            f"{name} put the "
            f"{first_item} on the "
            f"{first_place}. "
            f"Then {name} moved the "
            f"{second_item} to the "
            f"{second_place}. "
            f"Where is the "
            f"{second_item}?"
        ),
        [second_place],
        "contains_all",
    )


if len(benchmark) != 500:
    raise RuntimeError(
        "Benchmark should contain "
        f"500 items, got {len(benchmark)}."
    )


benchmark_prompts = {
    item["prompt"]
    .strip()
    .lower()

    for item in benchmark
}


# -------------------------------------------------
# TRAINING DATA
# -------------------------------------------------

categories = {
    category:
        set()

    for category in [
        "conversation",
        "identity",
        "knowledge",
        "reading",
        "language",
        "arithmetic",
        "logic",
        "sequences",
        "coding",
        "story_world",
    ]
}


def add_qa(
    category,
    question,
    answer,
):
    question = question.strip()
    answer = answer.strip()

    if (
        question.lower()
        in benchmark_prompts
    ):
        return

    text = (
        f"User: {question}\n"
        f"Vex: {answer}"
    )

    categories[
        category
    ].add(text)


wrappers = [
    "{}",
    "quick question: {}",
    "please answer this: {}",
    "can you answer this: {}",
    "i want to know: {}",
]


def add_wrapped(
    category,
    question,
    answer,
):
    for wrapper in wrappers:
        add_qa(
            category,
            wrapper.format(
                question
            ),
            answer,
        )


# -------------------------------------------------
# CONVERSATION TRAINING
# -------------------------------------------------

conversation_pairs = [
    (
        "hello",
        [
            "hello!",
            "hey!",
            "hi!",
        ],
    ),
    (
        "hi",
        [
            "hi!",
            "hello!",
            "hey!",
        ],
    ),
    (
        "hey",
        [
            "hey!",
            "hello!",
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
        "can you help me?",
        [
            "yes. what do you need help with?",
            "yes. i can try to help.",
        ],
    ),
    (
        "what can you do?",
        [
            "i can read text and generate short responses.",
            "i can answer some questions and generate text.",
        ],
    ),
    (
        "thanks",
        [
            "you're welcome!",
            "no problem!",
            "happy to help.",
        ],
    ),
    (
        "thank you",
        [
            "you're welcome!",
            "happy to help.",
        ],
    ),
    (
        "good morning",
        [
            "good morning!",
            "hello!",
        ],
    ),
    (
        "good evening",
        [
            "good evening!",
            "hello!",
        ],
    ),
]


conversation_prefixes = [
    "",
    "hey, ",
    "okay, ",
    "quick question: ",
    "just wondering: ",
    "vex, ",
]


conversation_suffixes = [
    "",
    " please.",
    " if you can.",
    " thanks.",
]


for question, answers in conversation_pairs:
    for prefix in conversation_prefixes:
        for suffix in conversation_suffixes:
            for answer in answers:
                add_qa(
                    "conversation",
                    (
                        prefix
                        + question
                        + suffix
                    ),
                    answer,
                )


# -------------------------------------------------
# IDENTITY TRAINING
# -------------------------------------------------

identity_training = [
    (
        [
            "what is your name?",
            "tell me your name.",
            "who am i talking to?",
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
            "who are you?",
            "tell me about yourself.",
            "what are you?",
        ],
        [
            "i am Vex, a small local language model.",
            "i am a small language model called Vex.",
        ],
    ),
    (
        [
            "are you an ai?",
            "are you artificial intelligence?",
        ],
        [
            "yes. i am a small AI model.",
            "yes. i am an artificial intelligence model.",
        ],
    ),
    (
        [
            "do you run locally?",
            "where do you run?",
            "are you local?",
        ],
        [
            "yes. i run locally on your computer.",
            "i am designed to run locally.",
        ],
    ),
    (
        [
            "do you need internet?",
            "can you work offline?",
            "can you run without internet?",
        ],
        [
            "no. i can run without an internet connection.",
            "i can generate text locally without internet.",
        ],
    ),
    (
        [
            "were you trained from scratch?",
            "did your weights start randomly?",
        ],
        [
            "yes. my weights started randomly and were trained from scratch.",
        ],
    ),
    (
        [
            "did you use pretrained weights?",
            "are your weights pretrained?",
        ],
        [
            "no. i was not initialized from pretrained model weights.",
            "no pretrained weights were used to initialize me.",
        ],
    ),
    (
        [
            "what is SimpleAI's Studio?",
            "what is SimpleAI?",
        ],
        [
            "SimpleAI's Studio is a project for building small local AI models.",
            "SimpleAI's Studio builds small models from scratch.",
        ],
    ),
]


for questions, answers in identity_training:
    for question in questions:
        for answer in answers:
            add_wrapped(
                "identity",
                question,
                answer,
            )


# -------------------------------------------------
# KNOWLEDGE TRAINING
# -------------------------------------------------

knowledge_forms = [
    "what is {thing}?",
    "tell me about {thing}.",
    "explain {thing}.",
    "describe {thing}.",
    "what can you tell me about {thing}?",
]


for thing, answer, keywords in knowledge:
    for form in knowledge_forms:
        add_wrapped(
            "knowledge",
            form.format(
                thing=thing
            ),
            answer,
        )

    add_qa(
        "knowledge",
        f"is this true: {answer}",
        "yes.",
    )


# -------------------------------------------------
# READING TRAINING
# -------------------------------------------------

for _ in range(9000):
    name = rng.choice(
        names
    )

    item = rng.choice(
        objects
    )

    place = rng.choice(
        places
    )

    form = rng.randint(
        0,
        2,
    )

    if form == 0:
        question = (
            f"Read this: {name} put "
            f"the {item} on the "
            f"{place}. Where is the "
            f"{item}?"
        )

    elif form == 1:
        question = (
            f"Text: The {item} was "
            f"moved by {name} to the "
            f"{place}. Question: "
            f"where is the {item}?"
        )

    else:
        question = (
            f"{name} carried a "
            f"{item} and left it on "
            f"the {place}. Where did "
            f"{name} leave the "
            f"{item}?"
        )

    add_qa(
        "reading",
        question,
        (
            f"the {item} is on "
            f"the {place}."
        ),
    )


# -------------------------------------------------
# LANGUAGE TRAINING
# -------------------------------------------------

for word, plural in plurals.items():
    forms = [
        f"plural of {word}?",
        f"make {word} plural.",
        f"give the plural form of {word}.",
    ]

    for question in forms:
        add_wrapped(
            "language",
            question,
            plural,
        )


for word, opposite in opposites.items():
    forms = [
        f"opposite of {word}?",
        f"give the opposite of {word}.",
        (
            f"what word means the "
            f"opposite of {word}?"
        ),
    ]

    for question in forms:
        add_wrapped(
            "language",
            question,
            opposite,
        )


# -------------------------------------------------
# ARITHMETIC TRAINING
# -------------------------------------------------

add_forms = [
    "what is {a} + {b}?",
    "what is {a} plus {b}?",
    "add {a} and {b}.",
    "calculate {a} + {b}.",
]


subtract_forms = [
    "what is {a} - {b}?",
    "what is {a} minus {b}?",
    "subtract {b} from {a}.",
    "calculate {a} - {b}.",
]


multiply_forms = [
    "what is {a} times {b}?",
    "what is {a} multiplied by {b}?",
    "multiply {a} and {b}.",
    "calculate {a} * {b}.",
]


divide_forms = [
    "what is {value} divided by {divisor}?",
    "calculate {value} / {divisor}.",
    "divide {value} by {divisor}.",
]


for _ in range(16000):
    operation = rng.randrange(
        4
    )

    if operation == 0:
        a = rng.randint(
            0,
            100,
        )

        b = rng.randint(
            0,
            100,
        )

        question = rng.choice(
            add_forms
        ).format(
            a=a,
            b=b,
        )

        answer = a + b

    elif operation == 1:
        a = rng.randint(
            0,
            100,
        )

        b = rng.randint(
            0,
            a,
        )

        question = rng.choice(
            subtract_forms
        ).format(
            a=a,
            b=b,
        )

        answer = a - b

    elif operation == 2:
        a = rng.randint(
            0,
            25,
        )

        b = rng.randint(
            0,
            25,
        )

        question = rng.choice(
            multiply_forms
        ).format(
            a=a,
            b=b,
        )

        answer = a * b

    else:
        divisor = rng.randint(
            1,
            20,
        )

        answer = rng.randint(
            0,
            20,
        )

        value = (
            divisor
            * answer
        )

        question = rng.choice(
            divide_forms
        ).format(
            value=value,
            divisor=divisor,
        )

    add_qa(
        "arithmetic",
        question,
        str(answer),
    )


# -------------------------------------------------
# LOGIC TRAINING
# -------------------------------------------------

for _ in range(12000):
    kind = rng.randrange(
        4
    )

    if kind == 0:
        number = rng.randint(
            0,
            300,
        )

        parity = (
            "even"
            if number % 2 == 0
            else "odd"
        )

        add_qa(
            "logic",
            f"is {number} {parity}?",
            "yes.",
        )

    elif kind == 1:
        number = rng.randint(
            0,
            300,
        )

        wrong = (
            "odd"
            if number % 2 == 0
            else "even"
        )

        add_qa(
            "logic",
            f"is {number} {wrong}?",
            "no.",
        )

    elif kind == 2:
        a = rng.randint(
            0,
            400,
        )

        b = rng.randint(
            0,
            400,
        )

        add_qa(
            "logic",
            f"is {a} greater than {b}?",
            (
                "yes."
                if a > b
                else "no."
            ),
        )

    else:
        a = rng.randint(
            0,
            400,
        )

        b = rng.randint(
            0,
            400,
        )

        if a > b:
            relation = "greater than"

        elif a < b:
            relation = "less than"

        else:
            relation = "equal to"

        add_qa(
            "logic",
            f"compare {a} and {b}.",
            (
                f"{a} is "
                f"{relation} {b}."
            ),
        )


# -------------------------------------------------
# SEQUENCE TRAINING
# -------------------------------------------------

training_steps = [
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


sequence_forms = [
    "what comes next: {sequence} ?",
    "find the next number: {sequence} ?",
    "continue: {sequence} ?",
]


for _ in range(10000):
    start = rng.randint(
        -50,
        100,
    )

    step = rng.choice(
        training_steps
    )

    values = [
        start + step * index
        for index in range(5)
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

    question = rng.choice(
        sequence_forms
    ).format(
        sequence=sequence
    )

    add_qa(
        "sequences",
        question,
        str(answer),
    )


# -------------------------------------------------
# CODING TRAINING
# -------------------------------------------------

coding_forms = [
    "what is {thing}?",
    "explain {thing}.",
    "define {thing}.",
    "tell me about {thing}.",
    "what does {thing} mean?",
]


for thing, answer, keywords in coding:
    for form in coding_forms:
        add_wrapped(
            "coding",
            form.format(
                thing=thing
            ),
            answer,
        )


# -------------------------------------------------
# STORY WORLD TRAINING
# -------------------------------------------------

for _ in range(10000):
    name = rng.choice(
        names
    )

    first_item, second_item = rng.sample(
        objects,
        2,
    )

    first_place, second_place = rng.sample(
        places,
        2,
    )

    kind = rng.randrange(
        3
    )

    if kind == 0:
        question = (
            f"{name} put the "
            f"{first_item} on the "
            f"{first_place}. Then "
            f"{name} moved the "
            f"{second_item} to the "
            f"{second_place}. "
            f"Where is the "
            f"{second_item}?"
        )

        answer = (
            f"the {second_item} is "
            f"on the {second_place}."
        )

    elif kind == 1:
        question = (
            f"The {first_item} is "
            f"on the {first_place}. "
            f"The {second_item} is "
            f"on the {second_place}. "
            f"Where is the "
            f"{first_item}?"
        )

        answer = (
            f"the {first_item} is "
            f"on the {first_place}."
        )

    else:
        question = (
            f"{name} left the "
            f"{first_item} on the "
            f"{first_place} and the "
            f"{second_item} on the "
            f"{second_place}. "
            f"Where is the "
            f"{second_item}?"
        )

        answer = (
            f"the {second_item} is "
            f"on the {second_place}."
        )

    add_qa(
        "story_world",
        question,
        answer,
    )


# -------------------------------------------------
# WRITE TRAIN / VALID
# -------------------------------------------------

manifest = {
    "version":
        "V3-C",

    "seed":
        seed,

    "benchmark_items":
        len(benchmark),

    "benchmark_in_training":
        False,

    "categories":
        {},
}


total_examples = 0
total_train = 0
total_valid = 0

total_train_bytes = 0
total_valid_bytes = 0


for category, values in categories.items():
    values = list(
        values
    )

    rng.shuffle(
        values
    )

    valid_count = max(
        1,
        int(
            len(values)
            * 0.10
        ),
    )

    valid_values = values[
        :valid_count
    ]

    train_values = values[
        valid_count:
    ]


    train_path = (
        train_dir
        / f"{category}.txt"
    )

    valid_path = (
        valid_dir
        / f"{category}.txt"
    )


    train_path.write_text(
        "\n\n".join(
            train_values
        )
        + "\n",
        encoding="utf-8",
    )

    valid_path.write_text(
        "\n\n".join(
            valid_values
        )
        + "\n",
        encoding="utf-8",
    )


    train_bytes = (
        train_path.stat().st_size
    )

    valid_bytes = (
        valid_path.stat().st_size
    )


    manifest[
        "categories"
    ][category] = {
        "total":
            len(values),

        "train":
            len(train_values),

        "valid":
            len(valid_values),

        "train_bytes":
            train_bytes,

        "valid_bytes":
            valid_bytes,
    }


    total_examples += len(
        values
    )

    total_train += len(
        train_values
    )

    total_valid += len(
        valid_values
    )

    total_train_bytes += (
        train_bytes
    )

    total_valid_bytes += (
        valid_bytes
    )


benchmark_path.write_text(
    json.dumps(
        benchmark,
        indent=2,
    ),
    encoding="utf-8",
)


manifest[
    "total_examples"
] = total_examples

manifest[
    "total_train_examples"
] = total_train

manifest[
    "total_valid_examples"
] = total_valid

manifest[
    "total_train_bytes"
] = total_train_bytes

manifest[
    "total_valid_bytes"
] = total_valid_bytes


manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


# -------------------------------------------------
# REPORT
# -------------------------------------------------

print()
print("Vex V3-C dataset")
print("----------------")
print()

print(
    f"Benchmark    : "
    f"{len(benchmark):,} untouched prompts"
)

print()

print(
    f"{'Category':<15}"
    f"{'Total':>8}"
    f"{'Train':>8}"
    f"{'Valid':>8}"
)

print(
    "-" * 39
)


for category, info in (
    manifest[
        "categories"
    ].items()
):
    print(
        f"{category:<15}"
        f"{info['total']:>8,}"
        f"{info['train']:>8,}"
        f"{info['valid']:>8,}"
    )


print(
    "-" * 39
)

print(
    f"{'TOTAL':<15}"
    f"{total_examples:>8,}"
    f"{total_train:>8,}"
    f"{total_valid:>8,}"
)

print()

print(
    f"Train size   : "
    f"{total_train_bytes / 1024 / 1024:.2f} MB"
)

print(
    f"Valid size   : "
    f"{total_valid_bytes / 1024 / 1024:.2f} MB"
)

print()

print(
    f"Data folder  : "
    f"{output_dir}"
)

print(
    f"Benchmark    : "
    f"{benchmark_path}"
)

print(
    f"Manifest     : "
    f"{manifest_path}"
)

print()
print(
    "Benchmark prompts were excluded "
    "from training."
)

print(
    "V3-C is ready for the "
    "category-balanced trainer."
)

print()