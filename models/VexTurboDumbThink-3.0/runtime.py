import math
from pathlib import Path

import numpy as np


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(
            text.encode(
                "utf-8",
                errors="replace",
            )
        )

    def decode(self, tokens):
        return bytes(
            max(0, min(255, int(token)))
            for token in tokens
        ).decode(
            "utf-8",
            errors="replace",
        )


def linear(x, weight, bias):
    return (
        x @ weight.T
        + bias
    )


def layer_norm(
    x,
    weight,
    bias,
    eps=1e-5,
):
    mean = x.mean(
        axis=-1,
        keepdims=True,
    )

    variance = (
        (
            x - mean
        ) ** 2
    ).mean(
        axis=-1,
        keepdims=True,
    )

    x = (
        x - mean
    ) / np.sqrt(
        variance + eps
    )

    return (
        x * weight
        + bias
    )


def erf_approx(x):
    sign = np.sign(x)
    x = np.abs(x)

    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (
        1.0 + p * x
    )

    y = 1.0 - (
        (
            (
                (
                    (
                        a5 * t
                        + a4
                    ) * t
                    + a3
                ) * t
                + a2
            ) * t
            + a1
        )
        * t
        * np.exp(
            -(x * x)
        )
    )

    return sign * y


def gelu(x):
    return (
        0.5
        * x
        * (
            1.0
            + erf_approx(
                x
                / math.sqrt(2.0)
            )
        )
    )


def softmax(x):
    x = (
        x
        - np.max(
            x,
            axis=-1,
            keepdims=True,
        )
    )

    values = np.exp(x)

    return (
        values
        / values.sum(
            axis=-1,
            keepdims=True,
        )
    )


class VexRuntime:
    def __init__(
        self,
        model_dir=None,
    ):
        if model_dir is None:
            model_dir = (
                Path(__file__).parent
            )

        self.model_dir = Path(
            model_dir
        )

        self.weights = np.load(
            self.model_dir
            / "weights.npz"
        )

        self.tokenizer = (
            ByteTokenizer()
        )

        self.context = 256
        self.hidden = 96
        self.heads = 4
        self.head_size = 24
        self.layers = 4

    def _attention(
        self,
        x,
        block,
    ):
        prefix = (
            f"blocks.{block}."
            f"attention."
        )

        qkv = linear(
            x,
            self.weights[
                prefix
                + "qkv.weight"
            ],
            self.weights[
                prefix
                + "qkv.bias"
            ],
        )

        q, k, v = np.split(
            qkv,
            3,
            axis=-1,
        )

        length = x.shape[0]

        q = q.reshape(
            length,
            self.heads,
            self.head_size,
        ).transpose(
            1,
            0,
            2,
        )

        k = k.reshape(
            length,
            self.heads,
            self.head_size,
        ).transpose(
            1,
            0,
            2,
        )

        v = v.reshape(
            length,
            self.heads,
            self.head_size,
        ).transpose(
            1,
            0,
            2,
        )

        scores = (
            q
            @ k.transpose(
                0,
                2,
                1,
            )
        )

        scores = (
            scores
            / math.sqrt(
                self.head_size
            )
        )

        mask = np.triu(
            np.ones(
                (
                    length,
                    length,
                ),
                dtype=bool,
            ),
            k=1,
        )

        scores = np.where(
            mask[None, :, :],
            -1e30,
            scores,
        )

        weights = softmax(
            scores
        )

        out = weights @ v

        out = out.transpose(
            1,
            0,
            2,
        ).reshape(
            length,
            self.hidden,
        )

        return linear(
            out,
            self.weights[
                prefix
                + "proj.weight"
            ],
            self.weights[
                prefix
                + "proj.bias"
            ],
        )

    def _feed_forward(
        self,
        x,
        block,
    ):
        prefix = (
            f"blocks.{block}."
            f"ffn."
        )

        x = linear(
            x,
            self.weights[
                prefix
                + "fc1.weight"
            ],
            self.weights[
                prefix
                + "fc1.bias"
            ],
        )

        x = gelu(x)

        return linear(
            x,
            self.weights[
                prefix
                + "fc2.weight"
            ],
            self.weights[
                prefix
                + "fc2.bias"
            ],
        )

    def forward(
        self,
        tokens,
    ):
        tokens = np.asarray(
            tokens,
            dtype=np.int64,
        )

        if len(tokens) > self.context:
            tokens = tokens[
                -self.context:
            ]

        positions = np.arange(
            len(tokens),
            dtype=np.int64,
        )

        x = (
            self.weights[
                "token_embedding.weight"
            ][tokens]
            + self.weights[
                "position_embedding.weight"
            ][positions]
        )

        for block in range(
            self.layers
        ):
            norm1 = layer_norm(
                x,
                self.weights[
                    f"blocks.{block}."
                    f"norm1.weight"
                ],
                self.weights[
                    f"blocks.{block}."
                    f"norm1.bias"
                ],
            )

            x = (
                x
                + self._attention(
                    norm1,
                    block,
                )
            )

            norm2 = layer_norm(
                x,
                self.weights[
                    f"blocks.{block}."
                    f"norm2.weight"
                ],
                self.weights[
                    f"blocks.{block}."
                    f"norm2.bias"
                ],
            )

            x = (
                x
                + self._feed_forward(
                    norm2,
                    block,
                )
            )

        x = layer_norm(
            x,
            self.weights[
                "final_norm.weight"
            ],
            self.weights[
                "final_norm.bias"
            ],
        )

        logits = (
            x
            @ self.weights[
                "token_embedding.weight"
            ].T
        )

        return logits

    def generate_tokens(
        self,
        prompt,
        max_new_tokens=120,
        temperature=0.0,
        top_k=20,
        seed=None,
    ):
        tokens = (
            self.tokenizer.encode(
                prompt
            )
        )

        original_length = len(
            tokens
        )

        rng = np.random.default_rng(
            seed
        )

        for _ in range(
            max_new_tokens
        ):
            current = tokens[
                -self.context:
            ]

            logits = self.forward(
                current
            )[-1]

            if temperature <= 0:
                next_token = int(
                    np.argmax(
                        logits
                    )
                )

            else:
                logits = (
                    logits
                    / temperature
                )

                if top_k is not None:
                    k = min(
                        top_k,
                        len(logits),
                    )

                    indexes = np.argpartition(
                        logits,
                        -k,
                    )[-k:]

                    chosen_logits = (
                        logits[indexes]
                    )

                    probabilities = (
                        softmax(
                            chosen_logits
                        )
                    )

                    next_token = int(
                        rng.choice(
                            indexes,
                            p=probabilities,
                        )
                    )

                else:
                    probabilities = (
                        softmax(
                            logits
                        )
                    )

                    next_token = int(
                        rng.choice(
                            len(logits),
                            p=probabilities,
                        )
                    )

            tokens.append(
                next_token
            )

            generated = (
                self.tokenizer.decode(
                    tokens[
                        original_length:
                    ]
                )
            )

            if (
                "\nUser:"
                in generated
                or "\n\n"
                in generated
            ):
                break

        return tokens[
            original_length:
        ]

    def respond(
        self,
        message,
        sampling=False,
    ):
        prompt = (
            f"User: {message}\n"
            f"Vex:"
        )

        generated = (
            self.generate_tokens(
                prompt,
                max_new_tokens=120,
                temperature=(
                    0.65
                    if sampling
                    else 0.0
                ),
                top_k=20,
            )
        )

        response = (
            self.tokenizer.decode(
                generated
            )
        )

        for marker in [
            "\nUser:",
            "\nVex:",
            "\n\n",
        ]:
            if marker in response:
                response = (
                    response.split(
                        marker,
                        1,
                    )[0]
                )

        return response.strip()