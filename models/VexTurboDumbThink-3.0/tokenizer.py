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
        data = bytes(
            max(0, min(255, int(token)))
            for token in tokens
        )

        return data.decode(
            "utf-8",
            errors="replace",
        )