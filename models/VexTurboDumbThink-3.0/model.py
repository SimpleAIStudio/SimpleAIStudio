import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(
        self,
        hidden,
        heads,
        dropout,
    ):
        super().__init__()

        if hidden % heads != 0:
            raise ValueError(
                "hidden must divide evenly by heads"
            )

        self.heads = heads
        self.head_size = hidden // heads
        self.dropout = dropout

        self.qkv = nn.Linear(
            hidden,
            hidden * 3,
        )

        self.proj = nn.Linear(
            hidden,
            hidden,
        )

    def forward(self, x):
        batch, length, hidden = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        q = q.view(
            batch,
            length,
            self.heads,
            self.head_size,
        ).transpose(1, 2)

        k = k.view(
            batch,
            length,
            self.heads,
            self.head_size,
        ).transpose(1, 2)

        v = v.view(
            batch,
            length,
            self.heads,
            self.head_size,
        ).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        out = out.transpose(
            1,
            2,
        ).contiguous()

        out = out.view(
            batch,
            length,
            hidden,
        )

        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden,
        middle,
        dropout,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            hidden,
            middle,
        )

        self.fc2 = nn.Linear(
            middle,
            hidden,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return self.dropout(x)


class Block(nn.Module):
    def __init__(
        self,
        hidden,
        heads,
        middle,
        dropout,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            hidden
        )

        self.attention = Attention(
            hidden,
            heads,
            dropout,
        )

        self.norm2 = nn.LayerNorm(
            hidden
        )

        self.ffn = FeedForward(
            hidden,
            middle,
            dropout,
        )

    def forward(self, x):
        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.ffn(
            self.norm2(x)
        )

        return x


class Vex(nn.Module):
    def __init__(
        self,
        vocab_size=256,
        context=256,
        hidden=96,
        heads=4,
        middle=384,
        layers=4,
        dropout=0.05,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context = context

        self.hidden = hidden
        self.heads = heads
        self.middle = middle
        self.layers = layers
        self.dropout = dropout

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden,
        )

        self.position_embedding = nn.Embedding(
            context,
            hidden,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden,
                    heads,
                    middle,
                    dropout,
                )
                for _ in range(layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            hidden
        )

        self.apply(
            self._init_weights
        )

    def _init_weights(self, module):
        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        tokens,
        targets=None,
    ):
        batch, length = tokens.shape

        if length > self.context:
            raise ValueError(
                "Sequence exceeds context window."
            )

        positions = torch.arange(
            length,
            device=tokens.device,
        )

        x = (
            self.token_embedding(tokens)
            + self.position_embedding(
                positions
            )
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = F.linear(
            x,
            self.token_embedding.weight,
        )

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    self.vocab_size,
                ),
                targets.reshape(-1),
            )

        return logits, loss

    def parameter_count(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens=120,
        temperature=0.7,
        top_k=30,
    ):
        self.eval()

        for _ in range(
            max_new_tokens
        ):
            current = tokens[
                :,
                -self.context:
            ]

            logits, _ = self(
                current
            )

            logits = logits[
                :,
                -1,
                :
            ]

            logits = logits / max(
                temperature,
                0.01,
            )

            if top_k is not None:
                values, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1),
                    ),
                )

                cutoff = values[
                    :,
                    -1:
                ]

                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
                )

            probabilities = F.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            tokens = torch.cat(
                (
                    tokens,
                    next_token,
                ),
                dim=1,
            )

        return tokens