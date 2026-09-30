import torch
from torch import nn


TASKS = {
    "add": 0,
    "sub": 1,
    "mul": 2,
    "div": 3,
    "gt": 4,
    "lt": 5,
    "next": 6,
    "sum3": 7,
    "max3": 8,
    "chain_gt": 9,
}


class MicroJet(nn.Module):
    def __init__(
        self,
        hidden=64,
        middle=96,
        think_steps=12,
    ):
        super().__init__()

        self.hidden = hidden
        self.middle = middle
        self.think_steps = think_steps

        self.task_embed = nn.Embedding(
            len(TASKS),
            12,
        )

        # task embedding + 5 values + 5 masks
        self.input_layer = nn.Linear(
            22,
            hidden,
        )

        self.brain = nn.Sequential(
            nn.Linear(hidden, middle),
            nn.Tanh(),
            nn.Linear(middle, hidden),
        )

        self.norm = nn.LayerNorm(hidden)

        self.heads = nn.ModuleList(
            nn.Linear(hidden, 1)
            for _ in TASKS
        )

    def forward(
        self,
        task,
        numbers,
        mask,
        return_states=False,
    ):
        task_vec = self.task_embed(task)

        x = torch.cat(
            [
                task_vec,
                numbers / 20.0,
                mask,
            ],
            dim=1,
        )

        state = torch.tanh(
            self.input_layer(x)
        )

        states = []

        for _ in range(self.think_steps):
            update = self.brain(state)

            state = self.norm(
                state + update * 0.35
            )

            if return_states:
                states.append(
                    state.detach().clone()
                )

        outputs = torch.cat(
            [
                head(state)
                for head in self.heads
            ],
            dim=1,
        )

        output = outputs.gather(
            1,
            task.unsqueeze(1),
        ).squeeze(1)

        if return_states:
            return output, states

        return output

    def parameter_count(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )