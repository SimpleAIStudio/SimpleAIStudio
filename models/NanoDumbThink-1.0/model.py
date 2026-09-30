import torch
from torch import nn


TASKS = {
    "add": 0,
    "sub": 1,
    "gt": 2,
    "lt": 3,
    "next": 4,
}


class NanoDumbThink(nn.Module):
    def __init__(self, hidden=32, middle=48, think_steps=8):
        super().__init__()

        self.hidden = hidden
        self.middle = middle
        self.think_steps = think_steps

        self.task_embed = nn.Embedding(len(TASKS), 8)
        self.input_layer = nn.Linear(14, hidden)

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

    def forward(self, task, numbers, mask, return_states=False):
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
                state + update * 0.5
            )

            if return_states:
                states.append(
                    state.detach().clone()
                )

        choices = torch.cat(
            [
                head(state)
                for head in self.heads
            ],
            dim=1,
        )

        output = choices.gather(
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