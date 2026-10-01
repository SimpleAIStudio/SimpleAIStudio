import time

import torch

from model import Vex


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )


device = "cuda"

batch_size = 128
context = 256

warmup_steps = 10
test_steps = 100


torch.manual_seed(3)


model = Vex(
    vocab_size=256,
    context=256,
    hidden=96,
    heads=4,
    middle=384,
    layers=4,
    dropout=0.05,
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
)


x = torch.randint(
    0,
    256,
    (
        batch_size,
        context,
    ),
    device=device,
)


y = torch.randint(
    0,
    256,
    (
        batch_size,
        context,
    ),
    device=device,
)


print()
print("Vex Turbo GPU test")
print("------------------")
print()

print(
    f"GPU        : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"Parameters : "
    f"{model.parameter_count():,}"
)

print(
    f"Context    : {context}"
)

print(
    f"Batch      : {batch_size}"
)

print()
print("Warming up...")


for _ in range(warmup_steps):
    _, loss = model(
        x,
        y,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()
    optimizer.step()


torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()

started = time.perf_counter()


for _ in range(test_steps):
    _, loss = model(
        x,
        y,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()


torch.cuda.synchronize()

elapsed = (
    time.perf_counter()
    - started
)


steps_per_second = (
    test_steps
    / elapsed
)

tokens_per_second = (
    test_steps
    * batch_size
    * context
    / elapsed
)

memory = (
    torch.cuda.max_memory_allocated()
    / 1024
    / 1024
)


print()
print("Done.")
print()

print(
    f"Time       : "
    f"{elapsed:.2f}s"
)

print(
    f"Speed      : "
    f"{steps_per_second:.2f} steps/s"
)

print(
    f"Tokens/sec : "
    f"{tokens_per_second:,.0f}"
)

print(
    f"Peak VRAM  : "
    f"{memory:.0f} MB"
)

print()