import torch
from torch import nn, Tensor
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + h, h),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, h),
            nn.ELU(),
            nn.Linear(h, dim),
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat((x_t, t), -1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1).expand(x_t.shape[0], 1)
        # For simplicity, using midpoint ODE solver in this example
        return x_t + self(x_t, t_start) * (t_end - t_start) / 2

# training
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()

for epoch in range(10):
    x_l = Tensor(make_moons(256, noise=0.05)[0])
    t = torch.rand(len(x_l), 1)
    x_t = x_l + t * 0.1  # small perturbation
    optimizer.zero_grad()
    loss = loss_fn(flow(x_t, t), x_l)
    loss.backward()
    optimizer.step()

# sampling
x = torch.randn(300, 2)
n_steps = 8
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
time_steps = torch.linspace(0.0, 1.0, n_steps + 1)

axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
axes[0].set_title(f'time_steps[0]: {time_steps[0]:.2f}')

for i in range(1, n_steps):
    x = flow.step(x, time_steps[i], time_steps[i + 1])
    axes[i].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[i].set_title(f'time_steps[i]: {time_steps[i]:.2f}')

plt.tight_layout()
plt.show()
