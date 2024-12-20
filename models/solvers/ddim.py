import numpy as np
import torch

from .base import BaseSolver


class DDIMSolver(BaseSolver):
    def __init__(self, beta_start, beta_end):
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end

    def __call__(self, net, noise, num_steps, labels=None, device='cuda'):
        betas = self.get_betas(num_steps)
        alphas = torch.Tensor(1 - betas).cumprod(dim=0)
        x = noise.to(device)
        with torch.no_grad():
            for i in reversed(range(len(betas) - 1)):
                alpha_t = alphas[i + 1]
                alpha_prev = alphas[i]
                t_net = ((1 - alpha_t) / alpha_t).sqrt() * torch.ones(x.shape[0], device=device)[:, None, None, None]
                x0_hat = net(x / alpha_t.sqrt(), t_net, labels)
                eps_hat = (x - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
                x = alpha_prev.sqrt() * x0_hat + (1 - alpha_prev).sqrt() * eps_hat
        return x

    def get_betas(self, num_steps=20):
        betas = (
            torch.linspace(self.beta_start**0.5, self.beta_end**0.5, num_steps) ** 2
        )
        return betas.numpy()

    def get_name(self):
        return "DDM"