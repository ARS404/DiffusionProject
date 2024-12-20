import numpy as np
import torch

from .base import BaseSolver


class EDMSolver(BaseSolver):
    def __init__(self, sigma_min, sigma_max, rho):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(self, 
                 net, latents, class_labels=None, randn_like=torch.randn_like,
                 num_steps=18,
                 S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
        
        sigma_min = max(self.sigma_min, net.sigma_min)
        sigma_max = min(self.sigma_max, net.sigma_max)

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (num_steps - 1) * 
                   (sigma_min ** (1 / self.rho) - sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        x_next = latents.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next


    def get_name(self):
        return "EDM"
