import numpy as np
import torch

from .base import BaseSolver


class EDMSolver(BaseSolver):
    def __init__(self, sigma_min, sigma_max, rho):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(self, net, noise, num_steps, device='cuda', 
                 labels=None, randn_like=torch.randn_like,
                 S_churn=80, S_min=0.05, S_max=1.0, S_noise=1.007):
        
        sigma_min = max(self.sigma_min, net.sigma_min)
        sigma_max = min(self.sigma_max, net.sigma_max)

        t_steps = self.get_timesteps(
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=self.rho
        )
        x = noise * self.sigma_max

        with torch.no_grad():
            for i in range(len(t_steps) - 1):
                t_cur = t_steps[i]
                t_next = t_steps[i + 1]
                
                gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
                t_hat_cur = (t_cur + gamma * t_cur)
                
                t_hat_cur_net = t_hat_cur * torch.ones(x.shape[0], device=device)
                t_next_net = t_next * torch.ones(x.shape[0], device=device)
                delta_t = (t_hat_cur - t_next).abs()
                
                x_hat = x + (t_hat_cur ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x)   
                v_hat = self.velocity_from_denoiser(x_hat, net, t_hat_cur_net, labels)
                x_next = x_hat + delta_t * v_hat
            
                if i < num_steps - 1:
                    v_next = self.velocity_from_denoiser(x_next, net, t_next_net, labels)
                    x_next = x_hat + delta_t * (0.5 * v_hat + 0.5 * v_next)

                x = x_next
            
        return x


    def get_name(self):
        return "EDM"
