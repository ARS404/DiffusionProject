import torch

from .base import BaseSolver


class EulerSolver(BaseSolver):
    def __init__(self, n_channels, sigma_min, sigma_max, rho):
        super().__init__()        
        self.n_channels = n_channels
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def __call__(
            self, net, noise, num_steps, labels=None, device='cuda'
    ):
        t_steps = self.get_timesteps(
            self.sigma_min, self.sigma_max, 
            num_steps, device, self.rho
        )
        x = noise * self.sigma_max
        for i in range(len(t_steps) - 1):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            t_net = t_steps[i] * torch.ones(x.shape[0], device=device)
            delta_t = (t_next - t_cur).abs()
            x = x + self.velocity_from_denoiser(x, net, t_net, class_labels=labels) * delta_t
        return x
        

    def get_name(self):
        return "Euler"