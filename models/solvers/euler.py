import torch

from .base import BaseSolver


class EulerSolver(BaseSolver):
    def __init__(self, n_channels=3):
        super().__init__()        
        self.n_channels = n_channels

    def __call__(
            self, net, noise, labels=None, sigma_min=0.02, 
            sigma_max=80.0, num_steps=256, device='cuda', 
            rho=7.0, stochastic=False, vis_steps=5
    ):
        t_steps = self.get_timesteps(sigma_min, sigma_max, num_steps, device, rho)
        x = noise * sigma_max
        with torch.no_grad():
            for i in range(len(t_steps) - 1):
                t_cur = t_steps[i]
                t_next = t_steps[i + 1]
                t_net = t_steps[i] * torch.ones(x.shape[0], device=device)
                delta_t = (t_next - t_cur).abs()
                x = x + self.velocity_from_denoiser(x, net, t_net, class_labels=labels, stochastic=stochastic) * delta_t
                if stochastic:
                    x = x + torch.sqrt(2 * delta_t * t_cur) * torch.randn_like(x)
        return x
        

    def get_name(self):
        return "Euler"