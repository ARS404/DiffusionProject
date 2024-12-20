import torch

from abc import abstractmethod


class BaseSolver(object):
    def __call__(self, net, *args, **kwds):
        raise NotImplementedError()
    
    @abstractmethod
    def normalize(x):
        return x / x.abs().max(dim=0)[0][None, ...]
    
    @abstractmethod
    def get_timesteps(sigma_min=0.02, sigma_max=80.0, num_steps=20, device='cuda', rho=7.0):
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        return t_steps
    
    @abstractmethod
    def velocity_from_denoiser(x, net, sigma, class_labels=None, error_eps=1e-4, stochastic=False):
        sigma = sigma[:, None, None, None] # [batch, 1, 1, 1]
        v = (net(x, sigma, class_labels) - x) / (sigma + error_eps)

        if stochastic:
            v = v * 2

        return v

    def get_name(self):
        return "I have no name :("