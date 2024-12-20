import torch

from abc import abstractmethod


class BaseSolver(object):
    def __call__(self, net, *args, **kwds):
        raise NotImplementedError()
    
    def normalize(self, x):
        return x / x.abs().max(dim=0)[0][None, ...]
    
    def get_timesteps(self, sigma_min=0.02, sigma_max=80.0, num_steps=20, device='cuda', rho=7.0):
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        return t_steps
    
    def velocity_from_denoiser(self, x, net, sigma, class_labels=None, error_eps=1e-4):
        sigma = sigma[:, None, None, None]
        v = (net(x, sigma, class_labels) - x) / (sigma + error_eps)
        return v

    def get_name(self):
        return "I have no name :("