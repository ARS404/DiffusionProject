import torch

from .base import BaseSolver


class EulerSolver(BaseSolver):
    def __init__(self, net=None):
        super().__init__()        

    def __call__(
            self, noise, net, labels=None, sigma_min=0.02, 
            sigma_max=80.0, num_steps=20, device='cuda', 
            rho=7.0, stochastic=False, vis_steps=5
    ):
        t_steps = self.get_timesteps(sigma_min, sigma_max, num_steps, device, rho) # t_steps[0] = 80.0, t_steps[-2] = 0.02, t_steps[-1] = 0.0
        x = noise * sigma_max # стартуем с N(0, sigma^2_T)
        x_history = [self.normalize(noise)] # история для визуализации
        with torch.no_grad():
            for i in range(len(t_steps) - 1):
                t_cur = t_steps[i]
                t_next = t_steps[i + 1]
                t_net = t_steps[i] * torch.ones(x.shape[0], device=device)
                delta_t = (t_next - t_cur).abs()
                x = x + self.velocity_from_denoiser(x, net, t_net, class_labels=labels, stochastic=stochastic) * delta_t
                if stochastic:
                    x = x + torch.sqrt(2 * delta_t * t_cur) * torch.randn_like(x)

                x_history.append(self.normalize(x).view(-1, self.n_channels, *x.shape[2:]))

        # добавляем начало, конец, и выкидываем часть траектории
        x_history = [x_history[0]] + x_history[::-(num_steps // (vis_steps - 2))][::-1] + [x_history[-1]]
        return x, x_history
        

    def get_name(self):
        return "Euler"