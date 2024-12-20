import torch

from .base import BaseSolver


class DPMSolver(BaseSolver):
    def __init__(self, t_start, t_end, skip_type):
        super().__init__()
        self.t_start = t_start
        self.t_end = t_end

        self.skip_type = skip_type

    def __call__(self, net, noise, num_steps, labels=None, device='cuda'):
        timesteps = self.get_time_steps(t_T=self.t_start, t_0=self.t_end, N=num_steps, device=device)
        # Init the initial values.
        step = 0
        t = timesteps[step]
        t_prev_list = [t]
        model_prev_list = [self.model_fn(x, t)]
        # Init the first `order` values by lower order multistep DPM-Solver.
        for step in range(1, self.order):
            t = timesteps[step]
            x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step)
            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step)
            t_prev_list.append(t)
            model_prev_list.append(self.model_fn(x, t))
        # Compute the remaining values by `order`-th order multistep DPM-Solver.
        for step in range(self.order, num_steps + 1):
            t = timesteps[step]
            # We only use lower order for steps < 10
            step_order = self.order
            x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order)
            for i in range(self.order - 1):
                t_prev_list[i] = t_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            t_prev_list[-1] = t
            # We do not need to evaluate the final model value.
            if step < num_steps:
                model_prev_list[-1] = self.model_fn(x, t)

        return x

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order):
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t)

    def get_time_steps(self, t_T, t_0, N, device):
        if self.skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif self.skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif self.skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                sigma_t / sigma_s * x
                - alpha_t * phi_1 * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t):
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        phi_1 = torch.expm1(-h)
        x_t = (
            (sigma_t / sigma_prev_0) * x
            - (alpha_t * phi_1) * model_prev_0
            - 0.5 * (alpha_t * phi_1) * D1_0
        )
        return x_t

    def get_name(self):
        return f"DPM"