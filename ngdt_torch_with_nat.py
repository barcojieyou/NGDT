# ngdt_torch_with_nat.py
import torch
from torch.optim import Optimizer

class NGDTorch(Optimizer):
    def __init__(self, params, lr=1.0, Q_budget=1e-3, damping=1e-3,
                 eta_min=1e-6, eta_max=1.0, beta_f=0.95, beta_mom=0.9,
                 eta_null_ratio=1e-3, eps=1e-8, fisher_method='diag'):
        defaults = dict(lr=lr, Q_budget=Q_budget, damping=damping,
                        eta_min=eta_min, eta_max=eta_max, beta_f=beta_f,
                        beta_mom=beta_mom, eta_null_ratio=eta_null_ratio,
                        eps=eps, fisher_method=fisher_method)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum_nat'] = torch.zeros_like(p.data)
                # optional: keep fisher_ema if you want local fallback
                self.state[p]['fisher_ema'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step_with_nat_grads(self, g_nat_list, grads=None, closure=None, return_diagnostics=False):
        """
        Apply NGD-T update using externally computed natural gradients.

        Args:
            g_nat_list: list of tensors or None aligned with optimizer.param_groups order.
                        Each element is the preconditioned natural gradient for the corresponding param.
            grads: optional list of raw gradients aligned with params (used for nullspace fallback).
            closure: optional closure to recompute loss.
            return_diagnostics: if True, returns dict with Delta_F and eta_T.

        Behavior:
            - computes Delta_F = sum_p (g_p . g_nat_p)
            - computes eta_T from Q_budget and lr (eta0)
            - applies natural-space momentum and updates params:
                p <- p - eta_T * m_nat - eta_null * grad
        """
        loss = None
        if closure is not None:
            loss = closure()

        # flatten params in same order as g_nat_list
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)

        # compute Delta_F
        Delta_F = 0.0
        for p, g_nat in zip(params, g_nat_list):
            if g_nat is None:
                continue
            # ensure same device/dtype
            Delta_F += float(( (p.grad.detach() if p.grad is not None else torch.zeros_like(p.data)) * g_nat ).sum().cpu().item()) \
                       if (p.grad is not None) else float((g_nat * g_nat).sum().cpu().item())

        # regulator (single global eta_T per param group; use first group's settings)
        group = self.param_groups[0]
        Q_budget = group['Q_budget']
        eta0 = group['lr']
        eps = group['eps']
        eta_min = group['eta_min']
        eta_max = group['eta_max']
        eta_T = eta0 * (Q_budget / (Delta_F + eps))
        eta_T = max(min(eta_T, eta_max), eta_min)

        # apply updates with natural-space momentum and small nullspace fallback
        idx = 0
        for group in self.param_groups:
            beta_mom = group['beta_mom']
            eta_null_ratio = group['eta_null_ratio']
            for p in group['params']:
                g_nat = g_nat_list[idx]
                raw_grad = None
                if grads is not None:
                    raw_grad = grads[idx]
                else:
                    raw_grad = p.grad if p.grad is not None else None
                idx += 1

                if g_nat is None:
                    # no preconditioner for this param: fallback to small SGD step if grad exists
                    if raw_grad is not None:
                        p.data.add_(-eta_T * raw_grad)
                    continue

                state = self.state[p]
                m = state['momentum_nat']
                # natural-space momentum update
                m.mul_(beta_mom).add_(g_nat * (1.0 - beta_mom))
                eta_null = eta_null_ratio * eta_T
                # apply parameter update: natural step + small Euclidean fallback
                if raw_grad is None:
                    p.data.add_(-eta_T * m)
                else:
                    p.data.add_(-eta_T * m - eta_null * raw_grad)

        diagnostics = {'Delta_F': Delta_F, 'eta_T': eta_T, 'Q_pred': 0.5 * eta_T * eta_T * Delta_F}
        if return_diagnostics:
            return loss, diagnostics
        return loss
