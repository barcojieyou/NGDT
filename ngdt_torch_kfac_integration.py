# ngdt_torch_kfac_integration.py
import torch
import torch.nn as nn
from torch.optim import Optimizer

# --- NGDT Torch optimizer with adapter integration ---
class NGDTorch(Optimizer):
    def __init__(self, params, lr=1.0, Q_budget=1e-3, damping=1e-3,
                 eta_min=1e-6, eta_max=1.0, beta_mom=0.9, eta_null_ratio=1e-3, eps=1e-8):
        defaults = dict(lr=lr, Q_budget=Q_budget, damping=damping,
                        eta_min=eta_min, eta_max=eta_max, beta_mom=beta_mom,
                        eta_null_ratio=eta_null_ratio, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum_nat'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step_with_nat_grads(self, g_nat_list, raw_grads=None, closure=None, return_diagnostics=False):
        """
        Apply NGD-T update using externally computed natural gradients g_nat_list.
        g_nat_list must be aligned with optimizer.param_groups order.
        raw_grads optional for nullspace fallback.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # flatten params in same order as g_nat_list
        params = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)

        # compute Delta_F = sum_p (g_p . g_nat_p)
        Delta_F = 0.0
        for p, g_nat in zip(params, g_nat_list):
            if g_nat is None:
                continue
            g = p.grad if p.grad is not None else None
            if g is not None:
                Delta_F += float((g * g_nat).sum().cpu().item())
            else:
                Delta_F += float((g_nat * g_nat).sum().cpu().item())

        # regulator using first param group settings
        group = self.param_groups[0]
        Q_budget = group['Q_budget']
        eta0 = group['lr']
        eps = group['eps']
        eta_min = group['eta_min']
        eta_max = group['eta_max']
        eta_T = eta0 * (Q_budget / (Delta_F + eps))
        eta_T = max(min(eta_T, eta_max), eta_min)

        # apply updates with natural-space momentum and nullspace fallback
        idx = 0
        for group in self.param_groups:
            beta_mom = group['beta_mom']
            eta_null_ratio = group['eta_null_ratio']
            for p in group['params']:
                g_nat = g_nat_list[idx]
                raw_grad = None
                if raw_grads is not None:
                    raw_grad = raw_grads[idx]
                else:
                    raw_grad = p.grad if p.grad is not None else None
                idx += 1

                if g_nat is None:
                    if raw_grad is not None:
                        p.data.add_(-eta_T * raw_grad)
                    continue

                state = self.state[p]
                m = state['momentum_nat']
                m.mul_(beta_mom).add_(g_nat * (1.0 - beta_mom))
                eta_null = eta_null_ratio * eta_T
                if raw_grad is None:
                    p.data.add_(-eta_T * m)
                else:
                    p.data.add_(-eta_T * m - eta_null * raw_grad)

        diagnostics = {'Delta_F': Delta_F, 'eta_T': eta_T, 'Q_pred': 0.5 * eta_T * eta_T * Delta_F}
        if return_diagnostics:
            return loss, diagnostics
        return loss

# --- Minimal K-FAC adapter stub (linear and conv support) ---
class KFACAdapter:
    def __init__(self, model, ema_decay=0.95, damping=1e-3, eig_freq=20, device=None):
        self.model = model
        self.ema_decay = ema_decay
        self.damping = damping
        self.eig_freq = eig_freq
        self.step = 0
        self.device = device
        self.factors = {}
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.factors[m] = {'A': None, 'G': None, 'A_eig': None, 'G_eig': None}

    @staticmethod
    def _extract_patches(x, conv_module):
        return nn.functional.unfold(x, kernel_size=conv_module.kernel_size,
                                    dilation=conv_module.dilation,
                                    padding=conv_module.padding,
                                    stride=conv_module.stride)

    def attach_hooks(self):
        def forward_hook(module, inp, out):
            module._kfac_input = inp[0].detach()
        def backward_hook(module, grad_in, grad_out):
            module._kfac_grad_output = grad_out[0].detach()
        for m in self.factors.keys():
            m.register_forward_hook(forward_hook)
            m.register_full_backward_hook(backward_hook)

    def step_batch(self):
        self.step += 1
        for m, info in self.factors.items():
            x = getattr(m, '_kfac_input', None)
            grad_out = getattr(m, '_kfac_grad_output', None)
            if x is None or grad_out is None:
                continue
            if isinstance(m, nn.Linear):
                A = (x.t() @ x) / x.shape[0]
                G = (grad_out.t() @ grad_out) / grad_out.shape[0]
            else:
                patches = self._extract_patches(x, m)  # (N, K, L)
                N, K, L = patches.shape
                patches2 = patches.permute(0, 2, 1).reshape(N * L, K)
                A = (patches2.t() @ patches2) / patches2.shape[0]
                go = grad_out.permute(0, 2, 3, 1).reshape(N * L, -1)
                G = (go.t() @ go) / go.shape[0]
            if info['A'] is None:
                info['A'] = A.detach().to(self.device) if self.device else A.detach()
                info['G'] = G.detach().to(self.device) if self.device else G.detach()
            else:
                info['A'].mul_(self.ema_decay).add_(A.detach() * (1.0 - self.ema_decay))
                info['G'].mul_(self.ema_decay).add_(G.detach() * (1.0 - self.ema_decay))
            if (self.step % self.eig_freq) == 0:
                a_vals, a_vecs = torch.linalg.eigh(info['A'])
                g_vals, g_vecs = torch.linalg.eigh(info['G'])
                a_vals = torch.clamp(a_vals, min=1e-6)
                g_vals = torch.clamp(g_vals, min=1e-6)
                info['A_eig'] = (a_vals, a_vecs)
                info['G_eig'] = (g_vals, g_vecs)

    def apply_preconditioner(self, model, grads_list):
        g_nat_list = []
        for p in grads_list:
            if p is None:
                g_nat_list.append(None)
                continue
            owner = None
            for m, info in self.factors.items():
                w = getattr(m, 'weight', None)
                if w is not None and w.shape == p.shape:
                    owner = m
                    break
            if owner is None:
                g_nat_list.append(p / (self.damping + 1e-6))
                continue
            info = self.factors[owner]
            if info.get('A_eig') is None or info.get('G_eig') is None:
                g_nat_list.append(p / (self.damping + 1e-6))
                continue
            a_vals, a_vecs = info['A_eig']
            g_vals, g_vecs = info['G_eig']
            if isinstance(owner, nn.Linear):
                grad_mat = p.view(owner.out_features, owner.in_features)
                G_inv = (g_vecs @ torch.diag(1.0 / (g_vals + self.damping)) @ g_vecs.t())
                A_inv = (a_vecs @ torch.diag(1.0 / (a_vals + self.damping)) @ a_vecs.t())
                nat = G_inv @ grad_mat @ A_inv
                g_nat_list.append(nat.reshape_as(p))
            else:
                out, inn, kh, kw = p.shape
                grad_mat = p.reshape(out, inn * kh * kw)
                G_inv = (g_vecs @ torch.diag(1.0 / (g_vals + self.damping)) @ g_vecs.t())
                A_inv = (a_vecs @ torch.diag(1.0 / (a_vals + self.damping)) @ a_vecs.t())
                nat = G_inv @ grad_mat @ A_inv
                g_nat_list.append(nat.reshape_as(p))
        return g_nat_list

# --- Minimal training loop example ---
def training_loop_example(model, dataloader, loss_fn, device='cpu'):
    model.to(device)
    adapter = KFACAdapter(model, device=device)
    adapter.attach_hooks()
    optimizer = NGDTorch(model.parameters(), lr=1.0, Q_budget=1e-3)
    for epoch in range(2):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            # update K-FAC factors from captured activations and grads
            adapter.step_batch()
            # collect raw grads
            grads_list = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
            # compute preconditioned natural gradients
            g_nat_list = adapter.apply_preconditioner(model, grads_list)
            # apply NGD-T step using adapter-provided natural gradients
            _, diag = optimizer.step_with_nat_grads(g_nat_list, raw_grads=grads_list, return_diagnostics=True)
            # optional: log diag['Delta_F'], diag['eta_T']
