# kfac_adapter.py
# Lightweight K-FAC adapter stub for PyTorch NGD-T
# Requires: torch, torch.nn.functional
import torch
import torch.nn as nn
import torch.nn.functional as F

class KFACAdapter:
    """
    Minimal K-FAC adapter:
      - supports nn.Linear and nn.Conv2d
      - maintains EMA of A (input covariance) and G (output-gradient covariance)
      - computes damped pseudoinverses via eigendecomposition
      - provides apply_preconditioner(model, grads) -> list of g_nat tensors aligned with model.parameters()
    Usage:
      adapter = KFACAdapter(model, ema_decay=0.95, damping=1e-3, diag_eps=1e-6, eig_freq=20)
      adapter.step_batch(inputs, outputs, loss, retain_graph=False)  # call during training loop
      g_nat_list = adapter.apply_preconditioner(model, grads_list)
    Note: This stub is synchronous and simple; production K-FAC uses block caching, distributed ops, and optimized solvers.
    """
    def __init__(self, model, ema_decay=0.95, damping=1e-3, diag_eps=1e-6, eig_freq=20, device=None):
        self.model = model
        self.ema_decay = ema_decay
        self.damping = damping
        self.diag_eps = diag_eps
        self.eig_freq = eig_freq
        self.step = 0
        self.device = device
        # store per-layer factors and cached inverses
        self.factors = {}   # key: module -> {'A': tensor, 'G': tensor, 'A_eig':(eigvals,eigvecs), 'G_eig':...}
        # register supported modules
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.factors[m] = {'A': None, 'G': None, 'A_eig': None, 'G_eig': None}

    @staticmethod
    def _extract_patches(x, conv_module):
        # x: input activations to conv layer (N,C,H,W)
        # returns patches shaped (N, C*kh*kw, L) where L is number of spatial locations
        kh, kw = conv_module.kernel_size
        stride = conv_module.stride
        padding = conv_module.padding
        # unfold returns (N, C*kh*kw, L)
        patches = F.unfold(x, kernel_size=(kh, kw), dilation=conv_module.dilation,
                           padding=padding, stride=stride)
        return patches  # (N, K, L)

    def step_batch(self, model_inputs, model_outputs, loss, retain_graph=False):
        """
        Call after forward and backward on a mini-batch.
        This function:
          - collects layer inputs and output-gradients (requires hooks or manual capture)
          - updates EMA of A and G for each supported layer
        For simplicity this stub expects the user to have attached forward/backward hooks that
        saved activations and output gradients on modules as attributes:
          module._kfac_input  (tensor)
          module._kfac_grad_output (tensor)
        See attach_hooks below for helper.
        """
        self.step += 1
        for m, info in self.factors.items():
            x = getattr(m, '_kfac_input', None)
            grad_out = getattr(m, '_kfac_grad_output', None)
            if x is None or grad_out is None:
                continue
            # compute A and G estimates
            if isinstance(m, nn.Linear):
                # x: (N, in_features)
                A = (x.t() @ x) / x.shape[0]  # (in,in)
                # grad_out: (N, out_features)
                G = (grad_out.t() @ grad_out) / grad_out.shape[0]  # (out,out)
            else:  # Conv2d
                # x: (N, C, H, W) saved before unfolding
                patches = self._extract_patches(x, m)  # (N, K, L)
                # reshape to (N*L, K)
                N, K, L = patches.shape
                patches2 = patches.permute(0, 2, 1).reshape(N * L, K)
                A = (patches2.t() @ patches2) / (patches2.shape[0])
                # grad_out: (N, out_channels, H_out, W_out)
                go = grad_out.permute(0, 2, 3, 1).reshape(N * L, -1)  # (N*L, out)
                G = (go.t() @ go) / (go.shape[0])

            # EMA update
            if info['A'] is None:
                info['A'] = A.detach().to(self.device) if self.device else A.detach()
                info['G'] = G.detach().to(self.device) if self.device else G.detach()
            else:
                info['A'].mul_(self.ema_decay).add_(A.detach() * (1.0 - self.ema_decay))
                info['G'].mul_(self.ema_decay).add_(G.detach() * (1.0 - self.ema_decay))

            # recompute eigendecompositions periodically
            if (self.step % self.eig_freq) == 0:
                # small symmetric eigendecomp (cpu/gpu)
                try:
                    a_vals, a_vecs = torch.linalg.eigh(info['A'])
                    g_vals, g_vecs = torch.linalg.eigh(info['G'])
                except Exception:
                    # fallback to svd for numerical robustness
                    a_vals, a_vecs = torch.symeig(info['A'], eigenvectors=True)
                    g_vals, g_vecs = torch.symeig(info['G'], eigenvectors=True)
                # clamp eigenvalues and store
                a_vals = torch.clamp(a_vals, min=self.diag_eps)
                g_vals = torch.clamp(g_vals, min=self.diag_eps)
                info['A_eig'] = (a_vals, a_vecs)
                info['G_eig'] = (g_vals, g_vecs)

    def apply_preconditioner(self, model, grads_list):
        """
        Given model and list of gradients aligned with model.parameters(),
        return list of preconditioned natural gradients g_nat aligned with same order.
        This uses cached eigendecompositions if available; otherwise falls back to diagonal approx.
        """
        g_nat_list = []
        param_iter = iter(model.parameters())
        # iterate modules in same order as factors to map params to modules
        # For Linear: weight, bias; Conv2d: weight, bias
        for p in grads_list:
            if p is None:
                g_nat_list.append(None)
                continue
            # find module owning this parameter (simple heuristic: match shapes)
            owner = None
            for m, info in self.factors.items():
                # check weight shape match
                w = getattr(m, 'weight', None)
                if w is not None and w.shape == p.shape:
                    owner = m
                    break
            if owner is None:
                # fallback to diagonal inverse
                # approximate F diag by mean of A and G diag if available
                g_nat_list.append(p / (self.damping + self.diag_eps))
                continue
            info = self.factors[owner]
            if info.get('A_eig') is None or info.get('G_eig') is None:
                # fallback to diagonal: use scalar factor
                g_nat_list.append(p / (self.damping + self.diag_eps))
                continue
            a_vals, a_vecs = info['A_eig']
            g_vals, g_vecs = info['G_eig']
            # compute inverse Kronecker product action: (A ⊗ G)^{-1} vec(grad) = vec(G^{-1} grad A^{-1})
            # reshape p to matrix form matching (out, in) for Linear; for Conv weight shape (out, in, kh, kw)
            if isinstance(owner, nn.Linear):
                grad_mat = p.view(owner.out_features, owner.in_features)
                # compute G^{-1} @ grad_mat @ A^{-1}
                G_inv = (g_vecs @ torch.diag(1.0 / g_vals) @ g_vecs.t())
                A_inv = (a_vecs @ torch.diag(1.0 / a_vals) @ a_vecs.t())
                nat = G_inv @ grad_mat @ A_inv
                g_nat_list.append(nat.reshape_as(p))
            else:
                # conv weight: (out, in, kh, kw) -> treat as (out, in*kh*kw)
                out, inn, kh, kw = p.shape
                grad_mat = p.reshape(out, inn * kh * kw)
                G_inv = (g_vecs @ torch.diag(1.0 / g_vals) @ g_vecs.t())
                A_inv = (a_vecs @ torch.diag(1.0 / a_vals) @ a_vecs.t())
                nat = G_inv @ grad_mat @ A_inv
                g_nat_list.append(nat.reshape_as(p))
        return g_nat_list

    @staticmethod
    def attach_hooks(model):
        """
        Attach forward/backward hooks to capture inputs and output-gradients on supported modules.
        After forward pass, module._kfac_input will be set.
        After backward, module._kfac_grad_output will be set.
        """
        def forward_hook(module, inp, out):
            # store input activations (detach)
            if isinstance(module, nn.Linear):
                module._kfac_input = inp[0].detach()
            elif isinstance(module, nn.Conv2d):
                module._kfac_input = inp[0].detach()
        def backward_hook(module, grad_in, grad_out):
            # grad_out is tuple; take first element
            module._kfac_grad_output = grad_out[0].detach()
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.register_forward_hook(forward_hook)
                m.register_full_backward_hook(backward_hook)
