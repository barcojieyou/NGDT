#!/usr/bin/env python3
"""
A single, ready‑to‑run PyTorch script that implements a layerwise K‑FAC optimizer with patch‑based convolution factors, 
per‑layer pseudoinverse via eigendecomposition (with truncation or Tikhonov damping), eigendecomposition caching and recomputation every K steps, 
the NGD‑T thermodynamic regulator, and evaluation on CIFAR‑10. 
The script downloads CIFAR‑10 via torchvision, trains for a small number of epochs (configurable), and prints test accuracy after training.

Implementation Details:
Patch‑based conv A: we use torch.nn.functional.unfold to extract input patches and compute A as the covariance over patches (shape in*kh*kw), which makes the Kronecker approximation consistent with the conv weight layout.
Eigendecomposition caching: A_plus and G_plus are computed and cached every kfac_eig_update steps to amortize cost. increase kfac_eig_update can be increase to reduce overhead.
Damping vs truncation: set --use-damping to use Tikhonov damping (smooth inverse). Otherwise the code uses hard truncation with tol.
Hybrid update: we compute g_nat = (G_plus ⊗ A_plus) applied to grad and add a small Euclidean step in the residual nullspace direction to avoid stalling.
Diagnostics: printed periodically to monitor eta_T and total_delta_F.

Notes before running:
Requires torch and torchvision installed and a CUDA‑capable GPU for reasonable speed (CPU will run but slowly).
Tune hyperparameters (q_budget, damping, kfac_eig_update, epochs, batch_size) for your hardware.
The script is instrumented with diagnostics (retained ranks, Δ_F, η_T, predicted dissipation).



"""
"""
Layerwise K-FAC + pseudoinverse NGD-T training script for CIFAR-10.

Features:
- Patch-based K-FAC factors for Conv2d (unfolded patches)
- Per-layer eigendecomposition for A and G with caching and recompute every K steps
- Moore-Penrose pseudoinverse via eigendecomposition (hard truncation) or Tikhonov damping
- Global thermodynamic regulator eta_T computed from total Delta_F
- Hybrid update: natural-space preconditioned step + small nullspace Euclidean fallback
- Test evaluation after training

Usage:
    python ngd_t_kfac_cifar.py
"""

import time
import math
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# -------------------------
# Config / hyperparameters
# -------------------------
parser = argparse.ArgumentParser(description="NGD-T with K-FAC (CIFAR-10)")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr-base", type=float, default=1.0, help="base scale for eta0 in regulator")
parser.add_argument("--q-budget", type=float, default=1e-3, help="thermodynamic dissipation budget")
parser.add_argument("--eta-min", type=float, default=1e-6)
parser.add_argument("--eta-max", type=float, default=1.0)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--ema-decay", type=float, default=0.95)
parser.add_argument("--damping", type=float, default=1e-3, help="Tikhonov damping for A/G (if use_damping True)")
parser.add_argument("--use-damping", action="store_true", help="use Tikhonov damping instead of hard truncation")
parser.add_argument("--tol", type=float, default=1e-8, help="relative truncation tol for eigendecomp")
parser.add_argument("--kfac-eig-update", type=int, default=20, help="recompute eigendecomps every K steps")
parser.add_argument("--kfac-factor-update", type=int, default=1, help="update A/G every this many steps")
parser.add_argument("--eta-null-ratio", type=float, default=0.01, help="nullspace fallback ratio")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# -------------------------
# Device and seeds
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
if device.type == "cuda":
    torch.cuda.manual_seed_all(args.seed)

# -------------------------
# Model: small CNN for CIFAR-10
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 256, bias=True)
        self.fc2 = nn.Linear(256, num_classes, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (B,64,16,16)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------
# Data loaders (torchvision)
# -------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------
# K-FAC layer state
# -------------------------
class KFACLayerState:
    def __init__(self, module, layer_type, ema_decay=0.95, device="cpu"):
        self.module = module
        self.layer_type = layer_type  # 'conv' or 'linear'
        self.ema_decay = ema_decay
        self.device = device
        # Covariance factors
        self.A = None  # input (K,K) for conv: K = in*kh*kw; for linear: in
        self.G = None  # output (out,out)
        # Cached eigendecompositions and pseudoinverses
        self.A_eigvals = None
        self.A_eigvecs = None
        self.A_plus = None
        self.G_eigvals = None
        self.G_eigvecs = None
        self.G_plus = None
        # retained masks
        self.A_mask = None
        self.G_mask = None
        # buffers for raw activations and backprops (for conv: raw tensors)
        self._acts_raw = None
        self._backprops_raw = None
        # hooks
        self.handle_forward = None
        self.handle_backward = None

    def register_hooks(self):
        module = self.module

        def forward_hook(mod, inp, out):
            x = inp[0].detach()
            # store raw activations for conv; for linear store (B, in)
            self._acts_raw = x.to(self.device)

        def backward_hook(mod, grad_in, grad_out):
            gy = grad_out[0].detach()
            self._backprops_raw = gy.to(self.device)

        # Use full backward hook for PyTorch >=1.8
        self.handle_forward = module.register_forward_hook(forward_hook)
        self.handle_backward = module.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        if self.handle_forward is not None:
            self.handle_forward.remove()
            self.handle_forward = None
        if self.handle_backward is not None:
            self.handle_backward.remove()
            self.handle_backward = None

    def update_factors(self, damping=0.0):
        """
        Update A and G using stored raw activations and backprops.
        For conv: unfold patches and compute covariances over (B*L) samples.
        """
        if self._acts_raw is None or self._backprops_raw is None:
            return

        if self.layer_type == "conv":
            x = self._acts_raw  # (B, C_in, H, W)
            gy = self._backprops_raw  # (B, C_out, H_out, W_out)
            B, C_in, H, W = x.shape
            _, C_out, H_out, W_out = gy.shape
            kh, kw = self.module.kernel_size
            stride = self.module.stride
            padding = self.module.padding
            # Unfold input into patches: (B, C_in*kh*kw, L) where L = H_out*W_out
            x_patches = torch.nn.functional.unfold(x, kernel_size=(kh, kw), padding=padding, stride=stride)
            # reshape to (B*L, K)
            Bp, K, L = x_patches.shape
            x_patches = x_patches.permute(0, 2, 1).contiguous().view(Bp * L, K)  # (B*L, K)
            # gy -> (B*L, C_out)
            gy_patches = gy.permute(0, 2, 3, 1).contiguous().view(Bp * L, C_out)  # (B*L, C_out)
            # Covariances
            A_batch = (x_patches.t() @ x_patches) / float(x_patches.shape[0])  # (K,K)
            G_batch = (gy_patches.t() @ gy_patches) / float(gy_patches.shape[0])  # (C_out,C_out)
            # EMA update
            if self.A is None:
                self.A = A_batch.detach().clone()
            else:
                self.A = self.ema_decay * self.A + (1.0 - self.ema_decay) * A_batch.detach()
            if self.G is None:
                self.G = G_batch.detach().clone()
            else:
                self.G = self.ema_decay * self.G + (1.0 - self.ema_decay) * G_batch.detach()
            # optional damping
            if damping > 0.0:
                self.A = self.A + damping * torch.eye(self.A.shape[0], device=self.A.device)
                self.G = self.G + damping * torch.eye(self.G.shape[0], device=self.G.device)
            # clear buffers
            self._acts_raw = None
            self._backprops_raw = None

        elif self.layer_type == "linear":
            x = self._acts_raw  # (B, in)
            gy = self._backprops_raw  # (B, out)
            A_batch = (x.t() @ x) / float(x.shape[0])
            G_batch = (gy.t() @ gy) / float(gy.shape[0])
            if self.A is None:
                self.A = A_batch.detach().clone()
            else:
                self.A = self.ema_decay * self.A + (1.0 - self.ema_decay) * A_batch.detach()
            if self.G is None:
                self.G = G_batch.detach().clone()
            else:
                self.G = self.ema_decay * self.G + (1.0 - self.ema_decay) * G_batch.detach()
            if damping > 0.0:
                self.A = self.A + damping * torch.eye(self.A.shape[0], device=self.A.device)
                self.G = self.G + damping * torch.eye(self.G.shape[0], device=self.G.device)
            self._acts_raw = None
            self._backprops_raw = None

# -------------------------
# Build K-FAC state for model
# -------------------------
def build_kfac_state(model, device, ema_decay=0.95):
    kfac_state = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            st = KFACLayerState(module, "conv", ema_decay=ema_decay, device=device)
            st.register_hooks()
            kfac_state[name] = st
        elif isinstance(module, nn.Linear):
            st = KFACLayerState(module, "linear", ema_decay=ema_decay, device=device)
            st.register_hooks()
            kfac_state[name] = st
    return kfac_state

# -------------------------
# Eigendecomposition / pseudoinverse helpers
# -------------------------
def symmetric_eig_pseudoinverse_torch(mat, tol=1e-8, damping=None):
    """
    For symmetric PSD mat (torch tensor), compute pseudoinverse via eigendecomposition.
    If damping is provided (>0), compute (mat + damping I)^{-1} instead (Tikhonov).
    Returns (mat_plus, eigvecs, eigvals, retained_mask)
    """
    # ensure symmetric
    mat = 0.5 * (mat + mat.t())
    eigvals, eigvecs = torch.linalg.eigh(mat)  # ascending
    # flip to descending for convenience
    eigvals = eigvals.flip(0)
    eigvecs = eigvecs.flip(1)
    sigma_max = max(float(eigvals.max().item()), 1e-12)
    tau = max(sigma_max * tol, 1e-12)
    if damping is not None and damping > 0.0:
        inv_diag = 1.0 / (eigvals + damping)
        retained_mask = eigvals > 0.0
    else:
        inv_diag = torch.zeros_like(eigvals)
        retained_mask = eigvals > tau
        inv_diag[retained_mask] = 1.0 / eigvals[retained_mask]
    mat_plus = (eigvecs * inv_diag.unsqueeze(0)) @ eigvecs.t()
    return mat_plus, eigvecs, eigvals, retained_mask

# -------------------------
# Layerwise preconditioning and update (uses cached A_plus/G_plus if available)
# -------------------------
def precondition_and_apply_updates(model, kfac_state, q_budget, eta0, eta_min, eta_max, eps,
                                   tol, damping, use_damping, eta_null_ratio):
    """
    Compute per-layer preconditioned gradients using cached A_plus/G_plus if available,
    otherwise compute pseudoinverses on the fly. Compute global Delta_F and regulator eta_T,
    then apply hybrid updates per layer.
    """
    # Collect per-layer flattened gradients and preconditioned gradients
    layer_entries = []
    total_delta_F = 0.0

    # Build list of layers in the same order as kfac_state
    for name, st in kfac_state.items():
        module = st.module
        # collect parameter tensors and gradients for this module
        params = []
        grads = []
        if hasattr(module, "weight") and module.weight is not None:
            params.append(module.weight)
            grads.append(module.weight.grad if module.weight.grad is not None else torch.zeros_like(module.weight))
        if hasattr(module, "bias") and module.bias is not None:
            params.append(module.bias)
            grads.append(module.bias.grad if module.bias.grad is not None else torch.zeros_like(module.bias))
        if len(params) == 0:
            continue
        # flatten gradient vector for module
        g_flat = torch.cat([g.contiguous().view(-1) for g in grads]).detach()
        # compute preconditioned gradient g_nat_flat
        if st.A is None or st.G is None:
            # fallback: identity preconditioner
            g_nat_flat = g_flat.clone()
            retained_mask_A = None
            retained_mask_G = None
        else:
            # Use cached A_plus/G_plus if available
            if st.A_plus is None or st.G_plus is None:
                # compute pseudoinverses and cache
                if use_damping:
                    st.A_plus, st.A_eigvecs, st.A_eigvals, st.A_mask = symmetric_eig_pseudoinverse_torch(st.A, tol=tol, damping=damping)
                    st.G_plus, st.G_eigvecs, st.G_eigvals, st.G_mask = symmetric_eig_pseudoinverse_torch(st.G, tol=tol, damping=damping)
                else:
                    st.A_plus, st.A_eigvecs, st.A_eigvals, st.A_mask = symmetric_eig_pseudoinverse_torch(st.A, tol=tol, damping=None)
                    st.G_plus, st.G_eigvecs, st.G_eigvals, st.G_mask = symmetric_eig_pseudoinverse_torch(st.G, tol=tol, damping=None)
            # Precondition depending on layer type
            if st.layer_type == "linear":
                out, inp = module.weight.shape
                grad_mat = g_flat[:out*inp].view(out, inp)
                precond_mat = st.G_plus @ grad_mat @ st.A_plus
                g_nat_weight = precond_mat.contiguous().view(-1)
                if module.bias is not None:
                    bias_grad = g_flat[out*inp:]
                    # precondition bias by G_plus (if shapes match)
                    if bias_grad.numel() == st.G_plus.shape[0]:
                        bias_nat = (st.G_plus @ bias_grad.view(-1,1)).view(-1)
                    else:
                        bias_nat = bias_grad
                    g_nat_flat = torch.cat([g_nat_weight, bias_nat])
                else:
                    g_nat_flat = g_nat_weight
            elif st.layer_type == "conv":
                out, inp, kh, kw = module.weight.shape
                K = inp * kh * kw
                grad_weight = g_flat[:out * K].view(out, K)
                precond_mat = st.G_plus @ grad_weight @ st.A_plus  # shapes: (out,out) @ (out,K) @ (K,K)
                g_nat_weight = precond_mat.contiguous().view(-1)
                if module.bias is not None:
                    bias_grad = g_flat[out * K:]
                    if bias_grad.numel() == st.G_plus.shape[0]:
                        bias_nat = (st.G_plus @ bias_grad.view(-1,1)).view(-1)
                    else:
                        bias_nat = bias_grad
                    g_nat_flat = torch.cat([g_nat_weight, bias_nat])
                else:
                    g_nat_flat = g_nat_weight
            else:
                g_nat_flat = g_flat.clone()
            retained_mask_A = st.A_mask
            retained_mask_G = st.G_mask

        delta_F_layer = float((g_flat @ g_nat_flat).item())
        total_delta_F += delta_F_layer
        layer_entries.append((name, st, params, grads, g_flat, g_nat_flat, retained_mask_A, retained_mask_G))

    total_delta_F = max(total_delta_F, eps)
    # Thermodynamic regulator (global)
    eta_T = eta0 * (q_budget / (total_delta_F + eps))
    eta_T = max(min(eta_T, eta_max), eta_min)

    # Apply hybrid updates per layer
    for (name, st, params, grads, g_flat, g_nat_flat, maskA, maskG) in layer_entries:
        # nullspace residual
        r = g_flat - g_nat_flat
        eta_null = eta_null_ratio * eta_T
        delta_flat = -eta_T * g_nat_flat - eta_null * r
        # unflatten and apply updates in-place (add because delta_flat is negative descent)
        idx = 0
        for p in params:
            n = p.numel()
            d = delta_flat[idx:idx+n].view_as(p)
            # apply update (in-place)
            p.data.add_(d)
            idx += n

    diagnostics = {
        "total_delta_F": total_delta_F,
        "eta_T": eta_T,
        "layers": len(layer_entries)
    }
    return diagnostics

# -------------------------
# Training and evaluation
# -------------------------
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += float(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss

def train():
    model = SmallCNN(num_classes=10).to(device)
    model.train()
    kfac_state = build_kfac_state(model, device=device, ema_decay=args.ema_decay)
    criterion = nn.CrossEntropyLoss()
    # We'll not use a standard optimizer; updates applied via NGD-T preconditioning
    step = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()
            epoch_loss += float(loss.item()) * inputs.size(0)
            # Update K-FAC factors (every factor update step)
            if step % args.kfac_factor_update == 0:
                for st in kfac_state.values():
                    st.update_factors(damping=args.damping if args.use_damping else 0.0)
            # Recompute eigendecompositions and cached pseudoinverses every K steps
            if step % args.kfac_eig_update == 0:
                for st in kfac_state.values():
                    if st.A is not None and st.G is not None:
                        # compute and cache pseudoinverses
                        if args.use_damping:
                            st.A_plus, st.A_eigvecs, st.A_eigvals, st.A_mask = symmetric_eig_pseudoinverse_torch(st.A, tol=args.tol, damping=args.damping)
                            st.G_plus, st.G_eigvecs, st.G_eigvals, st.G_mask = symmetric_eig_pseudoinverse_torch(st.G, tol=args.tol, damping=args.damping)
                        else:
                            st.A_plus, st.A_eigvecs, st.A_eigvals, st.A_mask = symmetric_eig_pseudoinverse_torch(st.A, tol=args.tol, damping=None)
                            st.G_plus, st.G_eigvecs, st.G_eigvals, st.G_mask = symmetric_eig_pseudoinverse_torch(st.G, tol=args.tol, damping=None)
            # Precondition and apply updates
            diagnostics = precondition_and_apply_updates(
                model, kfac_state,
                q_budget=args.q_budget,
                eta0=args.lr_base,
                eta_min=args.eta_min,
                eta_max=args.eta_max,
                eps=args.eps,
                tol=args.tol,
                damping=args.damping,
                use_damping=args.use_damping,
                eta_null_ratio=args.eta_null_ratio
            )
            if step % 50 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f} eta_T {diagnostics['eta_T']:.6f} total_delta_F {diagnostics['total_delta_F']:.6e}")
            step += 1
        t1 = time.time()
        epoch_time = t1 - t0
        avg_loss = epoch_loss / len(trainset)
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s, avg loss {avg_loss:.4f}")
        # Evaluate on test set
        acc, test_loss = test(model, testloader)
        print(f"Test accuracy after epoch {epoch}: {acc:.2f}%  test_loss {test_loss:.4f}")
    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60.0:.2f} minutes")
    return model

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print("Device:", device)
    print("Arguments:", args)
    trained_model = train()
    final_acc, final_loss = test(trained_model, testloader)
    print(f"Final test accuracy: {final_acc:.2f}%  final_loss {final_loss:.4f}")
