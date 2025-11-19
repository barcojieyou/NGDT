# ngdt_pseudoinverse.py
# PyTorch implementation: pseudoinverse natural gradient + thermodynamic regulator + hybrid update
'''
A compact, ready‑to‑run reference implementation that:
(1) forms a symmetric Fisher estimate F_est, 
(2) computes a Moore–Penrose pseudoinverse (via eigendecomposition), 
(3) computes the natural gradient g_nat, 
(4) evaluates the geometric norm Δ_F, 
(5) computes the thermodynamic regulator η_T, and 
(6) applies a hybrid update that also moves in the nullspace with a small Euclidean step.
'''

import torch
import torch.nn as nn

# -------------------------
# Hyperparameters / defaults
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thermodynamic regulator params
ETA0 = 1.0            # nominal scale
ETA_MIN = 1e-6
ETA_MAX = 1.0
EPS = 1e-8            # numerical stabilizer for Δ_F
QBUDGET = 1e-3        # dissipation budget (tune per task)

# Pseudoinverse / truncation params
TOL = 1e-8            # relative threshold for eigenvalues
EPS_FLOOR = 1e-12     # absolute floor
DAMPING = 1e-6        # optional Tikhonov damping (set >0 to avoid hard truncation)

# Nullspace fallback
ETA_NULL_RATIO = 0.01  # η_null = ETA_NULL_RATIO * η_T

# -------------------------
# Utility functions
# -------------------------
def symmetrize(F):
    return 0.5 * (F + F.t())

def compute_pseudoinverse_from_eig(F, tol=TOL, eps_floor=EPS_FLOOR, damping=None, return_basis=False):
    """
    Compute Moore-Penrose pseudoinverse of symmetric PSD matrix F via eigendecomposition.
    If damping is provided (float > 0), compute (F + damping*I)^{-1} instead (Tikhonov).
    Returns F_plus and optionally retained eigenvectors/mask.
    """
    # Ensure symmetric
    F = symmetrize(F)
    # Eigendecomposition (ascending eigenvalues)
    # Use torch.linalg.eigh for symmetric matrices
    eigvals, eigvecs = torch.linalg.eigh(F)
    # Numerical threshold
    sigma_max = eigvals.max().clamp(min=eps_floor)
    tau = max(sigma_max * tol, eps_floor)
    # Build pseudoinverse diagonal
    if damping is not None and damping > 0.0:
        # Tikhonov inverse: 1/(λ + damping)
        inv_diag = 1.0 / (eigvals + damping)
        retained_mask = eigvals > 0.0  # for diagnostics only
    else:
        inv_diag = torch.zeros_like(eigvals)
        retained_mask = eigvals > tau
        inv_diag[retained_mask] = 1.0 / eigvals[retained_mask]
    # Reconstruct pseudoinverse
    F_plus = (eigvecs * inv_diag.unsqueeze(0)) @ eigvecs.t()
    if return_basis:
        return F_plus, eigvecs, eigvals, retained_mask
    return F_plus

# -------------------------
# Core step: compute update
# -------------------------
def ngd_t_step(theta, grad, F_est, q_budget=QBUDGET, eta0=ETA0,
               eta_min=ETA_MIN, eta_max=ETA_MAX, eps=EPS,
               tol=TOL, eps_floor=EPS_FLOOR, damping=DAMPING,
               eta_null_ratio=ETA_NULL_RATIO, use_tikhonov=False):
    """
    Compute NGD-T hybrid update for parameter vector theta (flattened) given gradient grad and Fisher estimate F_est.
    - theta: 1D tensor (p,)
    - grad: 1D tensor (p,)
    - F_est: 2D tensor (p,p) symmetric PSD estimate of Fisher
    Returns: delta_theta (1D tensor), diagnostics dict
    """
    # Move to device
    theta = theta.to(DEVICE)
    g = grad.to(DEVICE)
    F = F_est.to(DEVICE)

    # Symmetrize F
    F = symmetrize(F)

    # Compute pseudoinverse (either Tikhonov or MP pseudoinverse)
    if use_tikhonov:
        # Use damping as Tikhonov parameter
        F_plus = compute_pseudoinverse_from_eig(F, tol=tol, eps_floor=eps_floor, damping=damping, return_basis=False)
        # For diagnostics, compute eigendecomp too
        eigvals, eigvecs = torch.linalg.eigh(F)
        retained_mask = eigvals > eps_floor
    else:
        F_plus, eigvecs, eigvals, retained_mask = compute_pseudoinverse_from_eig(F, tol=tol, eps_floor=eps_floor, damping=None, return_basis=True)

    # Natural gradient (preconditioned gradient)
    g_nat = F_plus @ g

    # Geometric norm Δ_F
    delta_F = float(torch.dot(g, g_nat).clamp(min=eps))

    # Thermodynamic regulator
    eta_T = eta0 * (q_budget / (delta_F + eps))
    eta_T = max(min(eta_T, eta_max), eta_min)

    # Nullspace projection
    # Build retained basis U_r from eigvecs where retained_mask True
    if retained_mask.sum() > 0:
        U_r = eigvecs[:, retained_mask]
        P_null = torch.eye(F.shape[0], device=F.device) - U_r @ U_r.t()
    else:
        # All truncated: nullspace is whole space
        P_null = torch.eye(F.shape[0], device=F.device)

    g_null = P_null @ g
    eta_null = eta_null_ratio * eta_T

    # Hybrid update: natural-space step + small nullspace Euclidean step
    delta_theta = -eta_T * g_nat - eta_null * g_null

    # Diagnostics
    diagnostics = {
        "delta_F": delta_F,
        "eta_T": eta_T,
        "eta_null": eta_null,
        "retained_rank": int(retained_mask.sum().item()) if 'retained_mask' in locals() else None,
        "eigvals_max": float(eigvals.max().item()) if 'eigvals' in locals() else None,
        "eigvals_min_retained": float(eigvals[retained_mask].min().item()) if ('eigvals' in locals() and retained_mask.sum()>0) else None,
        "predicted_Q": 0.5 * (eta_T**2) * delta_F
    }

    return delta_theta, diagnostics

# -------------------------
# Example usage (toy)
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    p = 64
    # Toy parameter vector and gradient
    theta = torch.randn(p, device=DEVICE)
    grad = torch.randn(p, device=DEVICE)

    # Toy Fisher estimate: construct low-rank + small noise to simulate degeneracy
    rank = 10
    U = torch.randn(p, rank, device=DEVICE)
    F_lowrank = U @ U.t()  # positive semidefinite, rank-deficient
    noise = 1e-6 * torch.randn(p, p, device=DEVICE)
    F_est = symmetrize(F_lowrank + noise)

    delta_theta, diag = ngd_t_step(theta, grad, F_est, use_tikhonov=False)
    print("Diagnostics:", diag)
    # Apply update (example)
    theta_new = theta + delta_theta
