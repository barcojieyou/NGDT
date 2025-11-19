'''
A self‑contained NumPy worked example (tiny 𝑝=4) that demonstrates every step: 
forming a rank‑deficient Fisher, 
computing the Moore–Penrose pseudoinverse by eigendecomposition, 
computing the natural gradient, 
evaluating the geometric norm Δ𝐹, 
computing the NGD‑T regulator 𝜂𝑇, 
projecting the nullspace and forming a hybrid update, 
and reporting diagnostics (predicted dissipation).

Problem setup (toy, 𝑝=4)
Parameter dimension: 𝑝=4
Fisher estimate (rank‑deficient): 𝐹=diag(2,1, 0, 0)
Gradient: 𝑔=[1, 2, 3, 4]⊤
Dissipation budget: 𝑄_budget=0.1
Regulator base: 𝜂_0=1.0
Small stabilizer: 𝜖=10^−8
Nullspace fallback ratio: 𝜂_null_ratio=0.01
Eigenvalue truncation threshold: 𝜏=10^−8 (so zero eigenvalues are treated as null).
'''
import numpy as np

# --- Inputs ---
p = 4
# Fisher estimate (diagonal, rank-deficient)
F = np.diag([2.0, 1.0, 0.0, 0.0])   # shape (4,4)
g = np.array([1.0, 2.0, 3.0, 4.0])  # gradient vector

# Hyperparameters
Q_budget = 0.1
eta0 = 1.0
eps = 1e-8
eta_min = 1e-6
eta_max = 1.0
eta_null_ratio = 0.01
tol = 1e-8

# --- Step A: eigendecomposition of symmetric F ---
eigvals, eigvecs = np.linalg.eigh(F)   # eigvals ascending
# For this diagonal F, eigvecs == identity, eigvals == [0,0,1,2] (ascending)
# We'll reorder descending for clarity (optional)
idx_desc = eigvals.argsort()[::-1]
eigvals = eigvals[idx_desc]
eigvecs = eigvecs[:, idx_desc]

print("Eigenvalues (desc):", eigvals)
print("Eigenvectors (columns):\n", eigvecs)

# --- Step B: build Moore-Penrose pseudoinverse via eigenbasis ---
sigma_max = max(eigvals.max(), 1e-12)
tau = max(sigma_max * tol, 1e-12)   # numerical threshold
inv_diag = np.zeros_like(eigvals)
retained_mask = eigvals > tau
inv_diag[retained_mask] = 1.0 / eigvals[retained_mask]
# reconstruct F_plus
F_plus = (eigvecs * inv_diag) @ eigvecs.T

print("\nRetained eigenvalue mask:", retained_mask)
print("Pseudoinverse diagonal (in eigenbasis):", inv_diag)
print("F_plus (pseudoinverse):\n", F_plus)

# --- Step C: natural gradient and geometric norm ---
g_nat = F_plus @ g
delta_F = float(g @ g_nat)   # scalar geometric norm
delta_F = max(delta_F, eps)

print("\ng (gradient):", g)
print("g_nat (natural gradient):", g_nat)
print("Delta_F (g^T F^+ g):", delta_F)

# --- Step D: thermodynamic regulator ---
eta_T = eta0 * (Q_budget / (delta_F + eps))
eta_T = float(min(max(eta_T, eta_min), eta_max))
print("\neta_T (thermodynamic regulator):", eta_T)

# --- Step E: nullspace projection and hybrid update ---
# Build retained basis U_r (columns of eigvecs with retained_mask True)
U_r = eigvecs[:, retained_mask]
if U_r.size == 0:
    P_null = np.eye(p)
else:
    P_null = np.eye(p) - U_r @ U_r.T

g_null = P_null @ g
eta_null = eta_null_ratio * eta_T

delta_theta = -eta_T * g_nat - eta_null * g_null

print("\nP_null (projector onto nullspace):\n", P_null)
print("g_null (component of g in nullspace):", g_null)
print("eta_null:", eta_null)
print("delta_theta (hybrid update):", delta_theta)

# --- Step F: predicted dissipation ---
predicted_Q = 0.5 * (eta_T**2) * delta_F
print("\npredicted_Q (0.5 * eta_T^2 * Delta_F):", predicted_Q)
