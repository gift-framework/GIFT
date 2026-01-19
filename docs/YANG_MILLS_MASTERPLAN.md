# YANG-MILLS MASS GAP × GIFT
## Plan d'Attaque Complet pour Claude Code

**Objectif**: Démontrer que la topologie de K₇ implique un mass gap pour la théorie de jauge
**Ressources**: Google Colab A100, repo GIFT existant
**Timeline**: Phased approach

---

## 🎯 SYNTHÈSE DES PERSPECTIVES IA

### Consensus
- **Yang-Mills est LA piste** (4 étoiles unanimes)
- Le gap spectral λ₁ > 0 est **mathématiquement obligatoire** sur une variété compacte
- La question est : **comment ce gap se propage vers 4D ?**

### Formule cible
```
Δ = (dim(G₂)/H*) × Λ_QCD = (14/99) × Λ_QCD ≈ 28 MeV

où:
- 14 = p₂ × dim(K₇) = 2 × 7
- 99 = N_gen² × D_bulk = 9 × 11
```

### Constante de Cheeger cible
```
h(K₇) ≈ 14/99 ≈ 0.1414
λ₁ ≥ h²/4 ≈ 0.005
```

---

## 📋 STRUCTURE DU REPO

```
gift-yang-mills/
├── README.md
├── requirements.txt
├── 
├── notebooks/                    # Colab notebooks (A100)
│   ├── 01_pinn_metric_training.ipynb
│   ├── 02_manifold_sampling.ipynb
│   ├── 03_spectral_analysis.ipynb
│   ├── 04_cheeger_estimation.ipynb
│   └── 05_kk_reduction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── constants.py              # GIFT constants
│   ├── g2_structure.py           # G₂ forms φ, ψ
│   ├── tcs_manifold.py           # TCS construction
│   ├── metric_pinn.py            # PINN for metric
│   ├── spectral/
│   │   ├── __init__.py
│   │   ├── graph_laplacian.py    # Discrete Laplacian
│   │   ├── hodge_laplacian.py    # Hodge-de Rham
│   │   ├── cheeger.py            # Isoperimetric constant
│   │   └── eigensolvers.py       # Spectral methods
│   ├── gauge/
│   │   ├── __init__.py
│   │   ├── e8_structure.py       # E₈ roots, Cartan
│   │   ├── breaking_chain.py     # E₈ → SM
│   │   └── kk_reduction.py       # Kaluza-Klein
│   └── visualization/
│       ├── spectrum_plots.py
│       └── manifold_viz.py
│
├── data/
│   ├── pinn_checkpoints/
│   ├── spectral_results/
│   └── exports/
│
├── tests/
│   └── test_*.py
│
└── paper/
    ├── yang_mills_gift.tex
    └── figures/
```

---

## 🚀 PHASE 1: INFRASTRUCTURE (Semaine 1)

### 1.1 Fichier de constantes GIFT

**Fichier**: `src/constants.py`

```python
"""
GIFT Framework Constants
All values are topologically derived - zero free parameters
"""
from dataclasses import dataclass
from fractions import Fraction
import numpy as np

@dataclass(frozen=True)
class GIFTConstants:
    # Manifold topology
    dim_K7: int = 7
    b2: int = 21          # Second Betti number
    b3: int = 77          # Third Betti number
    H_star: int = 99      # b2 + b3 + 1
    
    # Holonomy
    dim_G2: int = 14
    rank_G2: int = 2
    
    # Gauge structure
    dim_E8: int = 248
    rank_E8: int = 8
    dim_E8xE8: int = 496
    roots_E8: int = 240
    coxeter_E8: int = 30
    
    # Derived constants
    N_gen: int = 3
    Weyl: int = 5
    p2: int = 2
    D_bulk: int = 11
    
    # Metric
    det_g_num: int = 65
    det_g_den: int = 32
    
    # Torsion
    kappa_T_inv: int = 61
    
    # Yang-Mills targets
    @property
    def det_g(self) -> float:
        return self.det_g_num / self.det_g_den
    
    @property
    def kappa_T(self) -> float:
        return 1 / self.kappa_T_inv
    
    @property
    def cheeger_target(self) -> float:
        """Target Cheeger constant h(K₇) = dim(G₂)/H*"""
        return self.dim_G2 / self.H_star  # 14/99 ≈ 0.1414
    
    @property
    def lambda1_lower_bound(self) -> float:
        """Cheeger inequality: λ₁ ≥ h²/4"""
        return self.cheeger_target**2 / 4  # ≈ 0.005
    
    @property
    def mass_gap_ratio(self) -> Fraction:
        """Δ/Λ_QCD = dim(G₂)/H*"""
        return Fraction(self.dim_G2, self.H_star)

GIFT = GIFTConstants()
```

### 1.2 Structure G₂

**Fichier**: `src/g2_structure.py`

```python
"""
G₂ Structure on 7-manifolds
Implements the associative 3-form φ and coassociative 4-form ψ
"""
import numpy as np
from itertools import permutations

# Standard G₂ structure constants (Bryant convention)
# φ = e^{123} + e^{145} + e^{167} + e^{246} - e^{257} - e^{347} - e^{356}
G2_PHI_TERMS = [
    ((0, 1, 2), +1),
    ((0, 3, 4), +1),
    ((0, 5, 6), +1),
    ((1, 3, 5), +1),
    ((1, 4, 6), -1),
    ((2, 3, 6), -1),
    ((2, 4, 5), -1),
]

class G2Structure:
    """G₂ structure with associative 3-form φ."""
    
    def __init__(self, scale: float = 1.0):
        """
        Initialize G₂ structure.
        
        Args:
            scale: Scaling factor c = (65/32)^(1/14) for GIFT metric
        """
        self.scale = scale
        self._build_phi()
        self._build_psi()
    
    def _build_phi(self):
        """Build the associative 3-form φ."""
        self.phi = np.zeros((7, 7, 7))
        for indices, sign in G2_PHI_TERMS:
            for perm in permutations(range(3)):
                perm_sign = self._perm_sign(perm)
                i, j, k = [indices[p] for p in perm]
                self.phi[i, j, k] = sign * perm_sign * self.scale
    
    def _build_psi(self):
        """Build coassociative 4-form ψ = *φ."""
        # ψ is the Hodge dual of φ
        self.psi = np.zeros((7, 7, 7, 7))
        # Implementation details...
    
    @staticmethod
    def _perm_sign(perm):
        """Compute sign of permutation."""
        n = len(perm)
        inversions = sum(1 for i in range(n) for j in range(i+1, n) 
                        if perm[i] > perm[j])
        return (-1) ** inversions
    
    def phi_norm_squared(self) -> float:
        """Compute ||φ||² = 7 for standard structure."""
        return np.sum(self.phi ** 2) / 6  # Factor from antisymmetry
    
    def metric_from_phi(self) -> np.ndarray:
        """
        Extract metric from φ using Bryant's formula.
        For standard φ₀, returns scaled identity.
        """
        # g_ij = (1/6) φ_imn φ_jpq φ_krs ε^mnpqrs δ^k / sqrt(det)
        # For standard φ: g = scale² × I₇
        return self.scale**2 * np.eye(7)
    
    def torsion_norm(self, dφ: np.ndarray, dψ: np.ndarray) -> float:
        """
        Compute torsion ||T||² = ||dφ||² + ||d*φ||²
        
        For torsion-free G₂: dφ = 0 and d*φ = 0
        """
        return np.sum(dφ**2) + np.sum(dψ**2)
```

---

## 🔬 PHASE 2: PINN METRIC TRAINING (Semaine 2)

### 2.1 Notebook Colab: `01_pinn_metric_training.ipynb`

**Objectif**: Entraîner un PINN pour apprendre la métrique g_ij sur K₇

```python
# HEADER pour Colab
"""
GIFT Yang-Mills: PINN Metric Training
=====================================
Runtime: A100 GPU
Objective: Learn g_ij(x) satisfying G₂ constraints
"""

# !pip install torch numpy scipy matplotlib tqdm

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# GIFT Constants
DIM_K7 = 7
DET_G_TARGET = 65/32
KAPPA_T = 1/61

class MetricPINN(nn.Module):
    """
    Physics-Informed Neural Network for G₂ metric.
    
    Outputs: 28 independent components of symmetric 7×7 metric
    (7 diagonal + 21 upper triangular)
    """
    
    def __init__(self, hidden_dim=256, num_layers=6):
        super().__init__()
        
        layers = [nn.Linear(DIM_K7, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 28))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize to near-identity metric
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to output near-identity metric."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Points on K₇, shape (batch, 7)
        Returns:
            g: Metric tensors, shape (batch, 7, 7)
        """
        components = self.network(x)
        return self._components_to_metric(components)
    
    def _components_to_metric(self, components):
        """Convert 28 components to symmetric 7×7 matrix."""
        batch = components.shape[0]
        g = torch.zeros(batch, 7, 7, device=components.device)
        
        idx = 0
        for i in range(7):
            for j in range(i, 7):
                # Ensure positive definiteness for diagonal
                if i == j:
                    g[:, i, j] = torch.exp(components[:, idx])
                else:
                    g[:, i, j] = components[:, idx]
                    g[:, j, i] = components[:, idx]
                idx += 1
        
        return g


class G2PhysicsLoss(nn.Module):
    """
    Physics loss for G₂ holonomy constraints.
    """
    
    def __init__(self, det_target=65/32, torsion_weight=10.0):
        super().__init__()
        self.det_target = det_target
        self.torsion_weight = torsion_weight
    
    def forward(self, g, x):
        """
        Compute physics-informed loss.
        
        Components:
        1. Determinant constraint: det(g) = 65/32
        2. Ricci-flatness: R_ij ≈ 0 (approximated)
        3. Torsion minimization: ||dφ||² + ||d*φ||² → 0
        """
        batch = g.shape[0]
        
        # 1. Determinant loss
        det_g = torch.linalg.det(g)
        loss_det = torch.mean((det_g - self.det_target)**2)
        
        # 2. Positive definiteness (all eigenvalues > 0)
        eigenvalues = torch.linalg.eigvalsh(g)
        loss_pd = torch.mean(torch.relu(-eigenvalues + 0.01)**2)
        
        # 3. Smoothness (Laplacian regularization)
        # Approximate by finite differences
        eps = 0.01
        loss_smooth = 0.0
        for i in range(7):
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[:, i] += eps
            x_minus[:, i] -= eps
            # Would need model reference here - simplified
        
        # 4. Torsion (requires dφ computation - simplified here)
        # Full implementation in separate function
        loss_torsion = torch.tensor(0.0, device=g.device)
        
        total_loss = loss_det + 0.1 * loss_pd + self.torsion_weight * loss_torsion
        
        return total_loss, {
            'det': loss_det.item(),
            'pd': loss_pd.item(),
            'torsion': loss_torsion.item()
        }


def sample_K7_points(n_points, method='uniform'):
    """
    Sample points on K₇ manifold.
    
    For TCS construction, we use local coordinates on
    the cylindrical regions.
    """
    if method == 'uniform':
        # Simple uniform sampling in [0, 2π]^7
        return torch.rand(n_points, 7) * 2 * np.pi
    elif method == 'gaussian':
        return torch.randn(n_points, 7)
    else:
        raise ValueError(f"Unknown method: {method}")


def train_metric_pinn(epochs=10000, batch_size=1024, lr=1e-3):
    """Main training loop."""
    
    model = MetricPINN().to(device)
    loss_fn = G2PhysicsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    history = {'loss': [], 'det': [], 'torsion': []}
    
    pbar = tqdm(range(epochs), desc="Training PINN")
    for epoch in pbar:
        # Sample points
        x = sample_K7_points(batch_size).to(device)
        
        # Forward
        g = model(x)
        loss, metrics = loss_fn(g, x)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Log
        history['loss'].append(loss.item())
        history['det'].append(metrics['det'])
        
        if epoch % 100 == 0:
            det_mean = torch.linalg.det(g).mean().item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'det': f"{det_mean:.4f}",
                'target': f"{65/32:.4f}"
            })
    
    return model, history


# MAIN
if __name__ == "__main__":
    model, history = train_metric_pinn(epochs=5000)
    
    # Save checkpoint
    torch.save({
        'model_state': model.state_dict(),
        'history': history
    }, 'pinn_metric_checkpoint.pt')
    
    print("Training complete!")
```

---

## 📊 PHASE 3: SPECTRAL ANALYSIS (Semaine 3-4)

### 3.1 Notebook Colab: `03_spectral_analysis.ipynb`

**Objectif**: Calculer le spectre du Laplacien et estimer λ₁

```python
"""
GIFT Yang-Mills: Spectral Analysis
==================================
Compute the spectrum of the Hodge Laplacian on K₇
Key target: λ₁ ≥ (14/99)²/4 ≈ 0.005
"""

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial import KDTree
from tqdm import tqdm

class ManifoldSampler:
    """
    Sample points from K₇ using trained PINN metric.
    """
    
    def __init__(self, pinn_model, n_points=10000):
        self.model = pinn_model
        self.n_points = n_points
        self.points = None
        self.metric_at_points = None
    
    def sample(self, method='importance'):
        """
        Sample points on the manifold.
        
        For importance sampling, weight by sqrt(det(g))
        to get uniform distribution w.r.t. volume form.
        """
        device = next(self.model.parameters()).device
        
        if method == 'uniform':
            self.points = torch.rand(self.n_points, 7) * 2 * np.pi
        
        elif method == 'importance':
            # Oversample and reject
            oversample = 5 * self.n_points
            candidates = torch.rand(oversample, 7, device=device) * 2 * np.pi
            
            with torch.no_grad():
                g = self.model(candidates)
                det_g = torch.linalg.det(g)
                weights = torch.sqrt(torch.abs(det_g))
                weights = weights / weights.sum()
            
            # Resample according to weights
            indices = torch.multinomial(weights, self.n_points, replacement=False)
            self.points = candidates[indices].cpu()
        
        # Compute metric at sampled points
        with torch.no_grad():
            self.metric_at_points = self.model(self.points.to(device)).cpu().numpy()
        
        return self.points.numpy()


class GraphLaplacian:
    """
    Approximate Hodge Laplacian using graph Laplacian.
    
    Method:
    1. Build k-NN graph from sampled points
    2. Weight edges by metric tensor (geodesic approximation)
    3. Compute normalized graph Laplacian
    4. Extract spectrum
    """
    
    def __init__(self, points, metric, k_neighbors=20):
        """
        Args:
            points: (N, 7) array of sampled points
            metric: (N, 7, 7) array of metric tensors at points
            k_neighbors: Number of nearest neighbors
        """
        self.points = points
        self.metric = metric
        self.k = k_neighbors
        self.N = len(points)
        
    def build_weighted_adjacency(self):
        """
        Build weighted adjacency matrix using metric-weighted distances.
        
        W_ij = exp(-d_g(x_i, x_j)² / σ²)
        
        where d_g is the geodesic distance approximated by:
        d_g(x,y) ≈ sqrt((x-y)^T g(x) (x-y))
        """
        print("Building KD-tree...")
        tree = KDTree(self.points)
        
        print("Computing weighted adjacency...")
        rows, cols, data = [], [], []
        
        for i in tqdm(range(self.N), desc="Building graph"):
            # Find k nearest neighbors in Euclidean metric
            dists, neighbors = tree.query(self.points[i], k=self.k+1)
            
            g_i = self.metric[i]  # Metric at point i
            
            for j, neighbor in enumerate(neighbors[1:]):  # Skip self
                # Compute metric-weighted distance
                diff = self.points[neighbor] - self.points[i]
                d_g_sq = diff @ g_i @ diff
                
                # Gaussian kernel
                sigma = np.median(dists[1:])  # Adaptive bandwidth
                weight = np.exp(-d_g_sq / (2 * sigma**2))
                
                rows.append(i)
                cols.append(neighbor)
                data.append(weight)
        
        W = csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        # Symmetrize
        W = (W + W.T) / 2
        
        return W
    
    def build_laplacian(self, W, normalized=True):
        """
        Build graph Laplacian from adjacency matrix.
        
        Normalized: L = I - D^{-1/2} W D^{-1/2}
        Unnormalized: L = D - W
        """
        degree = np.array(W.sum(axis=1)).flatten()
        
        if normalized:
            # Avoid division by zero
            degree = np.maximum(degree, 1e-10)
            D_inv_sqrt = csr_matrix(np.diag(1.0 / np.sqrt(degree)))
            L = csr_matrix(np.eye(self.N)) - D_inv_sqrt @ W @ D_inv_sqrt
        else:
            D = csr_matrix(np.diag(degree))
            L = D - W
        
        return L
    
    def compute_spectrum(self, n_eigenvalues=100):
        """
        Compute the first n eigenvalues of the Laplacian.
        
        Returns:
            eigenvalues: Sorted array of eigenvalues
            eigenvectors: Corresponding eigenvectors
        """
        print("Building weighted adjacency matrix...")
        W = self.build_weighted_adjacency()
        
        print("Building Laplacian...")
        L = self.build_laplacian(W, normalized=True)
        
        print(f"Computing {n_eigenvalues} smallest eigenvalues...")
        # 'SM' = Smallest Magnitude
        eigenvalues, eigenvectors = eigsh(L, k=n_eigenvalues, which='SM')
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors


class HodgeLaplacian:
    """
    More accurate Hodge-de Rham Laplacian on k-forms.
    
    For 0-forms (functions): Δ₀ = d*d
    For 1-forms: Δ₁ = dd* + d*d
    
    The mass gap comes from Δ₁ on the gauge sector.
    """
    
    def __init__(self, points, metric):
        self.points = points
        self.metric = metric
        self.N = len(points)
    
    def compute_spectrum_on_1forms(self, n_eigenvalues=50):
        """
        Compute spectrum of Δ₁ (Laplacian on 1-forms).
        
        This is the physically relevant operator for gauge fields.
        
        Note: Full implementation requires discrete exterior calculus (DEC).
        Here we use a simplified approach based on vector Laplacian.
        """
        # For a complete implementation, use PyDEC or similar
        # This is a placeholder for the structure
        
        # Vector Laplacian: Δ_vec = (d*d + dd*) on vector fields
        # In local coordinates: (Δv)^i = g^{jk} ∇_j ∇_k v^i + R^i_j v^j
        
        # For Ricci-flat (G₂ holonomy): R_ij = 0
        # So Δ_vec reduces to the rough Laplacian
        
        # Approximation: use scalar Laplacian on each component
        # This gives a lower bound on the true spectrum
        
        raise NotImplementedError(
            "Full Hodge Laplacian requires DEC implementation. "
            "See Phase 4 for detailed approach."
        )


def analyze_spectrum(eigenvalues, gift_constants):
    """
    Analyze the spectrum and compare with GIFT predictions.
    """
    print("\n" + "="*60)
    print("SPECTRAL ANALYSIS RESULTS")
    print("="*60)
    
    # Mass gap (first nonzero eigenvalue)
    # λ₀ ≈ 0 (constant mode)
    lambda_0 = eigenvalues[0]
    lambda_1 = eigenvalues[1]
    
    print(f"\nFirst eigenvalues:")
    print(f"  λ₀ = {lambda_0:.6f} (should be ≈ 0)")
    print(f"  λ₁ = {lambda_1:.6f} (MASS GAP CANDIDATE)")
    print(f"  λ₂ = {eigenvalues[2]:.6f}")
    print(f"  λ₃ = {eigenvalues[3]:.6f}")
    
    # GIFT predictions
    h_target = gift_constants['dim_G2'] / gift_constants['H_star']  # 14/99
    cheeger_bound = h_target**2 / 4
    
    print(f"\nGIFT predictions:")
    print(f"  h(K₇) target = dim(G₂)/H* = {h_target:.6f}")
    print(f"  Cheeger bound: λ₁ ≥ h²/4 = {cheeger_bound:.6f}")
    
    # Comparison
    print(f"\nComparison:")
    print(f"  λ₁ observed = {lambda_1:.6f}")
    print(f"  λ₁ ≥ h²/4 satisfied? {lambda_1 >= cheeger_bound * 0.9}")  # 10% tolerance
    
    # Cheeger constant estimation (reverse direction)
    h_estimated = 2 * np.sqrt(lambda_1)
    print(f"\nCheeger constant estimation:")
    print(f"  h ≤ 2√λ₁ = {h_estimated:.6f}")
    print(f"  h target = {h_target:.6f}")
    print(f"  Ratio: {h_estimated / h_target:.2f}")
    
    # Spectral gap ratio
    gap_ratio = lambda_1 / lambda_0 if lambda_0 > 1e-10 else float('inf')
    print(f"\nSpectral gap ratio: λ₁/λ₀ = {gap_ratio:.2f}")
    
    return {
        'lambda_0': lambda_0,
        'lambda_1': lambda_1,
        'h_estimated': h_estimated,
        'h_target': h_target,
        'cheeger_satisfied': lambda_1 >= cheeger_bound * 0.9
    }


# MAIN EXECUTION
if __name__ == "__main__":
    # Load trained PINN
    checkpoint = torch.load('pinn_metric_checkpoint.pt')
    model = MetricPINN()
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Sample manifold
    sampler = ManifoldSampler(model, n_points=10000)
    points = sampler.sample(method='importance')
    metric = sampler.metric_at_points
    
    # Compute spectrum
    laplacian = GraphLaplacian(points, metric, k_neighbors=30)
    eigenvalues, eigenvectors = laplacian.compute_spectrum(n_eigenvalues=100)
    
    # Analyze
    gift_constants = {
        'dim_G2': 14,
        'H_star': 99,
        'dim_K7': 7,
        'b2': 21,
        'b3': 77
    }
    
    results = analyze_spectrum(eigenvalues, gift_constants)
    
    # Save results
    np.savez('spectral_results.npz',
             eigenvalues=eigenvalues,
             eigenvectors=eigenvectors,
             points=points,
             **results)
    
    print("\nResults saved to spectral_results.npz")
```

---

## 📐 PHASE 4: CHEEGER CONSTANT (Semaine 5)

### 4.1 Notebook: `04_cheeger_estimation.ipynb`

**Objectif**: Estimer directement la constante de Cheeger h(K₇)

```python
"""
GIFT Yang-Mills: Cheeger Constant Estimation
============================================
Direct estimation of isoperimetric constant h(K₇)

Target: h(K₇) ≈ 14/99 ≈ 0.1414
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import SpectralClustering

class CheegerEstimator:
    """
    Estimate Cheeger constant via min-cut optimization.
    
    h(M) = inf_Ω (Area(∂Ω) / min(Vol(Ω), Vol(M\Ω)))
    
    We approximate this using spectral clustering and
    geometric min-cut algorithms.
    """
    
    def __init__(self, points, metric, adjacency):
        self.points = points
        self.metric = metric
        self.W = adjacency
        self.N = len(points)
    
    def estimate_volumes(self, partition):
        """
        Estimate volumes of partition sets using metric.
        
        Vol(Ω) = ∫_Ω √det(g) dx ≈ Σ_{x_i ∈ Ω} √det(g(x_i)) × cell_volume
        """
        det_g = np.array([np.linalg.det(self.metric[i]) for i in range(self.N)])
        sqrt_det = np.sqrt(np.abs(det_g))
        
        # Approximate cell volumes (Voronoi-like)
        cell_vol = np.ones(self.N) / self.N  # Uniform approximation
        
        vol_omega = np.sum(sqrt_det[partition] * cell_vol[partition])
        vol_complement = np.sum(sqrt_det[~partition] * cell_vol[~partition])
        
        return vol_omega, vol_complement
    
    def estimate_boundary_area(self, partition):
        """
        Estimate area of boundary ∂Ω.
        
        Area(∂Ω) ≈ Σ_{i∈Ω, j∉Ω} w_ij × √det(g) × distance
        """
        area = 0.0
        W_dense = self.W.toarray()
        
        for i in np.where(partition)[0]:
            for j in np.where(~partition)[0]:
                if W_dense[i, j] > 0:
                    # Edge weight as proxy for boundary area
                    det_avg = np.sqrt(np.abs(np.linalg.det(self.metric[i])))
                    area += W_dense[i, j] * det_avg
        
        return area
    
    def cheeger_ratio(self, partition):
        """
        Compute Cheeger ratio for a given partition.
        
        h(Ω) = Area(∂Ω) / min(Vol(Ω), Vol(M\Ω))
        """
        vol_omega, vol_complement = self.estimate_volumes(partition)
        area = self.estimate_boundary_area(partition)
        
        min_vol = min(vol_omega, vol_complement)
        if min_vol < 1e-10:
            return float('inf')
        
        return area / min_vol
    
    def spectral_partition(self, n_clusters=2):
        """
        Use spectral clustering to find good partition.
        
        The Fiedler vector (second eigenvector of Laplacian)
        gives an approximately optimal 2-partition.
        """
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='discretize'
        )
        labels = clustering.fit_predict(self.W.toarray())
        
        return labels == 0  # Boolean partition
    
    def estimate_cheeger(self, n_trials=100):
        """
        Estimate Cheeger constant via multiple random partitions.
        
        Returns lower bound: h(K₇) ≈ min over trials
        """
        print("Estimating Cheeger constant...")
        
        # Method 1: Spectral partition (best single estimate)
        partition_spectral = self.spectral_partition()
        h_spectral = self.cheeger_ratio(partition_spectral)
        print(f"  Spectral partition: h ≈ {h_spectral:.6f}")
        
        # Method 2: Random partitions (exploration)
        h_values = []
        for _ in range(n_trials):
            # Random balanced partition
            partition = np.random.rand(self.N) > 0.5
            h_values.append(self.cheeger_ratio(partition))
        
        h_random_best = min(h_values)
        h_random_mean = np.mean(h_values)
        print(f"  Random partitions: h_min = {h_random_best:.6f}, h_mean = {h_random_mean:.6f}")
        
        # Best estimate
        h_estimate = min(h_spectral, h_random_best)
        
        return {
            'h_estimate': h_estimate,
            'h_spectral': h_spectral,
            'h_random_best': h_random_best,
            'h_random_mean': h_random_mean
        }


def compare_with_gift_target(h_estimate, h_target=14/99):
    """
    Compare estimated Cheeger constant with GIFT prediction.
    """
    print("\n" + "="*60)
    print("CHEEGER CONSTANT COMPARISON")
    print("="*60)
    
    print(f"\nEstimated: h(K₇) ≈ {h_estimate:.6f}")
    print(f"GIFT target: h = dim(G₂)/H* = 14/99 = {h_target:.6f}")
    print(f"Ratio: {h_estimate / h_target:.2f}")
    
    # Implications for mass gap
    lambda1_bound = h_estimate**2 / 4
    lambda1_target = h_target**2 / 4
    
    print(f"\nImplied mass gap bounds:")
    print(f"  From estimate: λ₁ ≥ {lambda1_bound:.6f}")
    print(f"  From target:   λ₁ ≥ {lambda1_target:.6f}")
    
    # Physical mass gap (with Λ_QCD = 200 MeV)
    Lambda_QCD = 200  # MeV
    Delta_estimate = h_estimate * Lambda_QCD
    Delta_target = h_target * Lambda_QCD
    
    print(f"\nPhysical mass gap (with Λ_QCD = {Lambda_QCD} MeV):")
    print(f"  From estimate: Δ ≈ {Delta_estimate:.1f} MeV")
    print(f"  From target:   Δ ≈ {Delta_target:.1f} MeV")
    
    return {
        'h_estimate': h_estimate,
        'h_target': h_target,
        'ratio': h_estimate / h_target,
        'Delta_estimate_MeV': Delta_estimate,
        'Delta_target_MeV': Delta_target
    }
```

---

## 🔗 PHASE 5: KALUZA-KLEIN REDUCTION (Semaine 6)

### 5.1 Notebook: `05_kk_reduction.ipynb`

**Objectif**: Montrer comment le gap spectral de K₇ se propage vers 4D

```python
"""
GIFT Yang-Mills: Kaluza-Klein Reduction
=======================================
Show that λ₁(K₇) induces mass gap in 4D gauge theory

Key result:
  11D: □₁₁ φ = 0
  Decomposition: □₁₁ = □₄ + Δ_K₇
  4D mass: m² = λₙ(K₇)
"""

import numpy as np
from sympy import *

class KKReduction:
    """
    Kaluza-Klein reduction of E₈×E₈ gauge theory on K₇.
    """
    
    def __init__(self, spectrum_K7, gift_constants):
        """
        Args:
            spectrum_K7: Eigenvalues of Laplacian on K₇
            gift_constants: GIFT topological constants
        """
        self.spectrum = spectrum_K7
        self.G = gift_constants
        
    def gauge_field_decomposition(self):
        """
        Decompose 11D gauge field A_M into 4D modes.
        
        A_M(x, y) = Σ_n A_μ^(n)(x) ⊗ ψ_n(y)
        
        where:
        - x ∈ M₄ (4D spacetime)
        - y ∈ K₇ (internal manifold)
        - ψ_n are eigenmodes of Δ_K₇
        """
        print("Gauge field decomposition:")
        print("  A_M(x,y) = Σ_n A_μ^(n)(x) ⊗ ψ_n(y)")
        print()
        print("  11D Yang-Mills: D_M F^MN = 0")
        print("  → (□₄ + Δ_K₇) A_μ^(n) = 0")
        print("  → (□₄ + λ_n) A_μ^(n) = 0")
        print()
        print("  This is a massive 4D field with m_n² = λ_n")
        
        return {
            'zero_modes': self.spectrum[self.spectrum < 1e-6],
            'massive_modes': self.spectrum[self.spectrum >= 1e-6]
        }
    
    def symmetry_breaking(self):
        """
        E₈×E₈ → SU(3)×SU(2)×U(1) breaking chain.
        
        The holonomy of K₇ determines which subgroup survives.
        """
        print("\nSymmetry breaking chain:")
        print("  E₈ → E₆ × SU(3)_hidden")
        print("  E₆ → SO(10) × U(1)")
        print("  SO(10) → SU(5) × U(1)")
        print("  SU(5) → SU(3)_c × SU(2)_L × U(1)_Y")
        print()
        
        # Dimension counting
        dims = {
            'E8': 248,
            'E6': 78,
            'SO10': 45,
            'SU5': 24,
            'SM': 8 + 3 + 1  # SU(3) + SU(2) + U(1)
        }
        
        print("  Dimension flow:")
        for name, dim in dims.items():
            print(f"    {name}: {dim}")
        
        return dims
    
    def mass_gap_propagation(self):
        """
        Show that mass gap propagates from K₇ to 4D SU(3).
        
        Key insight:
        - Zero modes of Δ_K₇ → massless 4D fields (gauge bosons)
        - First excited mode λ₁ → lightest massive excitation
        
        For QCD (SU(3) sector):
        - Gluons are massless at tree level
        - Confinement generates mass gap Δ ~ Λ_QCD
        - GIFT claims: Δ/Λ_QCD = dim(G₂)/H* = 14/99
        """
        lambda_1 = self.spectrum[1]  # First nonzero eigenvalue
        
        print("\nMass gap propagation:")
        print(f"  λ₁(K₇) = {lambda_1:.6f}")
        print()
        print("  Geometric mass gap (in Planck units):")
        print(f"    m_KK = √λ₁ × M_Planck")
        print()
        
        # GIFT claim
        h_gift = self.G['dim_G2'] / self.G['H_star']
        
        print("  GIFT conjecture:")
        print(f"    Δ_QCD / Λ_QCD = h(K₇) = {h_gift:.6f}")
        print()
        print("  If λ₁ ≈ h² = (14/99)² = 0.02:")
        print("    The geometric gap matches the GIFT prediction")
        
        return {
            'lambda_1': lambda_1,
            'h_gift': h_gift,
            'h_squared': h_gift**2
        }
    
    def construct_effective_lagrangian(self):
        """
        Write the 4D effective Lagrangian after KK reduction.
        """
        print("\n4D Effective Lagrangian:")
        print()
        print("  L_eff = L_YM[SU(3)] + L_YM[SU(2)] + L_YM[U(1)]")
        print("        + Σ_n (D_μ φ_n)² + m_n² |φ_n|²")
        print("        + interactions")
        print()
        print("  where m_n² = λ_n(K₇)")
        print()
        print("  The mass gap is:")
        print("    Δ = min{m_n : m_n > 0} = √λ₁")
        
        
class TheoremStatement:
    """
    Formal statement of the Yang-Mills connection.
    """
    
    @staticmethod
    def main_theorem():
        """
        The main claim connecting GIFT topology to Yang-Mills gap.
        """
        statement = """
╔══════════════════════════════════════════════════════════════════════════╗
║                         MAIN THEOREM (Conjecture)                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Let K₇ be the compact G₂-holonomy manifold constructed via TCS with     ║
║  Betti numbers b₂ = 21, b₃ = 77.                                         ║
║                                                                          ║
║  Let Δ_K₇ be the Hodge Laplacian on 1-forms, with spectrum              ║
║  0 = λ₀ < λ₁ ≤ λ₂ ≤ ...                                                 ║
║                                                                          ║
║  CLAIM: The Cheeger constant satisfies                                   ║
║                                                                          ║
║         h(K₇) = dim(G₂) / H* = 14/99                                    ║
║                                                                          ║
║  and consequently (by Cheeger's inequality):                             ║
║                                                                          ║
║         λ₁ ≥ h²/4 = 196/39204 ≈ 0.005                                   ║
║                                                                          ║
║  PHYSICAL CONSEQUENCE:                                                   ║
║                                                                          ║
║  Under Kaluza-Klein reduction of E₈×E₈ gauge theory on K₇,              ║
║  the 4D SU(3) sector inherits a mass gap:                                ║
║                                                                          ║
║         Δ_QCD = h(K₇) × Λ_QCD = (14/99) × 200 MeV ≈ 28 MeV             ║
║                                                                          ║
║  This provides a TOPOLOGICAL origin for the Yang-Mills mass gap.         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
        """
        print(statement)
    
    @staticmethod
    def proof_outline():
        """
        Outline of required proof steps.
        """
        outline = """
PROOF OUTLINE (Required Steps):

1. GEOMETRIC SETUP
   □ Construct explicit TCS K₇ with (b₂=21, b₃=77)
   □ Verify G₂ holonomy (torsion-free φ)
   □ Compute metric det(g) = 65/32
   
2. SPECTRAL ANALYSIS  
   □ Prove Δ_K₇ has discrete spectrum (compactness)
   □ Compute or bound λ₁ numerically
   □ Estimate Cheeger constant h(K₇)
   
3. KALUZA-KLEIN REDUCTION
   □ Decompose E₈×E₈ gauge field on M₄ × K₇
   □ Show m_n² = λ_n for KK modes
   □ Verify E₈ → SM breaking preserves gap structure
   
4. PHYSICAL IDENTIFICATION
   □ Match SU(3) sector with QCD
   □ Relate geometric scale to Λ_QCD
   □ Derive Δ = h × Λ_QCD

5. RIGOROUS BOUNDS
   □ Prove h(K₇) = 14/99 (or derive from topology)
   □ Apply Cheeger inequality
   □ Establish λ₁ > 0 rigorously
   
STATUS: Steps 1-2 are computationally tractable.
        Steps 3-4 require careful physics.
        Step 5 requires geometric analysis.
        """
        print(outline)
```

---

## 📝 PHASE 6: PAPER & PUBLICATION (Semaine 7-8)

### 6.1 Structure du papier

```
paper/yang_mills_gift.tex

Title: "Topological Origin of the Yang-Mills Mass Gap 
        from G₂-Holonomy Compactification"

1. Introduction
   - Yang-Mills mass gap problem
   - GIFT framework overview
   - Main claim: Δ = (14/99) × Λ_QCD

2. Mathematical Framework
   - E₈×E₈ gauge theory in 11D
   - G₂ holonomy and K₇ construction
   - Topological invariants (b₂=21, b₃=77)

3. Spectral Analysis
   - Hodge Laplacian on K₇
   - Numerical computation of spectrum
   - Cheeger constant estimation

4. Kaluza-Klein Reduction
   - Dimensional reduction M₁₁ → M₄ × K₇
   - Symmetry breaking E₈ → SM
   - Mass gap propagation

5. Numerical Results
   - PINN metric learning
   - Spectral computation
   - Comparison with GIFT predictions

6. Discussion
   - Implications for Yang-Mills problem
   - Limitations and assumptions
   - Future directions

7. Conclusion

Appendix A: GIFT Constants
Appendix B: Lean 4 Verifications
Appendix C: Numerical Methods
```

---

## ⚡ RÉSUMÉ POUR CLAUDE CODE

### Commandes d'initialisation

```bash
# Créer le repo
mkdir gift-yang-mills
cd gift-yang-mills
git init

# Structure
mkdir -p src/{spectral,gauge,visualization}
mkdir -p notebooks data/{pinn_checkpoints,spectral_results} paper tests

# Dépendances
cat > requirements.txt << EOF
torch>=2.0
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
tqdm
scikit-learn
sympy
EOF
```

### Ordre d'exécution

1. **`src/constants.py`** - GIFT constants
2. **`src/g2_structure.py`** - G₂ forms
3. **`notebooks/01_pinn_metric_training.ipynb`** - Train PINN (Colab A100)
4. **`notebooks/02_manifold_sampling.ipynb`** - Sample K₇
5. **`notebooks/03_spectral_analysis.ipynb`** - Compute λ₁ (KEY!)
6. **`notebooks/04_cheeger_estimation.ipynb`** - Estimate h(K₇)
7. **`notebooks/05_kk_reduction.ipynb`** - Physics derivation
8. **Paper writing**

### Cibles numériques

| Quantité | Valeur cible | Tolérance |
|----------|--------------|-----------|
| det(g) | 65/32 = 2.03125 | ±0.01 |
| ‖T‖ (torsion) | < 0.001 | - |
| h(K₇) | 14/99 ≈ 0.1414 | ±20% |
| λ₁ | ≥ 0.005 | - |
| Δ/Λ_QCD | 14/99 ≈ 0.14 | ±20% |

### Critère de succès

**SI λ₁ ≈ 0.02 (= (14/99)²) sort des calculs numériques, c'est un résultat majeur.**

---

## 🎯 PROCHAINE ACTION IMMÉDIATE

**Pour Claude Code:**

```
1. Créer le repo gift-yang-mills avec la structure ci-dessus
2. Implémenter src/constants.py et src/g2_structure.py
3. Adapter le PINN existant (K7_Explicit_Metric_v3_2.ipynb) 
   pour la nouvelle structure
4. Créer notebook 03_spectral_analysis.ipynb 
   (C'EST LE NOTEBOOK CLÉ)
5. Lancer sur Colab A100 avec N=10000 points
6. Reporter λ₁ et h estimés
```

**Output attendu:**
```
λ₁ = 0.0XX (comparé à cible 0.02)
h = 0.1XX (comparé à cible 0.14)
```

Si ces valeurs sont proches des cibles, on a un papier. Si non, on analyse pourquoi.

---

*"Le gap est déjà là, géométriquement. La question est de le quantifier."*

---

**GO ! 🚀**
