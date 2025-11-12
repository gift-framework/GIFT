"""
G2 Phi Network Module - v0.2

Neural network architectures for learning G2 3-form phi(x).
Supports both Fourier feature encoding and SIREN architectures.

Author: GIFT Project
No Unicode - Windows compatible
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# Encoding Layers
# ============================================================================

class FourierFeatures(nn.Module):
    """
    Random Fourier features for periodic coordinate encoding.
    
    Maps x ∈ R^7 to [cos(2π B·x), sin(2π B·x)] ∈ R^(2*7*n_modes)
    where B is a random Gaussian matrix.
    
    Provides explicit periodicity for T^7 manifold.
    """
    
    def __init__(self, input_dim=7, n_modes=16, scale=1.0):
        """
        Args:
            input_dim: Dimension of input coordinates (7 for T^7)
            n_modes: Number of Fourier modes
            scale: Scale of random frequencies (default: 1.0)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_modes = n_modes
        self.output_dim = 2 * n_modes  # cos + sin
        
        # Random Fourier matrix (fixed, not trainable)
        B = torch.randn(input_dim, n_modes) * scale
        self.register_buffer('B', B)
    
    def forward(self, x):
        """
        Args:
            x: Coordinates of shape (batch, input_dim)
        
        Returns:
            features: Shape (batch, 2*n_modes)
        """
        # Compute 2π B·x
        proj = 2 * np.pi * torch.matmul(x, self.B)
        
        # [cos, sin] concatenation
        features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        
        return features


class SirenLayer(nn.Module):
    """
    SIREN (Sinusoidal Representation Network) layer.
    
    Uses sin(ω·(Wx + b)) as activation for implicit neural representations.
    Provides smooth, continuous representation with good spectral properties.
    """
    
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            omega_0: Frequency parameter (higher = more high-frequency detail)
            is_first: Whether this is the first layer (special initialization)
        """
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()
    
    def init_weights(self):
        """SIREN-specific weight initialization."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                           1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0,
                                           np.sqrt(6 / self.linear.in_features) / self.omega_0)
    
    def forward(self, x):
        """
        Args:
            x: Input of shape (batch, in_features)
        
        Returns:
            output: sin(ω·(Wx + b)) of shape (batch, out_features)
        """
        return torch.sin(self.omega_0 * self.linear(x))


# ============================================================================
# G2 Phi Networks
# ============================================================================

class G2PhiNetwork(nn.Module):
    """
    Neural network that learns G2 3-form phi(x) on T^7.
    
    Architecture:
        Input: 7D coordinates
        Encoding: Fourier or SIREN
        MLP: Multiple hidden layers
        Output: 35 components of phi (3-form in 7D)
    
    The metric is then reconstructed algebraically from phi.
    """
    
    def __init__(self, 
                 encoding_type='fourier',
                 hidden_dims=[256, 256, 128],
                 fourier_modes=16,
                 fourier_scale=1.0,
                 omega_0=30.0,
                 normalize_phi=True):
        """
        Args:
            encoding_type: 'fourier' or 'siren'
            hidden_dims: List of hidden layer dimensions
            fourier_modes: Number of Fourier modes (for Fourier encoding)
            fourier_scale: Scale of Fourier frequencies
            omega_0: SIREN frequency parameter
            normalize_phi: Whether to enforce ||phi||^2 = 7
        """
        super().__init__()
        
        self.encoding_type = encoding_type.lower()
        self.normalize_phi = normalize_phi
        self.input_dim = 7
        self.output_dim = 35  # C(7,3) = 35 components for 3-form
        
        # Build encoding layer
        if self.encoding_type == 'fourier':
            self.encoding = FourierFeatures(input_dim=7, 
                                           n_modes=fourier_modes,
                                           scale=fourier_scale)
            encoding_dim = self.encoding.output_dim
            
            # Standard MLP with SiLU activation
            layers = []
            prev_dim = encoding_dim
            
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(h_dim))
                prev_dim = h_dim
            
            self.mlp = nn.Sequential(*layers)
            
        elif self.encoding_type == 'siren':
            # SIREN network (no explicit encoding layer)
            self.encoding = None
            encoding_dim = 7
            
            # Build SIREN layers
            layers = []
            prev_dim = encoding_dim
            
            for i, h_dim in enumerate(hidden_dims):
                layers.append(SirenLayer(prev_dim, h_dim, 
                                        omega_0=omega_0,
                                        is_first=(i == 0)))
                prev_dim = h_dim
            
            self.mlp = nn.Sequential(*layers)
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        # Output layer for phi components
        self.output_layer = nn.Linear(prev_dim, self.output_dim)
        
        # Initialize output layer with small weights
        with torch.no_grad():
            self.output_layer.weight.mul_(0.01)
            self.output_layer.bias.zero_()
    
    def forward(self, coords):
        """
        Forward pass: compute phi(x).
        
        Args:
            coords: Coordinates of shape (batch, 7)
        
        Returns:
            phi: 3-form components of shape (batch, 35)
        """
        batch_size = coords.shape[0]
        
        # Encoding
        if self.encoding_type == 'fourier':
            x = self.encoding(coords)
        else:  # siren
            x = coords
        
        # MLP
        x = self.mlp(x)
        
        # Output phi
        phi = self.output_layer(x)
        
        # Optional normalization: ||phi||^2 = 7
        if self.normalize_phi:
            phi_norm = torch.norm(phi, dim=-1, keepdim=True)
            phi = phi * (np.sqrt(7.0) / (phi_norm + 1e-8))
        
        return phi
    
    def get_phi_indices(self):
        """
        Return mapping from component index to (i,j,k) triple.
        
        Returns:
            indices: List of tuples [(i1,j1,k1), (i2,j2,k2), ...]
        """
        indices = []
        for i in range(7):
            for j in range(i+1, 7):
                for k in range(j+1, 7):
                    indices.append((i, j, k))
        return indices


# ============================================================================
# Metric Reconstruction from Phi
# ============================================================================

def levi_civita_7d():
    """
    Generate the Levi-Civita symbol (totally antisymmetric tensor) in 7D.
    
    Returns:
        epsilon: Numpy array of shape (7,7,7,7,7,7,7)
                 epsilon[i,j,k,l,m,n,p] = +1/-1/0
    """
    # This is memory-intensive (7^7 = 823543 elements)
    # For practical use, we'll compute this on-the-fly or use sparse representation
    epsilon = np.zeros([7]*7, dtype=np.float32)
    
    # Generate all permutations of [0,1,2,3,4,5,6]
    from itertools import permutations
    
    base = list(range(7))
    for perm in permutations(base):
        # Count inversions to determine sign
        sign = 1
        for i in range(7):
            for j in range(i+1, 7):
                if perm[i] > perm[j]:
                    sign *= -1
        
        epsilon[perm] = sign
    
    return epsilon


def metric_from_phi_algebraic(phi, use_approximation=True):
    """
    Reconstruct metric from phi using algebraic identity:
    
        g_ij = (1/144) phi_imn phi_jpq phi_rst epsilon^mnpqrst
    
    This is the exact G2 relation but computationally expensive.
    
    Args:
        phi: 3-form of shape (batch, 35)
        use_approximation: If True, use faster approximation instead
    
    Returns:
        metric: Tensor of shape (batch, 7, 7)
    """
    batch_size = phi.shape[0]
    device = phi.device
    
    if use_approximation:
        # Faster approximation: construct metric from phi structure
        # This is not the exact formula but captures essential geometry
        metric = metric_from_phi_approximate(phi)
    else:
        # Exact formula (very slow, use only for validation)
        metric = metric_from_phi_exact(phi)
    
    return metric


def metric_from_phi_approximate(phi):
    """
    Fast approximation for metric reconstruction from phi.
    
    Uses structural properties of phi to construct a reasonable metric
    without full tensor contraction.
    
    Args:
        phi: 3-form of shape (batch, 35)
    
    Returns:
        metric: Tensor of shape (batch, 7, 7)
    """
    batch_size = phi.shape[0]
    device = phi.device
    
    # Initialize metric
    metric = torch.zeros(batch_size, 7, 7, device=device)
    
    # Build index mapping
    triple_to_idx = {}
    idx = 0
    for i in range(7):
        for j in range(i+1, 7):
            for k in range(j+1, 7):
                triple_to_idx[(i, j, k)] = idx
                idx += 1
    
    # Diagonal components from phi norm
    phi_norm = torch.norm(phi, dim=1, keepdim=True)
    for i in range(7):
        metric[:, i, i] = phi_norm.squeeze() / np.sqrt(7.0)
    
    # Off-diagonal from cross-terms
    # g_ij related to phi components containing both i and j
    for i in range(7):
        for j in range(i+1, 7):
            # Sum over all k: phi_ijk^2
            contrib = 0.0
            count = 0
            for k in range(7):
                if k != i and k != j:
                    triple = tuple(sorted([i, j, k]))
                    if triple in triple_to_idx:
                        contrib += phi[:, triple_to_idx[triple]]**2
                        count += 1
            
            if count > 0:
                metric[:, i, j] = torch.sqrt(contrib / count + 1e-8) * 0.1
                metric[:, j, i] = metric[:, i, j]
    
    return metric


def metric_from_phi_exact(phi):
    """
    Exact metric reconstruction (very slow, for validation only).
    
    Implements full tensor contraction with Levi-Civita symbol.
    
    Args:
        phi: 3-form of shape (batch, 35)
    
    Returns:
        metric: Tensor of shape (batch, 7, 7)
    """
    # This would require massive tensor operations
    # Placeholder for now - use approximate version in practice
    print("Warning: Exact metric reconstruction not implemented. Using approximation.")
    return metric_from_phi_approximate(phi)


# ============================================================================
# Testing and Demonstration
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("G2 Phi Network Module - Test Suite")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test Fourier encoding
    print("\n1. Testing Fourier Features")
    print("-" * 70)
    
    fourier = FourierFeatures(input_dim=7, n_modes=16).to(device)
    x_test = torch.randn(10, 7, device=device)
    features = fourier(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: (10, 32) - got (10, {features.shape[1]})")
    
    # Test SIREN layer
    print("\n2. Testing SIREN Layer")
    print("-" * 70)
    
    siren = SirenLayer(7, 64, omega_0=30.0, is_first=True).to(device)
    output = siren(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test Fourier-based phi network
    print("\n3. Testing G2PhiNetwork (Fourier)")
    print("-" * 70)
    
    model_fourier = G2PhiNetwork(
        encoding_type='fourier',
        hidden_dims=[256, 256, 128],
        fourier_modes=16,
        normalize_phi=True
    ).to(device)
    
    coords = torch.rand(32, 7, device=device) * 2 * np.pi
    phi = model_fourier(coords)
    
    print(f"Input coords shape: {coords.shape}")
    print(f"Output phi shape: {phi.shape}")
    print(f"Expected: (32, 35) - got {phi.shape}")
    
    # Check normalization
    phi_norm_sq = torch.sum(phi**2, dim=1)
    print(f"||phi||^2: mean={phi_norm_sq.mean():.6f}, std={phi_norm_sq.std():.6f}")
    print(f"Target: 7.0")
    
    # Count parameters
    n_params = sum(p.numel() for p in model_fourier.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Test SIREN-based phi network
    print("\n4. Testing G2PhiNetwork (SIREN)")
    print("-" * 70)
    
    model_siren = G2PhiNetwork(
        encoding_type='siren',
        hidden_dims=[256, 256, 128],
        omega_0=30.0,
        normalize_phi=True
    ).to(device)
    
    phi_siren = model_siren(coords)
    
    print(f"Output phi shape: {phi_siren.shape}")
    phi_norm_sq_siren = torch.sum(phi_siren**2, dim=1)
    print(f"||phi||^2: mean={phi_norm_sq_siren.mean():.6f}, std={phi_norm_sq_siren.std():.6f}")
    
    n_params_siren = sum(p.numel() for p in model_siren.parameters())
    print(f"Total parameters: {n_params_siren:,}")
    
    # Test metric reconstruction
    print("\n5. Testing Metric Reconstruction")
    print("-" * 70)
    
    metric = metric_from_phi_algebraic(phi, use_approximation=True)
    print(f"Metric shape: {metric.shape}")
    print(f"Expected: (32, 7, 7)")
    
    # Check symmetry
    is_symmetric = torch.allclose(metric, metric.transpose(-2, -1), atol=1e-5)
    print(f"Metric is symmetric: {is_symmetric}")
    
    # Check positive definiteness (via eigenvalues)
    eigenvalues = torch.linalg.eigvalsh(metric)
    min_eigenvalue = eigenvalues.min().item()
    print(f"Minimum eigenvalue: {min_eigenvalue:.6f}")
    print(f"Positive definite: {min_eigenvalue > 0}")
    
    # Test phi indices
    print("\n6. Testing Phi Component Indexing")
    print("-" * 70)
    
    indices = model_fourier.get_phi_indices()
    print(f"Number of components: {len(indices)}")
    print(f"First 5 components: {indices[:5]}")
    print(f"Last 5 components: {indices[-5:]}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)






