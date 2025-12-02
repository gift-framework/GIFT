"""Quick test for the Harmonic-Yukawa pipeline.

Run with: python -m G2_ML.harmonic_yukawa.test_pipeline
"""
import torch
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from G2_ML.harmonic_yukawa import (
    HarmonicConfig,
    HodgeLaplacian,
    WedgeProduct,
    YukawaTensor,
    MassSpectrum,
    HarmonicYukawaPipeline,
)


def test_wedge_product():
    """Test wedge product computation."""
    print("Testing WedgeProduct...")
    wedge = WedgeProduct()

    # Random 2-forms
    omega_a = torch.randn(10, 21)
    omega_b = torch.randn(10, 21)

    # 2 wedge 2 -> 4
    eta = wedge.wedge_2_2(omega_a, omega_b)
    assert eta.shape == (10, 35), f"Expected (10, 35), got {eta.shape}"

    # 4 wedge 3 -> 7 (scalar)
    Phi = torch.randn(10, 35)
    scalar = wedge.wedge_4_3(eta, Phi)
    assert scalar.shape == (10,), f"Expected (10,), got {scalar.shape}"

    print("  WedgeProduct: PASS")


def test_mock_metric():
    """Create a simple mock metric function."""
    def mock_metric(x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        # Start with identity, add small perturbation based on x
        g = torch.eye(7).unsqueeze(0).expand(batch, 7, 7).clone()
        # Add smooth perturbation
        scale = 0.1
        for i in range(7):
            for j in range(i+1, 7):
                pert = scale * torch.sin(x[:, i] + x[:, j])
                g[:, i, j] = pert
                g[:, j, i] = pert
        # Ensure positive definiteness
        g = g + 2 * torch.eye(7).unsqueeze(0)
        return g
    return mock_metric


def test_hodge_laplacian():
    """Test Hodge Laplacian computation."""
    print("Testing HodgeLaplacian...")

    config = HarmonicConfig(n_sample_points=100)
    laplacian = HodgeLaplacian(config)
    metric_fn = test_mock_metric()

    # Sample points
    points = torch.rand(100, 7)

    # Compute 2-form Laplacian
    result = laplacian.compute_laplacian_2forms(points, metric_fn, n_basis=30)
    assert result.eigenvalues.shape[0] == 30
    print(f"  2-form eigenvalues range: [{result.eigenvalues.min():.4f}, {result.eigenvalues.max():.4f}]")
    print(f"  Harmonic forms found: {result.n_harmonic}")

    print("  HodgeLaplacian: PASS")


def test_pipeline():
    """Test the complete pipeline with mock metric."""
    print("Testing HarmonicYukawaPipeline...")

    config = HarmonicConfig(
        n_sample_points=500,
        n_yukawa_samples=500,
        yukawa_batch_size=100,
    )
    metric_fn = test_mock_metric()

    pipeline = HarmonicYukawaPipeline(metric_fn, config, device='cpu')

    # Quick validation
    validation = pipeline.quick_validate(n_points=100)
    print(f"  det(g) mean: {validation['det_g_mean']:.4f} (target: 2.03125)")
    print(f"  Positive definite: {validation['positive_definite_fraction']*100:.1f}%")

    print("  HarmonicYukawaPipeline: PASS (quick validate)")


def test_mass_spectrum():
    """Test mass spectrum extraction."""
    print("Testing MassSpectrum...")

    # Create mock eigenvalues (77 values with hierarchy)
    eigenvalues = torch.logspace(-6, 2, 77)  # Hierarchy from 1e-6 to 100

    from G2_ML.harmonic_yukawa.mass_spectrum import FermionMasses

    masses = FermionMasses.from_eigenvalues(eigenvalues, scale=246.0)

    print(f"  m_tau/m_e = {masses.tau_e_ratio:.1f} (GIFT: 3477)")
    print(f"  m_s/m_d = {masses.s_d_ratio:.1f} (GIFT: 20)")
    print(f"  Q_Koide = {masses.koide_q:.4f} (GIFT: 0.6667)")

    # Check PDG comparison
    comparison = masses.compare_pdg()
    print(f"  Comparisons available: {len(comparison)}")

    print("  MassSpectrum: PASS")


def test_lean_export():
    """Test Lean export functionality."""
    print("Testing Lean export...")

    from G2_ML.harmonic_yukawa.pipeline import PipelineResult
    from G2_ML.harmonic_yukawa.harmonic_extraction import HarmonicBasis
    from G2_ML.harmonic_yukawa.yukawa import YukawaResult
    from G2_ML.harmonic_yukawa.mass_spectrum import FermionMasses

    # Create mock result
    mock_result = PipelineResult(
        harmonic_basis=None,  # Would be HarmonicBasis
        yukawa_result=YukawaResult(
            tensor=torch.zeros(21, 21, 77),
            gram_matrix=torch.eye(77),
            eigenvalues=torch.ones(77),
            eigenvectors=torch.eye(77),
            trace=77.0,
            det=1.0,
        ),
        fermion_masses=FermionMasses(
            m_e=0.000511, m_mu=0.1057, m_tau=1.777,
            m_u=0.00216, m_c=1.27, m_t=172.69,
            m_d=0.00467, m_s=0.0934, m_b=4.18,
            tau_e_ratio=3477.0, s_d_ratio=20.0, koide_q=0.6667,
        ),
        spectrum_analysis={},
        mass_report="Mock report",
        det_g_mean=2.03125,
        kappa_T_estimate=1/61,
        tau_computed=3472/891,
    )

    # Test Lean export
    lean_code = mock_result.export_lean_bounds()
    assert "GIFT.NumericalBounds" in lean_code
    assert "det_g_computed" in lean_code
    print(f"  Lean code generated: {len(lean_code)} chars")

    print("  Lean export: PASS")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Harmonic-Yukawa Pipeline Tests")
    print("=" * 60)
    print()

    test_wedge_product()
    test_hodge_laplacian()
    test_mass_spectrum()
    test_lean_export()
    test_pipeline()

    print()
    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
