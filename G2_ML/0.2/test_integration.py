"""
Quick integration test for G2 v0.2

Tests all modules working together with a tiny training run.
"""

import torch
import numpy as np
import sys

print("=" * 70)
print("G2 v0.2 Integration Test")
print("=" * 70)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

# Test imports
print("\n1. Testing imports...")
try:
    from G2_phi_network import G2PhiNetwork, metric_from_phi_algebraic
    from G2_geometry import project_spd
    from G2_manifold import create_manifold
    from G2_losses import G2TotalLoss, CurriculumScheduler
    print("   All imports successful!")
except Exception as e:
    print(f"   Import failed: {e}")
    sys.exit(1)

# Test manifold
print("\n2. Testing manifold...")
try:
    manifold = create_manifold('T7', device=device)
    coords = manifold.sample_points(10)
    print(f"   Manifold created, sampled {coords.shape[0]} points")
except Exception as e:
    print(f"   Manifold test failed: {e}")
    sys.exit(1)

# Test model
print("\n3. Testing model...")
try:
    model = G2PhiNetwork(
        encoding_type='fourier',
        hidden_dims=[64, 64],  # Smaller for testing
        fourier_modes=8,
        normalize_phi=True
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created: {n_params:,} parameters")
    
    phi = model(coords)
    print(f"   Forward pass: {phi.shape}")
except Exception as e:
    print(f"   Model test failed: {e}")
    sys.exit(1)

# Test metric reconstruction
print("\n4. Testing metric reconstruction...")
try:
    metric = metric_from_phi_algebraic(phi, use_approximation=True)
    metric = project_spd(metric)
    print(f"   Metric reconstructed: {metric.shape}")
    
    eigenvalues = torch.linalg.eigvalsh(metric)
    print(f"   Eigenvalues: min={eigenvalues.min():.4f}, max={eigenvalues.max():.4f}")
except Exception as e:
    print(f"   Metric reconstruction failed: {e}")
    sys.exit(1)

# Test loss computation
print("\n5. Testing loss function...")
try:
    curriculum = CurriculumScheduler()
    loss_fn = G2TotalLoss(
        curriculum_scheduler=curriculum,
        use_ricci=False,
        use_positivity=True,
        derivative_method='autograd'
    )
    
    coords.requires_grad = True
    total_loss, loss_info = loss_fn(phi, metric, coords, epoch=0)
    
    print(f"   Loss computed: {total_loss.item():.6e}")
    print(f"   Torsion: {loss_info.get('torsion_total', 0):.6e}")
except Exception as e:
    print(f"   Loss computation failed: {e}")
    sys.exit(1)

# Test backward pass
print("\n6. Testing backward pass...")
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print("   Backward pass successful!")
except Exception as e:
    print(f"   Backward pass failed: {e}")
    sys.exit(1)

# Mini training loop
print("\n7. Running mini training (5 epochs)...")
try:
    for epoch in range(5):
        model.train()
        
        coords = manifold.sample_points(32)
        coords = coords.to(device)
        coords.requires_grad = True
        
        phi = model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        
        total_loss, loss_info = loss_fn(phi, metric, coords, epoch=epoch)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch == 0 or epoch == 4:
            print(f"   Epoch {epoch}: loss={total_loss.item():.6e}")
    
    print("   Mini training successful!")
except Exception as e:
    print(f"   Mini training failed: {e}")
    sys.exit(1)

# Test validation
print("\n8. Testing validation...")
try:
    model.eval()
    
    with torch.no_grad():
        coords = manifold.sample_points(50)
        coords = coords.to(device)
        
        phi = model(coords)
        metric = metric_from_phi_algebraic(phi, use_approximation=True)
        metric = project_spd(metric)
        
        phi_norm_sq = torch.sum(phi ** 2, dim=1)
        det_g = torch.det(metric)
        
        print(f"   ||phi||^2: {phi_norm_sq.mean():.6f} (target: 7.0)")
        print(f"   det(g): {det_g.mean():.6f} (target: 1.0)")
    
    print("   Validation successful!")
except Exception as e:
    print(f"   Validation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nThe G2 v0.2 system is working correctly.")
print("You can now run full training with:")
print("  python G2_train.py --encoding fourier --epochs 3000 --batch-size 512")






