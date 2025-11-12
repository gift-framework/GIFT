"""
G2 Export Module - v0.2

Export trained G2 models to ONNX format for deployment.

Usage:
    python G2_export.py --model checkpoints/final_model.pt --output g2_metric.onnx

Author: GIFT Project
No Unicode - Windows compatible
"""

import argparse
import os
import torch
import numpy as np

from G2_phi_network import G2PhiNetwork


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(model, output_path, input_shape=(1, 7), opset_version=14, device='cpu'):
    """
    Export G2 model to ONNX format.
    
    Args:
        model: Trained G2PhiNetwork
        output_path: Path to save ONNX model
        input_shape: Shape of input tensor (batch_size, 7)
        opset_version: ONNX opset version
        device: Device
    
    Returns:
        success: Boolean indicating success
    """
    print(f"\nExporting model to ONNX...")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")
    print(f"  Output path: {output_path}")
    
    try:
        model.eval()
        model = model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape, device=device) * 2 * np.pi
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['coordinates'],
            output_names=['phi'],
            dynamic_axes={
                'coordinates': {0: 'batch_size'},
                'phi': {0: 'batch_size'}
            }
        )
        
        print(f"\nONNX export successful!")
        
        # Verify the exported model
        try:
            import onnx
            import onnxruntime as ort
            
            # Check model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("  ONNX model verification passed")
            
            # Test inference
            ort_session = ort.InferenceSession(output_path)
            
            # Run inference
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            # Compare with PyTorch
            with torch.no_grad():
                pytorch_output = model(dummy_input).cpu().numpy()
            
            diff = np.abs(pytorch_output - ort_outputs[0]).max()
            print(f"  Max difference between PyTorch and ONNX: {diff:.6e}")
            
            if diff < 1e-5:
                print("  ONNX inference test passed!")
            else:
                print("  Warning: Large difference between PyTorch and ONNX outputs")
        
        except ImportError:
            print("  Note: Install onnx and onnxruntime for verification")
        
        return True
        
    except Exception as e:
        print(f"\nError during ONNX export: {e}")
        return False


# ============================================================================
# Model Checkpointing Utilities
# ============================================================================

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
    
    Returns:
        model: Loaded G2PhiNetwork
        config: Configuration dict
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {
            'encoding_type': 'fourier',
            'hidden_dims': [256, 256, 128],
            'fourier_modes': 16,
            'fourier_scale': 1.0,
            'omega_0': 30.0,
            'normalize_phi': True
        }
        print("Warning: No config in checkpoint, using defaults")
    
    # Create model
    model = G2PhiNetwork(
        encoding_type=config['encoding_type'],
        hidden_dims=config['hidden_dims'],
        fourier_modes=config.get('fourier_modes', 16),
        fourier_scale=config.get('fourier_scale', 1.0),
        omega_0=config.get('omega_0', 30.0),
        normalize_phi=config.get('normalize_phi', True)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.6e}")
    
    return model, config


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main export entry point."""
    parser = argparse.ArgumentParser(description='Export G2 model to ONNX')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pt)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for ONNX model (.onnx)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for export (cpu/cuda)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for dummy input')
    parser.add_argument('--opset-version', type=int, default=14,
                       help='ONNX opset version')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("G2 Model Export to ONNX - v0.2")
    print("=" * 70)
    
    # Load model
    model, config = load_model_from_checkpoint(args.model, device=args.device)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel information:")
    print(f"  Encoding: {config['encoding_type']}")
    print(f"  Hidden dims: {config['hidden_dims']}")
    print(f"  Parameters: {n_params:,}")
    
    # Export to ONNX
    success = export_to_onnx(
        model,
        args.output,
        input_shape=(args.batch_size, 7),
        opset_version=args.opset_version,
        device=args.device
    )
    
    if success:
        file_size = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\nONNX model saved: {args.output}")
        print(f"File size: {file_size:.2f} MB")
    
    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()






