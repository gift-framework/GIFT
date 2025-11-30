"""
G2 Metric ONNX Export Script

Export trained PyTorch model to ONNX format for cross-platform inference.
No Unicode - Windows compatible.

Usage:
    python G2_export_onnx.py --model G2_final_model.pt --output G2_metric.onnx
"""

import argparse
import torch
import numpy as np
from G2_phi_wrapper import load_model


def export_to_onnx(model, output_path, opset_version=11):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save ONNX file
        opset_version: ONNX opset version (default 11 for compatibility)
    """
    model.eval()
    
    # Create dummy input for tracing
    batch_size = 1
    dummy_input = torch.randn(batch_size, 7, requires_grad=False)
    
    print(f"\nExporting model to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Opset version: {opset_version}")
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['coordinates'],
        output_names=['metric'],
        dynamic_axes={
            'coordinates': {0: 'batch_size'},
            'metric': {0: 'batch_size'}
        }
    )
    
    print(f"  Model exported to: {output_path}")


def verify_onnx_export(onnx_path, pytorch_model, n_test=10):
    """
    Verify ONNX export matches PyTorch model output.
    
    Args:
        onnx_path: Path to ONNX model file
        pytorch_model: Original PyTorch model
        n_test: Number of test points
    
    Returns:
        bool: True if outputs match within tolerance
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("\nWarning: onnx or onnxruntime not installed")
        print("Install with: pip install onnx onnxruntime")
        return None
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Create ONNX runtime session
    ort_session = ort.InferenceSession(onnx_path)
    
    print(f"\nVerifying ONNX export with {n_test} test points...")
    
    # Test on random points
    test_coords = torch.randn(n_test, 7)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_coords).numpy()
    
    # ONNX inference
    onnx_output = ort_session.run(
        None,
        {'coordinates': test_coords.numpy()}
    )[0]
    
    # Compare
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"  Max difference:  {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    
    tolerance = 1e-5
    match = max_diff < tolerance
    
    if match:
        print(f"  Status: PASS (within tolerance {tolerance:.0e})")
    else:
        print(f"  Status: FAIL (exceeds tolerance {tolerance:.0e})")
    
    return match


def test_onnx_inference(onnx_path, test_point=None):
    """
    Test ONNX model inference.
    
    Args:
        onnx_path: Path to ONNX model
        test_point: Optional test point (7D array)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nError: onnxruntime not installed")
        print("Install with: pip install onnxruntime")
        return
    
    # Create session
    ort_session = ort.InferenceSession(onnx_path)
    
    # Get input/output info
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    print(f"\nONNX Model Info:")
    print(f"  Input name: {input_name}")
    print(f"  Output name: {output_name}")
    
    # Test inference
    if test_point is None:
        test_point = np.random.randn(1, 7).astype(np.float32)
    else:
        test_point = np.array(test_point).reshape(1, 7).astype(np.float32)
    
    print(f"\nTest inference at point: {test_point[0]}")
    
    result = ort_session.run(None, {input_name: test_point})
    metric = result[0]
    
    print(f"  Output shape: {metric.shape}")
    print(f"  Determinant: {np.linalg.det(metric[0]):.6f}")
    print(f"  Min eigenvalue: {np.linalg.eigvalsh(metric[0]).min():.6f}")
    print(f"  Max eigenvalue: {np.linalg.eigvalsh(metric[0]).max():.6f}")


def main():
    parser = argparse.ArgumentParser(description='Export G2 metric to ONNX')
    parser.add_argument('--model', type=str, default='G2_final_model.pt',
                        help='Path to PyTorch model')
    parser.add_argument('--output', type=str, default='G2_metric.onnx',
                        help='Path for ONNX output')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ONNX export matches PyTorch')
    parser.add_argument('--test', action='store_true',
                        help='Test ONNX inference')
    parser.add_argument('--test-point', type=str, default=None,
                        help='Test point for ONNX inference (comma-separated)')
    
    args = parser.parse_args()
    
    print("\nG2 Metric ONNX Export")
    print("=" * 70)
    print(f"PyTorch model: {args.model}")
    print(f"ONNX output: {args.output}")
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    model = load_model(args.model, device='cpu')
    print("Model loaded successfully!")
    
    # Export to ONNX
    export_to_onnx(model, args.output, opset_version=args.opset)
    
    # Verify if requested
    if args.verify:
        match = verify_onnx_export(args.output, model)
        if match is False:
            print("\nWarning: ONNX export verification failed!")
            return 1
    
    # Test ONNX inference if requested
    if args.test:
        test_point = None
        if args.test_point:
            try:
                test_point = [float(x) for x in args.test_point.split(',')]
                if len(test_point) != 7:
                    print("\nError: Test point must have 7 coordinates")
                    return 1
            except ValueError:
                print("\nError: Invalid test point format")
                return 1
        
        test_onnx_inference(args.output, test_point=test_point)
    
    print("\n" + "=" * 70)
    print("Export complete!")
    print("\nUsage examples:")
    print(f"  Python: import onnxruntime; session = onnxruntime.InferenceSession('{args.output}')")
    print(f"  C++: Use ONNX Runtime C++ API")
    print(f"  JavaScript: Use onnxruntime-web")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    exit(main())









