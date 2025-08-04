#!/usr/bin/env python3
"""
Example script demonstrating how to use the modified quantization script
with PyTorch's built-in quantization API.
"""

import subprocess
import sys
import os

def run_quantization_test(quantization_method="dynamic", bitwidth=8):
    """
    Run the quantization test with specified parameters.
    """
    print(f"Testing {quantization_method} quantization with {bitwidth}-bit precision...")
    
    # Example command - modify paths as needed
    cmd = [
        "python", "test_real_quant.py",
        "--model_path", "/path/to/your/model.pth",
        "--data_dir", "/path/to/your/dataset",
        "--krt_dir", "/path/to/your/krt",
        "--framelist_test", "/path/to/your/frame_list.txt",
        "--result_path", f"./results/{quantization_method}_{bitwidth}bit",
        "--quantization_method", quantization_method,
        "--bitwidth", str(bitwidth),
        "--calibration_batches", "50",
        "--val_batch_size", "4",
        "--save_img"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running quantization test: {e}")
        return False

def main():
    """
    Run quantization tests with different configurations.
    """
    print("PyTorch Quantization Test Suite")
    print("=" * 40)
    
    # Test configurations
    test_configs = [
        ("dynamic", 8),
        ("dynamic", 16),
        ("static", 8),
    ]
    
    results = {}
    
    for method, bitwidth in test_configs:
        print(f"\n{'='*20} Testing {method.upper()} {bitwidth}-bit {'='*20}")
        success = run_quantization_test(method, bitwidth)
        results[f"{method}_{bitwidth}bit"] = success
        
        if success:
            print(f"✅ {method} {bitwidth}-bit quantization test PASSED")
        else:
            print(f"❌ {method} {bitwidth}-bit quantization test FAILED")
    
    # Summary
    print(f"\n{'='*20} Test Summary {'='*20}")
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")

if __name__ == "__main__":
    main() 